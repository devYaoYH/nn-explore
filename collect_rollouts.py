"""Collect free-generation rollouts on factual QA prompts (plain + distractor
context), auto-label each rollout as correct/incorrect against ground truth,
and record per-generated-token residual-stream activations at chosen layers.

This is the fix for the train/deploy mismatch diagnosed in steer.py: the
earlier "truthfulness direction" was trained on completed-statement or
forced-continuation activations and then applied live during generation,
where it turned out to behave as uncalibrated noise. Here the activations
*are* genuine in-progress generation states -- the model is actually
sampling, not being fed a scripted continuation -- so a probe/direction
trained on this data lives in the same distribution it will be applied to.

Sampling multiple rollouts per prompt also gives on-policy contrastive pairs
for the *same* question when a prompt's rollouts diverge (some correct, some
not) -- a matched design that should isolate a cleaner truthfulness
direction than pooling activations across unrelated questions, the same
logic as CAA's matched contrastive pairs (see steer.py's docstring).

Two granularities are recorded per rollout:
  - per-token rows (one row per generated token position) -- for probing
    "does a live gate work at token position X", same shape as
    extract_continuation_probe.py's output.
  - a mean-pooled-per-rollout vector (averaged over all generated token
    positions) -- matches the "Detecting Strategic Deception" paper's
    mean-pooled-generation-activation approach, and is what the
    rollout-level probe AUROC and diff-of-means directions below are
    computed from.
"""
import argparse
import gc
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dataset import build_distractor_dataset
from steer import SYSTEM_PROMPT, classify_answer
from extract_continuation_probe import get_target_layers, grouped_train_test_split


class Recorder:
    """Forward hook that records the residual stream at every *generated*
    token position. The first call to a decoder layer during model.generate()
    is the prefill pass (seq_len == prompt length); every call after that is
    exactly one new token under KV-cache decoding, so skipping call #1 and
    taking the last position of every later call gives exactly the generated
    tokens' activations, in order. Same hook pattern as steer.py's Steerer,
    but recording instead of intervening."""

    def __init__(self, model, layer):
        self.layer = layer
        self.buffer = []
        self.n_calls = 0
        self.handle = model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.n_calls += 1
        if self.n_calls > 1:
            self.buffer.append(output[:, -1, :].detach().float().cpu())
        return output

    def reset(self):
        self.buffer = []
        self.n_calls = 0

    def stacked(self):
        if not self.buffer:
            return np.zeros((0, 0))
        return torch.cat(self.buffer, dim=0).numpy()

    def remove(self):
        self.handle.remove()


def sample_rollout(model, tokenizer, question_text, temperature, max_new_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_prompts():
    """Plain + distractor prompts from the same country/capital set steer.py
    evaluates against, so this training data and that eval set are directly
    comparable."""
    examples = build_distractor_dataset()
    prompts = []
    for ex in examples:
        for kind, text in [("plain", ex["prompt_plain"]), ("distractor", ex["prompt_distractor"])]:
            prompts.append({
                "prompt_id": len(prompts), "kind": kind, "text": text,
                "capital": ex["capital"], "distractor_capital": ex["distractor_capital"],
            })
    return prompts


def compute_directions(mean_pooled, labels, prompt_ids):
    """Both a pooled diff-of-means (all label=1 vs all label=0 rollouts) and
    a paired diff-of-means restricted to prompts that yielded *both* a
    correct and an incorrect rollout -- the matched, on-policy contrastive
    pair design discussed before building this script."""
    pos, neg = mean_pooled[labels == 1], mean_pooled[labels == 0]
    pooled = pos.mean(axis=0) - neg.mean(axis=0)

    paired_diffs = []
    for pid in np.unique(prompt_ids):
        mask = prompt_ids == pid
        pos_p = mean_pooled[mask & (labels == 1)]
        neg_p = mean_pooled[mask & (labels == 0)]
        if len(pos_p) and len(neg_p):
            paired_diffs.append(pos_p.mean(axis=0) - neg_p.mean(axis=0))
    paired = np.mean(paired_diffs, axis=0) if paired_diffs else pooled
    return pooled, paired, len(paired_diffs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quant", choices=["none", "nf4", "prequantized"], default="none")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                         help="Layers to record. Default: auto-spaced (get_target_layers).")
    parser.add_argument("--samples_per_prompt", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--out", default="rollout_activations.npz")
    parser.add_argument("--directions_out", default="rollout_directions.npz")
    parser.add_argument("--results_csv", default=None)
    parser.add_argument("--rollouts_jsonl", default=None,
                         help="If set, dump every rollout's text + label here for inspection.")
    args = parser.parse_args()

    if args.quant == "none":
        load_kwargs = dict(device_map="auto", dtype=torch.bfloat16)
    elif args.quant == "nf4":
        from transformers import BitsAndBytesConfig
        load_kwargs = dict(
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
    else:
        load_kwargs = dict(device_map="auto")

    print(f"Loading {args.model} (quant={args.quant})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    n_layers = len(model.model.layers)
    target_layers = args.layers or get_target_layers(n_layers)
    print(f"Model has {n_layers} layers. Recording layers: {target_layers}")

    prompts = build_prompts()
    print(f"{len(prompts)} prompts x {args.samples_per_prompt} samples = "
          f"{len(prompts) * args.samples_per_prompt} rollouts to generate "
          f"(temperature={args.temperature}, max_new_tokens={args.max_new_tokens})")

    recorders = {l: Recorder(model, l) for l in target_layers}

    per_token_acts = {l: [] for l in target_layers}
    per_token_labels, per_token_prompt_ids, per_token_positions, per_token_kinds = [], [], [], []
    pooled_acts = {l: [] for l in target_layers}
    pooled_labels, pooled_prompt_ids, pooled_kinds = [], [], []
    rollout_records = []

    n_done = 0
    for p in prompts:
        for s in range(args.samples_per_prompt):
            for r in recorders.values():
                r.reset()
            text = sample_rollout(model, tokenizer, p["text"], args.temperature, args.max_new_tokens)
            correct, confused = classify_answer(text, p["capital"], p["distractor_capital"])
            label = 1 if correct else 0

            n_tok = recorders[target_layers[0]].stacked().shape[0]
            if n_tok == 0:
                continue  # degenerate empty generation; skip

            for l in target_layers:
                acts_l = recorders[l].stacked()
                per_token_acts[l].append(acts_l)
                pooled_acts[l].append(acts_l.mean(axis=0))

            per_token_labels.extend([label] * n_tok)
            per_token_prompt_ids.extend([p["prompt_id"]] * n_tok)
            per_token_positions.extend(range(n_tok))
            per_token_kinds.extend([p["kind"]] * n_tok)
            pooled_labels.append(label)
            pooled_prompt_ids.append(p["prompt_id"])
            pooled_kinds.append(p["kind"])

            rollout_records.append({
                "prompt_id": p["prompt_id"], "kind": p["kind"], "sample": s,
                "text": text, "label": label, "confused": confused,
            })
            n_done += 1
            if n_done % 50 == 0:
                print(f"  ...{n_done} rollouts done")

    for r in recorders.values():
        r.remove()

    per_token_labels = np.array(per_token_labels)
    per_token_prompt_ids = np.array(per_token_prompt_ids)
    per_token_positions = np.array(per_token_positions)
    pooled_labels = np.array(pooled_labels)
    pooled_prompt_ids = np.array(pooled_prompt_ids)

    print(f"\nCollected {len(pooled_labels)} rollouts, {len(per_token_labels)} total token rows.")
    print(f"Overall label balance: {pooled_labels.sum()} correct / {len(pooled_labels) - pooled_labels.sum()} incorrect")
    for kind in ["plain", "distractor"]:
        mask = np.array(pooled_kinds) == kind
        n = mask.sum()
        n_pos = pooled_labels[mask].sum()
        print(f"  {kind:10s}: {n_pos}/{n} correct")

    np.savez(
        args.out,
        labels=per_token_labels, prompt_ids=per_token_prompt_ids, positions=per_token_positions,
        pooled_labels=pooled_labels, pooled_prompt_ids=pooled_prompt_ids,
        **{f"layer_{l}": np.concatenate(per_token_acts[l], axis=0) for l in target_layers},
        **{f"pooled_layer_{l}": np.stack(pooled_acts[l], axis=0) for l in target_layers},
    )
    print(f"Saved activations to {args.out}")

    if args.rollouts_jsonl:
        with open(args.rollouts_jsonl, "w") as f:
            for rec in rollout_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved rollout transcripts to {args.rollouts_jsonl}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    train_mask, test_mask = grouped_train_test_split(pooled_prompt_ids)

    print("\n--- Per-layer rollout-level probe AUROC (mean-pooled, grouped held-out split) ---")
    results = {}
    directions_pooled, directions_paired = {}, {}
    for l in target_layers:
        X = np.stack(pooled_acts[l], axis=0)
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = pooled_labels[train_mask], pooled_labels[test_mask]

        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
            print(f"Layer {l:3d}: skipped (not enough label diversity in split)")
            continue

        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_train, y_train)
        auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        results[l] = auroc

        pooled_dir, paired_dir, n_pairs = compute_directions(X_train, y_train, pooled_prompt_ids[train_mask])
        directions_pooled[l] = pooled_dir
        directions_paired[l] = paired_dir

        print(f"Layer {l:3d}: AUROC = {auroc:.4f}  |pooled diff| = {np.linalg.norm(pooled_dir):.3f}  "
              f"|paired diff| = {np.linalg.norm(paired_dir):.3f}  (n_matched_prompts={n_pairs})")

    if directions_paired:
        np.savez(args.directions_out,
                  **{f"layer_{l}": directions_paired[l] for l in directions_paired},
                  **{f"pooled_layer_{l}": directions_pooled[l] for l in directions_pooled})
        print(f"Saved directions (paired + pooled) to {args.directions_out}")

    if args.results_csv and results:
        import csv
        write_header = not os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["model", "quant", "n_layers", "layer", "layer_pct", "auroc",
                            "samples_per_prompt", "temperature"])
            for l in results:
                w.writerow([args.model, args.quant, n_layers, l, round(100 * l / n_layers, 1),
                            results[l], args.samples_per_prompt, args.temperature])
        print(f"Appended results to {args.results_csv}")


if __name__ == "__main__":
    main()
