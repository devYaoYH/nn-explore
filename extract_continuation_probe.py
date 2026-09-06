"""Per-token probing on forced (teacher-forced) continuations.

Unlike extract_and_probe.py (one activation snapshot per whole statement, at
the last token), this captures an activation at *every token position of the
continuation* -- i.e. it approximates what you'd see walking token-by-token
through a real generation, without the labeling ambiguity of free generation
(the continuation is forced, so every position's label is exact by
construction).

Also computes the difference-of-means direction per layer, which is what the
steering experiment (steer.py) uses to actually push the residual stream --
see README.md for why this is preferred over the raw logistic-regression
weight vector for steering.

Train/test is split by *example* (question), not by individual token row --
splitting by row would leak the same statement's tokens across train and
test and make the probe look trivially better than it is.
"""
import argparse
import csv
import gc
import os

import numpy as np
import torch
from nnsight import LanguageModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dataset import build_steering_dataset


def get_target_layers(n_layers, n_probes=7):
    step = max(1, n_layers // (n_probes + 1))
    return list(range(step, n_layers, step))[:n_probes]


def tokenize_forced(tokenizer, prompt, continuation):
    """Concatenate token ids directly (not strings) so the prompt/continuation
    boundary can't be shifted by a BPE merge across the join point."""
    ids_prompt = tokenizer(prompt)["input_ids"]
    ids_cont = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    return ids_prompt, ids_cont


def extract_continuation_activations(model, examples, target_layers):
    """Returns, per layer: activations [n_token_rows, hidden], labels
    [n_token_rows], example_id [n_token_rows] (for grouped train/test split),
    and position-within-continuation [n_token_rows] (0-indexed)."""
    tok = model.tokenizer
    acts = {l: [] for l in target_layers}
    labels, example_ids, positions = [], [], []

    ex_id = 0
    for ex in examples:
        for label, continuation in [(1, ex["true_continuation"]), (0, ex["false_continuation"])]:
            ids_prompt, ids_cont = tokenize_forced(tok, ex["prompt"], continuation)
            input_ids = torch.tensor([ids_prompt + ids_cont])
            cont_start = len(ids_prompt)
            n_cont = len(ids_cont)

            with torch.no_grad(), model.trace(input_ids):
                for l in target_layers:
                    # transformers 5.x: decoder layer .output is the
                    # hidden_states tensor directly, [batch, seq, hidden].
                    proxy = model.model.layers[l].output[:, cont_start:, :].save()
                    acts[l].append(proxy)

            labels.extend([label] * n_cont)
            example_ids.extend([ex_id] * n_cont)
            positions.extend(range(n_cont))
        ex_id += 1

    out = {}
    for l in target_layers:
        out[l] = torch.cat([a.squeeze(0) for a in acts[l]], dim=0).float().cpu().numpy()
    return out, np.array(labels), np.array(example_ids), np.array(positions)


def grouped_train_test_split(example_ids, test_frac=0.3, seed=0):
    rng = np.random.RandomState(seed)
    unique_ex = np.unique(example_ids)
    rng.shuffle(unique_ex)
    n_test = max(1, int(len(unique_ex) * test_frac))
    test_ex = set(unique_ex[:n_test])
    is_test = np.array([e in test_ex for e in example_ids])
    return ~is_test, is_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quant", choices=["none", "nf4", "prequantized"], default="none")
    parser.add_argument("--out", default="continuation_activations.npz")
    parser.add_argument("--results_csv", default=None)
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
    model = LanguageModel(args.model, **load_kwargs)
    n_layers = len(model.model.layers)
    target_layers = get_target_layers(n_layers)
    print(f"Model has {n_layers} layers. Probing layers: {target_layers}")

    examples = build_steering_dataset()
    print(f"Extracting per-token continuation activations for {len(examples)} question pairs "
          f"({2 * len(examples)} forced continuations)...")
    acts, labels, example_ids, positions = extract_continuation_activations(model, examples, target_layers)
    print(f"Total token rows: {len(labels)}")

    np.savez(args.out, labels=labels, example_ids=example_ids, positions=positions,
              **{f"layer_{l}": acts[l] for l in target_layers})
    print(f"Saved activations to {args.out}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    train_mask, test_mask = grouped_train_test_split(example_ids)

    print("\n--- Per-layer per-token probe AUROC (grouped held-out split) ---")
    results = {}
    directions = {}
    for l in target_layers:
        X = acts[l]
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = labels[train_mask], labels[test_mask]

        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, preds)
        results[l] = auroc

        # Difference-of-means direction (train split only) -- this is what
        # steer.py uses to actually push the residual stream.
        diff = X_train[y_train == 1].mean(axis=0) - X_train[y_train == 0].mean(axis=0)
        directions[l] = diff

        print(f"Layer {l:3d}: AUROC = {auroc:.4f}  |diff-of-means| = {np.linalg.norm(diff):.3f}")

    best_layer = max(results, key=results.get)
    print(f"\nBest layer: {best_layer} (AUROC={results[best_layer]:.4f})")

    np.savez("steering_directions.npz" if args.out == "continuation_activations.npz"
              else args.out.replace(".npz", "_directions.npz"),
              **{f"layer_{l}": directions[l] for l in target_layers})

    if args.results_csv:
        write_header = not os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["model", "quant", "n_layers", "layer", "layer_pct", "auroc", "diff_of_means_norm"])
            for l in target_layers:
                w.writerow([args.model, args.quant, n_layers, l, round(100 * l / n_layers, 1),
                            results[l], np.linalg.norm(directions[l])])
        print(f"Appended results to {args.results_csv}")

    return best_layer, results, directions


if __name__ == "__main__":
    main()
