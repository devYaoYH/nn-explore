"""Causal activation patching: does the residual stream at the *subject
token* position causally determine the factual completion, or is the
correlation from the linear probes merely epiphenomenal?

Standard causal-tracing setup (as in ROME / Meng et al.): patch the subject
token's residual stream (not the last token -- downstream attention can still
see the unperturbed subject token elsewhere in the sequence, which masks the
effect) at a single layer, sweep across layers, and watch the target logits
cross over.
"""
import argparse
import csv
import os

import torch
from nnsight import LanguageModel


def find_subject_index(tokenizer, prompt, subject_word):
    ids = tokenizer(prompt)["input_ids"]
    tokens = [tokenizer.decode([t]).strip() for t in ids]
    for i, t in enumerate(tokens):
        if t == subject_word:
            return i
    raise ValueError(f"Could not find {subject_word!r} in {tokens}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quant", choices=["none", "nf4", "prequantized"], default="none")
    parser.add_argument("--results_csv", default=None,
                         help="If set, append per-layer patching rows to this CSV.")
    args = parser.parse_args()

    if args.quant == "none":
        load_kwargs = dict(device_map="auto", dtype=torch.bfloat16)
    elif args.quant == "nf4":
        from transformers import BitsAndBytesConfig
        load_kwargs = dict(
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
    else:  # prequantized
        load_kwargs = dict(device_map="auto")

    print(f"Loading {args.model} (quant={args.quant})...")
    model = LanguageModel(args.model, **load_kwargs)
    tok = model.tokenizer
    n_layers = len(model.model.layers)
    target_layers = list(range(1, n_layers, max(1, n_layers // 12)))

    clean_prompt = "The capital of France is"
    corrupt_prompt = "The capital of Germany is"
    clean_subj_idx = find_subject_index(tok, clean_prompt, "France")
    corrupt_subj_idx = find_subject_index(tok, corrupt_prompt, "Germany")
    assert clean_subj_idx == corrupt_subj_idx, "prompts must align token-for-token"
    subj_idx = clean_subj_idx

    paris_id = tok(" Paris")["input_ids"][0]
    berlin_id = tok(" Berlin")["input_ids"][0]

    with torch.no_grad(), model.trace(clean_prompt):
        clean_logits = model.lm_head.output[:, -1, :].save()
    with torch.no_grad(), model.trace(corrupt_prompt):
        corrupt_logits = model.lm_head.output[:, -1, :].save()

    print(f"\nClean prompt:   {clean_prompt!r} -> top token: {tok.decode([torch.argmax(clean_logits[0]).item()])!r}")
    print(f"Corrupt prompt: {corrupt_prompt!r} -> top token: {tok.decode([torch.argmax(corrupt_logits[0]).item()])!r}")
    print(f"Baseline logit(Paris)={clean_logits[0, paris_id]:.2f}  logit(Berlin)={clean_logits[0, berlin_id]:.2f}\n")

    print(f"Patching subject token (index {subj_idx}, {tok.decode([tok(clean_prompt)['input_ids'][subj_idx]])!r}) "
          f"from corrupt run into clean run, layer by layer:\n")
    print(f"{'layer':>5} | {'top token':<12} | {'logit(Paris)':>12} | {'logit(Berlin)':>13}")
    print("-" * 52)

    rows = []
    for l in target_layers:
        with torch.no_grad():
            with model.trace(corrupt_prompt):
                corrupt_rep = model.model.layers[l].output[:, subj_idx:subj_idx + 1, :].save()
            with model.trace(clean_prompt):
                model.model.layers[l].output[:, subj_idx:subj_idx + 1, :] = corrupt_rep
                patched_logits = model.lm_head.output[:, -1, :].save()

        top_id = torch.argmax(patched_logits[0]).item()
        p_paris = patched_logits[0, paris_id].item()
        p_berlin = patched_logits[0, berlin_id].item()
        print(f"{l:5d} | {tok.decode([top_id])!r:<12} | {p_paris:12.2f} | {p_berlin:13.2f}")
        rows.append((l, round(100 * l / n_layers, 1), tok.decode([top_id]), p_paris, p_berlin))

    if args.results_csv:
        write_header = not os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["model", "quant", "n_layers", "layer", "layer_pct",
                            "top_token", "logit_paris", "logit_berlin"])
            for l, pct, top_token, p_paris, p_berlin in rows:
                w.writerow([args.model, args.quant, n_layers, l, pct, top_token, p_paris, p_berlin])
        print(f"Appended results to {args.results_csv}")


if __name__ == "__main__":
    main()
