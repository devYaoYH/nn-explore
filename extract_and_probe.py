"""Extract last-token residual-stream activations across layers and train
per-layer logistic-regression probes to distinguish true vs. false factual
statements.
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
from sklearn.model_selection import train_test_split

from dataset import build_dataset


def get_target_layers(n_layers, n_probes=7):
    step = max(1, n_layers // (n_probes + 1))
    return list(range(step, n_layers, step))[:n_probes]


def extract_activations(model, texts, target_layers):
    """Returns {layer: np.ndarray [n_examples, hidden]} for last-token residual
    stream after each target decoder layer. One example at a time (batch=1) so
    this same code works unmodified once we swap in the 4-bit 32B model."""
    acts = {l: [] for l in target_layers}
    for text in texts:
        with torch.no_grad(), model.trace(text):
            for l in target_layers:
                # transformers 5.x: decoder layer .output is the hidden_states
                # tensor directly, shape [batch, seq, hidden] -- no [0] index.
                proxy = model.model.layers[l].output[:, -1, :].save()
                acts[l].append(proxy)
    return {l: torch.cat(acts[l], dim=0).float().cpu().numpy()
            for l in target_layers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quant", choices=["none", "nf4", "prequantized"], default="none")
    parser.add_argument("--out", default="activations.npz")
    parser.add_argument("--results_csv", default=None,
                         help="If set, append per-layer AUROC rows to this CSV.")
    parser.add_argument("--directions_out", default=None,
                         help="If set, save a {layer: diff-of-means vector} .npz here -- "
                              "the generic truthfulness direction used for steering.")
    args = parser.parse_args()

    if args.quant == "none":
        load_kwargs = dict(device_map="auto", dtype=torch.bfloat16)
    elif args.quant == "nf4":
        # Quantize on load from full-precision weights (downloads the full
        # bf16 checkpoint first, then quantizes -- use for models with no
        # pre-quantized checkpoint available).
        from transformers import BitsAndBytesConfig
        load_kwargs = dict(
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
    else:  # prequantized: checkpoint already stores NF4 weights + quantization_config
        load_kwargs = dict(device_map="auto")

    print(f"Loading {args.model} (quant={args.quant})...")
    model = LanguageModel(args.model, **load_kwargs)
    n_layers = len(model.model.layers)
    target_layers = get_target_layers(n_layers)
    print(f"Model has {n_layers} layers. Probing layers: {target_layers}")

    data = build_dataset()
    texts = [e["text"] for e in data]
    labels = np.array([e["label"] for e in data])

    print(f"Extracting activations for {len(texts)} examples...")
    acts = extract_activations(model, texts, target_layers)

    np.savez(args.out, labels=labels, **{f"layer_{l}": acts[l] for l in target_layers})
    print(f"Saved activations to {args.out}")

    # Free GPU memory before probing (probing itself is CPU/sklearn only).
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- Per-layer probe AUROC (held-out test split) ---")
    results = {}
    directions = {}
    for l in target_layers:
        X = acts[l]
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.3, random_state=0, stratify=labels
        )
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, preds)
        results[l] = auroc

        # Generic "truthfulness" direction for steering: difference of class
        # means (train split only), NOT the LR weight vector -- see README
        # for why diff-of-means is preferred for steering vs. classification.
        directions[l] = X_train[y_train == 1].mean(axis=0) - X_train[y_train == 0].mean(axis=0)

        print(f"Layer {l:3d}: AUROC = {auroc:.4f}  |diff-of-means| = {np.linalg.norm(directions[l]):.3f}")

    best_layer = max(results, key=results.get)
    print(f"\nBest layer: {best_layer} (AUROC={results[best_layer]:.4f})")

    if args.directions_out:
        np.savez(args.directions_out, **{f"layer_{l}": directions[l] for l in target_layers})
        print(f"Saved steering directions to {args.directions_out}")

    if args.results_csv:
        write_header = not os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["model", "quant", "n_layers", "layer", "layer_pct", "auroc"])
            for l in target_layers:
                w.writerow([args.model, args.quant, n_layers, l, round(100 * l / n_layers, 1), results[l]])
        print(f"Appended results to {args.results_csv}")

    return best_layer, results


if __name__ == "__main__":
    main()
