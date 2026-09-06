"""Per-token-position AUROC breakdown for the forced-continuation dataset.

Confirms (and visualizes) why extract_continuation_probe.py's pooled AUROC is
much lower than the static-statement probe: positions before the answer word
are byte-identical between the true/false continuation of the same question,
so a probe can only find signal at/after the point where they diverge. This
is a construction artifact of forced-continuation with matched prefixes, not
evidence about what the model "knows" before generating -- see README.md.

Run extract_continuation_probe.py first to produce the input .npz.
"""
import argparse
import csv
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True,
                         help="continuation_activations.npz from extract_continuation_probe.py")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--results_csv", default=None)
    parser.add_argument("--model_label", default="")
    args = parser.parse_args()

    d = np.load(args.activations)
    labels, example_ids, positions = d["labels"], d["example_ids"], d["positions"]
    X = d[f"layer_{args.layer}"]

    # Continuation length varies by example; align by distance-from-end
    # (negative = distance from the final token) since every continuation
    # ends "... is <answer>." regardless of length.
    lens = {int(e): int((example_ids == e).sum() // 2) for e in np.unique(example_ids)}
    rel_pos = np.array([positions[i] - lens[int(example_ids[i])] for i in range(len(positions))])

    rows = []
    print(f"{'rel_pos':>8} | {'n':>5} | {'train-set AUROC':>16}")
    print("-" * 36)
    for rp in sorted(set(rel_pos.tolist())):
        mask = rel_pos == rp
        if mask.sum() < 20 or len(set(labels[mask].tolist())) < 2:
            continue
        Xr, yr = X[mask], labels[mask]
        clf = LogisticRegression(max_iter=2000).fit(Xr, yr)
        auroc = roc_auc_score(yr, clf.predict_proba(Xr)[:, 1])
        rows.append((rp, int(mask.sum()), auroc))
        print(f"{rp:8d} | {mask.sum():5d} | {auroc:16.3f}")

    if args.results_csv:
        write_header = not os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["model", "layer", "rel_pos", "n", "auroc"])
            for rp, n, auroc in rows:
                w.writerow([args.model_label, args.layer, rp, n, auroc])
        print(f"Appended to {args.results_csv}")


if __name__ == "__main__":
    main()
