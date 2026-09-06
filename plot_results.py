"""Generate comparison figures from results/probe_auroc.csv and
results/causal_patch.csv. Run after extract_and_probe.py and causal_patch.py
have been run (with --results_csv) for each model of interest.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SHORT_NAMES = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B (bf16)",
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": "Qwen2.5-32B (NF4)",
}


def plot_probe_auroc():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "probe_auroc.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in df.groupby("model"):
        g = g.sort_values("layer_pct")
        ax.plot(g["layer_pct"], g["auroc"], marker="o", label=SHORT_NAMES.get(model, model))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("Layer depth (% of network)")
    ax.set_ylabel("Held-out probe AUROC")
    ax.set_title("Linear probe: true vs. false factual statements")
    ax.set_ylim(0.35, 1.03)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "probe_auroc_by_layer.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_causal_patch():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "causal_patch.csv"))
    models = df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        g = df[df["model"] == model].sort_values("layer_pct")
        ax.plot(g["layer_pct"], g["logit_paris"], marker="o", label="logit(' Paris')  [clean answer]")
        ax.plot(g["layer_pct"], g["logit_berlin"], marker="o", label="logit(' Berlin') [corrupt answer]")
        ax.set_xlabel("Layer depth (% of network)")
        ax.set_ylabel("Logit")
        ax.set_title(SHORT_NAMES.get(model, model))
        ax.legend(fontsize=8)
    fig.suptitle("Causal patch of subject-token residual stream (clean ← corrupt)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "causal_patch_by_layer.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_probe_auroc()
    plot_causal_patch()
