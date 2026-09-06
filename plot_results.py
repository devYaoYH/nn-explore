"""Generate comparison figures from results/probe_auroc.csv and
results/causal_patch.csv. Run after extract_and_probe.py and causal_patch.py
have been run (with --results_csv) for each model of interest.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def plot_rollout_probe_auroc():
    path = os.path.join(RESULTS_DIR, "rollout_probe.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in df.groupby("model"):
        g = g.sort_values("layer_pct")
        ax.plot(g["layer_pct"], g["auroc"], marker="o", label=SHORT_NAMES.get(model, model))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("Layer depth (% of network)")
    ax.set_ylabel("Rollout-level probe AUROC (mean-pooled, held out by question)")
    ax.set_title("On-policy free-generation rollouts: real signal, not a construction artifact")
    ax.set_ylim(0.4, 1.03)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "rollout_probe_auroc_by_layer.png")
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


def plot_position_breakdown():
    path = os.path.join(RESULTS_DIR, "position_breakdown.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in df.groupby("model"):
        g = g.sort_values("rel_pos")
        ax.plot(g["rel_pos"], g["auroc"], marker="o", label=model)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("Token position, relative to end of continuation (-1 = last token)")
    ax.set_ylabel("Train-set probe AUROC")
    ax.set_title("Forced-continuation: signal only appears at/after the answer token")
    ax.set_ylim(0.4, 1.03)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "position_breakdown.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def _plot_steering_sweep_file(csv_name, out_name, title):
    path = os.path.join(RESULTS_DIR, csv_name)
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, gated in zip(axes, [False, True]):
        g = df[df["gated"] == gated].sort_values("coeff")
        ax.plot(g["coeff"], g["plain_correct"] / g["plain_total"], marker="o", label="plain prompt accuracy")
        ax.plot(g["coeff"], g["distractor_correct"] / g["distractor_total"], marker="o", label="distractor prompt accuracy")
        ax.plot(g["coeff"], g["confused"] / g["distractor_total"], marker="o", label="confused (matched distractor)")
        ax.set_xlabel("Steering coefficient")
        ax.set_title("Gated" if gated else "Unconditional (every token)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Rate")
    fig.suptitle(title)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, out_name)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_steering_sweep():
    """Original attempt: direction trained on completed-statement /
    forced-continuation activations, applied live -- fails (train/deploy
    mismatch), see README section 3b."""
    _plot_steering_sweep_file(
        "steering_sweep.csv", "steering_sweep.png",
        "Live steering sweep: no coefficient improves distractor accuracy (Qwen2.5-1.5B, layer 21)",
    )


def plot_rollout_steering_sweep():
    """Fixed version: direction trained on genuine free-generation rollouts
    (collect_rollouts.py) -- works, see README section 4."""
    _plot_steering_sweep_file(
        "rollout_steering_sweep.csv", "rollout_steering_sweep.png",
        "On-policy rollout-trained direction: distractor accuracy 31/37->35/37, no fluency cost (layer 18)",
    )


def plot_cross_domain_generalization():
    path = os.path.join(RESULTS_DIR, "cross_domain_generalization.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df["condition"] = df["direction_domain"] + " dir\n-> " + df["eval_domain"] + " eval"
    conditions = df["condition"].unique()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(conditions))
    width = 0.35
    baseline = [df[(df["condition"] == c) & (df["coeff"] == 0.0)]["distractor_correct"].iloc[0]
                / df[(df["condition"] == c) & (df["coeff"] == 0.0)]["distractor_total"].iloc[0]
                for c in conditions]
    steered = [df[(df["condition"] == c) & (df["coeff"] != 0.0)]["distractor_correct"].iloc[0]
               / df[(df["condition"] == c) & (df["coeff"] != 0.0)]["distractor_total"].iloc[0]
               for c in conditions]
    ax.bar(x - width / 2, baseline, width, label="baseline (coeff=0)")
    ax.bar(x + width / 2, steered, width, label="steered (best coeff)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Distractor-prompt accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Steering only helps within its own training domain")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "cross_domain_generalization.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_direction_cosine_similarity():
    path = os.path.join(RESULTS_DIR, "direction_cosine_similarity.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (dir_a, dir_b), g in df.groupby(["dir_a", "dir_b"]):
        g = g.sort_values("layer_pct")
        ax.plot(g["layer_pct"], g["cosine"], marker="o", label=f"cos({dir_a}, {dir_b})")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="orthogonal")
    ax.set_xlabel("Layer depth (% of network)")
    ax.set_ylabel("Cosine similarity between domain directions")
    ax.set_title("Small shared component, growing with depth -- but far from aligned")
    ax.set_ylim(-0.1, 0.4)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "direction_cosine_similarity.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_probe_auroc()
    plot_causal_patch()
    plot_position_breakdown()
    plot_steering_sweep()
    plot_rollout_probe_auroc()
    plot_rollout_steering_sweep()
    plot_cross_domain_generalization()
    plot_direction_cosine_similarity()
