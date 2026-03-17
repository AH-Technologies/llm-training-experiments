#!/usr/bin/env python3
"""Plot RSR vs MATH score for Li 9 examples."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Load RSR metrics
rsr = pd.read_csv("results/li_run/entropy_profiles/rsr_metrics.csv")

# Load benchmark with math scores
bench = pd.read_csv("/cluster/projects/nn12068k/haaklau/llm-training-experiments/one_shot_metrics/li_examples_benchmark.csv")

# Merge on example name
df = rsr.merge(bench[["example", "math"]], on="example", how="left")
df = df.dropna(subset=["math"])

print(f"Li examples with MATH score: {len(df)}")
print(df[["example", "math", "avg_all", "rsr_mean", "mean_rank_mean", "mean_surprisal_mean"]].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (metric, label) in zip(axes, [
    ("rsr_mean", "Mean RSR"),
    ("mean_rank_mean", "Mean Token Rank"),
    ("mean_surprisal_mean", "Mean Surprisal"),
]):
    x = df[metric].values
    y = df["math"].values

    ax.scatter(x, y, s=80, c="tab:blue", edgecolors="black", linewidth=0.8, alpha=0.9)

    for j, name in enumerate(df["example"].values):
        ax.annotate(name, (x[j], y[j]), fontsize=7, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    if len(set(x)) > 1:
        r_s, p_s = stats.spearmanr(x, y)
        r_p, p_p = stats.pearsonr(x, y)
        ax.set_title(f"{label}\nSpearman \u03c1={r_s:.3f} (p={p_s:.3f}) | Pearson r={r_p:.3f}",
                     fontsize=10)
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.6)
    else:
        ax.set_title(label, fontsize=10)

    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("MATH Score", fontsize=9)

fig.suptitle(f"Rank-Surprisal Ratio vs MATH Score — Li Examples (n={len(df)})", fontsize=12)
fig.tight_layout()

out_dir = Path("results/li_run/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "rsr_vs_math_score.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved to {out_path}")
