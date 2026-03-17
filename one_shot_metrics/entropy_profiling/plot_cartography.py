#!/usr/bin/env python3
"""Plot Dataset Cartography (confidence vs variability) for all 1209 RLVR examples.
Highlights Wang 14 examples with MATH500 scores in red."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ACC_JSON = "one_shot_metrics/One-Shot-RLVR/data/acc_step_500.json"

# Wang 14 examples: index -> (name, math500_score)
WANG_EXAMPLES = {
    124: ("π₁", 74.0),
    267: ("π₂", 70.6),
    931: ("π₄", 65.6),
    568: ("π₇", 64.0),
    875: ("π₁₁", 64.0),
    682: ("π₁₃", 74.4),
    50:  ("π₁₆", 67.0),
    870: ("π₁₇", 67.2),
    60:  ("π₆₀₅", 71.8),
    906: ("π₆₀₆", 64.4),
    847: ("π₁₂₀₁", 71.4),
    174: ("π₁₂₀₇", 54.0),
    770: ("π₁₂₀₈", 45.0),
    789: ("π₁₂₀₉", 72.2),
}

with open(ACC_JSON) as f:
    acc_data = json.load(f)

# Compute confidence and variability for all 1209 examples
all_conf, all_var, all_idx = [], [], []
wang_conf, wang_var, wang_scores, wang_names = [], [], [], []

for idx_str, accs in acc_data.items():
    idx = int(idx_str)
    conf = np.mean(accs)
    var = np.std(accs)
    all_conf.append(conf)
    all_var.append(var)
    all_idx.append(idx)

    if idx in WANG_EXAMPLES:
        wang_conf.append(conf)
        wang_var.append(var)
        wang_scores.append(WANG_EXAMPLES[idx][1])
        wang_names.append(WANG_EXAMPLES[idx][0])

all_conf = np.array(all_conf)
all_var = np.array(all_var)

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 8))

# All 1209 examples in grey
ax.scatter(all_var, all_conf, s=15, c="lightgrey", edgecolors="grey",
           linewidth=0.3, alpha=0.6, label=f"All examples (n={len(all_conf)})", zorder=1)

# Wang 14 in red, sized by MATH500 score
sizes = [(s - 40) * 3 + 40 for s in wang_scores]  # scale for visibility
sc = ax.scatter(wang_var, wang_conf, s=sizes, c=wang_scores, cmap="RdYlGn",
                edgecolors="black", linewidth=1.2, zorder=3, vmin=40, vmax=80)

# Annotate Wang examples
for i in range(len(wang_names)):
    ax.annotate(f"{wang_names[i]}\n{wang_scores[i]:.0f}",
                (wang_var[i], wang_conf[i]),
                fontsize=7, fontweight="bold", color="darkred",
                xytext=(8, 6), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8),
                zorder=4)

# Colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("MATH500 Score (one-shot training)", fontsize=10)

# Region labels
ax.text(0.02, 0.95, "EASY\n(high confidence,\nlow variability)",
        transform=ax.transAxes, fontsize=9, color="green", alpha=0.7,
        va="top", style="italic")
ax.text(0.02, 0.08, "HARD\n(low confidence,\nlow variability)",
        transform=ax.transAxes, fontsize=9, color="red", alpha=0.7,
        va="bottom", style="italic")
ax.text(0.65, 0.5, "AMBIGUOUS\n(high variability)",
        transform=ax.transAxes, fontsize=9, color="blue", alpha=0.7,
        va="center", style="italic")

ax.set_xlabel("Variability (std of accuracy across 500 training steps)", fontsize=11)
ax.set_ylabel("Confidence (mean accuracy across 500 training steps)", fontsize=11)
ax.set_title("Dataset Cartography for 1,209 RLVR Examples\n"
             "Wang 14 one-shot examples highlighted with MATH500 scores",
             fontsize=13)
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(-0.02, max(all_var) * 1.1)
ax.set_ylim(-0.05, 1.05)

fig.tight_layout()
out_dir = Path("one_shot_metrics/entropy_profiling/results/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "dataset_cartography.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {out_path}")

# Print distribution stats
print(f"\nAll 1209 examples:")
print(f"  Confidence: mean={all_conf.mean():.3f}, std={all_conf.std():.3f}")
print(f"  Variability: mean={all_var.mean():.3f}, std={all_var.std():.3f}")
print(f"  Zero-variability (never solved): {(all_var == 0).sum()}")
print(f"  High-confidence (>0.8): {(all_conf > 0.8).sum()}")
print(f"  Ambiguous (var > top 33%): threshold={np.percentile(all_var, 67):.4f}")

# Categorize
var_thresh = np.percentile(all_var, 67)
easy = ((all_conf > 0.7) & (all_var < var_thresh)).sum()
hard = ((all_conf < 0.3) & (all_var < var_thresh)).sum()
ambig = (all_var >= var_thresh).sum()
print(f"\n  Easy (conf>0.7, low var): {easy}")
print(f"  Hard (conf<0.3, low var): {hard}")
print(f"  Ambiguous (top 33% var): {ambig}")
