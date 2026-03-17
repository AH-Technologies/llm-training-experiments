#!/usr/bin/env python3
"""Plot RSR vs MATH500 for Wang 14, with density of all 644 large-run examples."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Load Wang 14 RSR (with MATH500 scores)
wang = pd.read_csv("results/entropy_profiles/rsr_metrics.csv")

# Load large-run RSR (no MATH500 scores)
large = pd.read_csv("results/large_run/entropy_profiles/rsr_metrics.csv")

print(f"Wang 14 examples: {len(wang)}")
print(f"Large run examples: {len(large)}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (metric, label) in zip(axes, [
    ("rsr_mean", "Mean RSR"),
    ("mean_rank_mean", "Mean Token Rank"),
    ("mean_surprisal_mean", "Mean Surprisal"),
]):
    # --- Density of large-run examples (vertical distribution) ---
    large_vals = large[metric].dropna().values

    # KDE density on the x-axis
    if len(large_vals) > 5:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(large_vals)
        x_grid = np.linspace(large_vals.min(), large_vals.max(), 200)
        density = kde(x_grid)

        # Draw as filled region along the bottom of the plot
        # We'll use a secondary axis trick: shade a horizontal band
        # Use histogram instead for clarity
        ax.hist(large_vals, bins=30, orientation="vertical", alpha=0.15,
                color="gray", edgecolor="none", density=False, zorder=1,
                label=f"Large run (n={len(large_vals)})")

        # Also show rug plot
        y_rug = np.full_like(large_vals, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 40)

    # --- Wang 14 scatter with MATH500 ---
    x = wang[metric].values
    y = wang["math500_score"].values

    ax.scatter(x, y, s=100, c="tab:blue", edgecolors="black", linewidth=1,
               alpha=0.95, zorder=5, label="Wang 14 (with MATH500)")

    for j, name in enumerate(wang["example"].values):
        ax.annotate(name, (x[j], y[j]), fontsize=6.5, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points", zorder=6)

    # Correlation for Wang 14
    if len(set(x)) > 1:
        r_s, p_s = stats.spearmanr(x, y)
        r_p, p_p = stats.pearsonr(x, y)
        ax.set_title(f"{label}\nSpearman \u03c1={r_s:.3f} (p={p_s:.3f}) | Pearson r={r_p:.3f}",
                     fontsize=10)
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.6, zorder=4)
    else:
        ax.set_title(label, fontsize=10)

    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("MATH500 Score", fontsize=9)
    ax.legend(fontsize=7, loc="best")

fig.suptitle(f"RSR vs MATH500 — Wang 14 with Large Run Density (n={len(large)})", fontsize=12)
fig.tight_layout()

out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "rsr_vs_math500_with_density.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved to {out_path}")

# Also make a better version with marginal density on the x-axis
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 8),
                            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05})

for col_idx, (metric, label) in enumerate([
    ("rsr_mean", "Mean RSR"),
    ("mean_rank_mean", "Mean Token Rank"),
    ("mean_surprisal_mean", "Mean Surprisal"),
]):
    ax_hist = axes2[0, col_idx]
    ax_main = axes2[1, col_idx]

    large_vals = large[metric].dropna().values
    wang_vals = wang[metric].values
    wang_scores = wang["math500_score"].values

    # Top: histogram of large-run values
    ax_hist.hist(large_vals, bins=30, alpha=0.4, color="gray", edgecolor="none",
                 density=True, label=f"All {len(large_vals)} examples")
    # Mark Wang 14 positions with rug
    for v in wang_vals:
        ax_hist.axvline(v, color="tab:blue", alpha=0.5, linewidth=1)
    ax_hist.set_ylabel("Density", fontsize=8)
    ax_hist.set_title(f"{label}", fontsize=10)
    ax_hist.tick_params(labelbottom=False)
    ax_hist.legend(fontsize=7)

    # Bottom: scatter of Wang 14
    ax_main.scatter(wang_vals, wang_scores, s=100, c="tab:blue", edgecolors="black",
                    linewidth=1, alpha=0.95, zorder=5)

    for j, name in enumerate(wang["example"].values):
        ax_main.annotate(name, (wang_vals[j], wang_scores[j]), fontsize=6.5,
                         ha="left", va="bottom", xytext=(4, 4),
                         textcoords="offset points", zorder=6)

    if len(set(wang_vals)) > 1:
        r_s, p_s = stats.spearmanr(wang_vals, wang_scores)
        r_p, p_p = stats.pearsonr(wang_vals, wang_scores)
        ax_main.set_title(f"Spearman \u03c1={r_s:.3f} (p={p_s:.3f})", fontsize=9)
        z = np.polyfit(wang_vals, wang_scores, 1)
        x_line = np.linspace(wang_vals.min(), wang_vals.max(), 50)
        ax_main.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.6, zorder=4)

    ax_main.set_xlabel(label, fontsize=9)
    ax_main.set_ylabel("MATH500 Score", fontsize=9)

    # Align x limits
    all_vals = np.concatenate([large_vals, wang_vals])
    margin = (all_vals.max() - all_vals.min()) * 0.05
    xlim = (all_vals.min() - margin, all_vals.max() + margin)
    ax_hist.set_xlim(xlim)
    ax_main.set_xlim(xlim)

fig2.suptitle(f"RSR vs MATH500 — Wang 14 Scatter + Marginal Density of {len(large)} Examples",
              fontsize=12)
fig2.tight_layout()
out_path2 = out_dir / "rsr_vs_math500_marginal_density.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved to {out_path2}")
