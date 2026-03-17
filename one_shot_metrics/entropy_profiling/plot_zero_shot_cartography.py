#!/usr/bin/env python3
"""Plot 4-panel Dataset Cartography comparison:
1. Training-dynamics cartography (confidence vs variability) for all 1209
2. Zero-shot Method 1: pass_rate (confidence) vs answer_entropy (variability)
3. Zero-shot Method 2: RSR (confidence) vs num_spikes (variability)
4. Training-dynamics cartography colored by MATH500 for Wang 14

All panels highlight Wang 14 examples with MATH500 scores.
The large run (659 examples) is plotted where available.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from scipy import stats

# --- Data sources ---
ACC_JSON = "one_shot_metrics/One-Shot-RLVR/data/acc_step_500.json"
ENTROPY_SUMMARY = "one_shot_metrics/entropy_profiling/results/entropy_summary.csv"
RSR_CSV = "one_shot_metrics/entropy_profiling/results/entropy_profiles/rsr_metrics.csv"
LARGE_DIVERSITY = "one_shot_metrics/entropy_profiling/results/large_run/entropy_profiles/diversity_metrics.csv"
LARGE_FEATURES = "one_shot_metrics/entropy_profiling/results/large_run/features_full.csv"

# Wang 14: pi_name -> (acc_step_500 index, math500_score)
WANG_MAP = {
    "pi_1": (124, 74.0), "pi_2": (267, 70.6), "pi_4": (931, 65.6),
    "pi_7": (568, 64.0), "pi_11": (875, 64.0), "pi_13": (682, 74.4),
    "pi_16": (50, 67.0), "pi_17": (870, 67.2), "pi_605": (60, 71.8),
    "pi_606": (906, 64.4), "pi_1201": (847, 71.4), "pi_1207": (174, 54.0),
    "pi_1208": (770, 45.0), "pi_1209": (789, 72.2),
}

# Unicode labels for the plot
WANG_LABELS = {
    "pi_1": "π₁", "pi_2": "π₂", "pi_4": "π₄", "pi_7": "π₇",
    "pi_11": "π₁₁", "pi_13": "π₁₃", "pi_16": "π₁₆", "pi_17": "π₁₇",
    "pi_605": "π₆₀₅", "pi_606": "π₆₀₆", "pi_1201": "π₁₂₀₁",
    "pi_1207": "π₁₂₀₇", "pi_1208": "π₁₂₀₈", "pi_1209": "π₁₂₀₉",
}

# --- Load data ---
with open(ACC_JSON) as f:
    acc_data = json.load(f)

df_summary = pd.read_csv(ENTROPY_SUMMARY)
try:
    df_rsr = pd.read_csv(RSR_CSV)
    has_rsr = True
except FileNotFoundError:
    df_rsr = None
    has_rsr = False

# Merge RSR into summary
if has_rsr:
    df_wang = df_summary.merge(df_rsr[["example", "rsr_mean", "rsr_correct_mean", "rsr_wrong_mean"]],
                               on="example", how="left")
else:
    df_wang = df_summary.copy()
    df_wang["rsr_mean"] = np.nan

# Large run data
try:
    df_large_div = pd.read_csv(LARGE_DIVERSITY)
except FileNotFoundError:
    df_large_div = pd.DataFrame()
try:
    df_large_feat = pd.read_csv(LARGE_FEATURES)
except FileNotFoundError:
    df_large_feat = None

# --- Compute training-dynamics cartography for all 1209 ---
all_indices = []
all_conf = []
all_var = []
for idx_str, accs in acc_data.items():
    all_indices.append(int(idx_str))
    all_conf.append(np.mean(accs))
    all_var.append(np.std(accs))
all_conf = np.array(all_conf)
all_var = np.array(all_var)

# Wang 14 training-dynamics values
wang_td_conf = []
wang_td_var = []
wang_scores = []
wang_names = []
for name, (idx, score) in WANG_MAP.items():
    curve = acc_data[str(idx)]
    wang_td_conf.append(np.mean(curve))
    wang_td_var.append(np.std(curve))
    wang_scores.append(score)
    wang_names.append(name)

# --- Large run: map ex_N to acc_step_500 index ---
large_conf = []
large_var = []
large_pass_rate = []
large_answer_entropy = []
large_names = []
if df_large_feat is not None:
    for _, row in df_large_div.iterrows():
        ex_name = row["example"]
        # Extract index from ex_N format
        if ex_name.startswith("ex_"):
            idx_str = ex_name.replace("ex_", "")
            if idx_str in acc_data:
                curve = acc_data[idx_str]
                large_conf.append(np.mean(curve))
                large_var.append(np.std(curve))
                large_pass_rate.append(row.get("pass_rate", np.nan))
                large_answer_entropy.append(row.get("answer_entropy", np.nan))
                large_names.append(ex_name)

large_conf = np.array(large_conf)
large_var = np.array(large_var)
large_pass_rate = np.array(large_pass_rate)
large_answer_entropy = np.array(large_answer_entropy)

# --- Zero-shot metrics for Wang 14 ---
wang_pass_rate = df_wang.set_index("example").loc[wang_names, "pass_rate"].values.astype(float)
wang_answer_entropy = df_wang.set_index("example").loc[wang_names, "answer_entropy"].values.astype(float)
wang_num_spikes = df_wang.set_index("example").loc[wang_names, "num_spikes_mean"].values.astype(float)
wang_rsr = df_wang.set_index("example").loc[wang_names, "rsr_mean"].values.astype(float)

# --- Colormap setup ---
vmin, vmax = 40, 80
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.RdYlGn


def annotate_wang(ax, x_vals, y_vals, scores, names, fontsize=6.5):
    """Annotate Wang 14 points with name and score."""
    for i in range(len(names)):
        label = WANG_LABELS.get(names[i], names[i])
        ax.annotate(f"{label}\n{scores[i]:.0f}",
                    (x_vals[i], y_vals[i]),
                    fontsize=fontsize, fontweight="bold", color="darkred",
                    xytext=(6, 5), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="red", alpha=0.85),
                    zorder=5)


def add_correlation_text(ax, x, y, label=""):
    """Add Spearman + Pearson correlation to axis."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return
    r_s, p_s = stats.spearmanr(x[mask], y[mask])
    r_p, p_p = stats.pearsonr(x[mask], y[mask])
    ax.text(0.03, 0.97,
            f"Wang 14 vs MATH500:\n"
            f"  x: ρ={stats.spearmanr(x[mask], np.array(wang_scores)[mask])[0]:.2f}\n"
            f"  y: ρ={stats.spearmanr(y[mask], np.array(wang_scores)[mask])[0]:.2f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(fc="lightyellow", ec="grey", alpha=0.8))


# --- PLOT ---
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# ============================================================
# Panel 1: Training-dynamics cartography (all 1209)
# ============================================================
ax = axes[0, 0]
ax.scatter(all_var, all_conf, s=12, c="lightgrey", edgecolors="grey",
           linewidth=0.2, alpha=0.5, zorder=1, label=f"All 1209 examples")

# Large run overlay in blue
if len(large_conf) > 0:
    ax.scatter(large_var, large_conf, s=18, c="cornflowerblue", edgecolors="navy",
               linewidth=0.3, alpha=0.4, zorder=2, label=f"Large run ({len(large_conf)} examples)")

# Wang 14 colored by MATH500
sc1 = ax.scatter(wang_td_var, wang_td_conf, s=120, c=wang_scores, cmap=cmap,
                 norm=norm, edgecolors="black", linewidth=1.2, zorder=4)
annotate_wang(ax, wang_td_var, wang_td_conf, wang_scores, wang_names)

ax.set_xlabel("Variability (std of accuracy curve)", fontsize=10)
ax.set_ylabel("Confidence (mean accuracy)", fontsize=10)
ax.set_title("Training-Dynamics Cartography\n(from acc_step_500.json, multi-example training)", fontsize=11)
ax.legend(loc="upper right", fontsize=8)

# Region labels
ax.text(0.02, 0.95, "EASY", transform=ax.transAxes, fontsize=9, color="green",
        alpha=0.6, va="top", style="italic")
ax.text(0.02, 0.08, "HARD", transform=ax.transAxes, fontsize=9, color="red",
        alpha=0.6, va="bottom", style="italic")
ax.text(0.7, 0.5, "AMBIGUOUS", transform=ax.transAxes, fontsize=9, color="blue",
        alpha=0.6, va="center", style="italic")

# ============================================================
# Panel 2: Zero-shot Method 1 — pass_rate vs answer_entropy
# ============================================================
ax = axes[0, 1]

# Large run background (these have pass_rate and answer_entropy)
if len(large_pass_rate) > 0:
    ax.scatter(large_answer_entropy, large_pass_rate, s=18, c="lightgrey",
               edgecolors="grey", linewidth=0.2, alpha=0.5, zorder=1,
               label=f"Large run ({len(large_pass_rate)} examples)")

# Wang 14
sc2 = ax.scatter(wang_answer_entropy, wang_pass_rate, s=120, c=wang_scores, cmap=cmap,
                 norm=norm, edgecolors="black", linewidth=1.2, zorder=4)
annotate_wang(ax, wang_answer_entropy, wang_pass_rate, wang_scores, wang_names)

ax.set_xlabel("Answer Entropy (Shannon entropy of distinct answers)", fontsize=10)
ax.set_ylabel("Pass Rate (fraction correct in 256 rollouts)", fontsize=10)
ax.set_title("Zero-Shot Cartography — Method 1\n(pass_rate vs answer_entropy, no training needed)", fontsize=11)
ax.legend(loc="upper right", fontsize=8)

# Region labels
ax.text(0.02, 0.95, "EASY\n(high pass rate,\nlow entropy)", transform=ax.transAxes,
        fontsize=8, color="green", alpha=0.6, va="top", style="italic")
ax.text(0.02, 0.08, "HARD\n(low pass rate,\nlow entropy)", transform=ax.transAxes,
        fontsize=8, color="red", alpha=0.6, va="bottom", style="italic")
ax.text(0.6, 0.5, "AMBIGUOUS\n(high entropy)", transform=ax.transAxes,
        fontsize=8, color="blue", alpha=0.6, va="center", style="italic")

# Correlation of each axis with MATH500 for Wang 14
w_scores = np.array(wang_scores)
r_pr, p_pr = stats.spearmanr(wang_pass_rate, w_scores)
r_ae, p_ae = stats.spearmanr(wang_answer_entropy, w_scores)
ax.text(0.03, 0.72,
        f"vs MATH500 (n=14):\n"
        f"  pass_rate ρ={r_pr:.2f} (p={p_pr:.2f})\n"
        f"  answer_ent ρ={r_ae:.2f} (p={p_ae:.2f})",
        transform=ax.transAxes, fontsize=7.5, va="top",
        bbox=dict(fc="lightyellow", ec="grey", alpha=0.85))

# ============================================================
# Panel 3: Zero-shot Method 2 — RSR vs num_spikes
# ============================================================
ax = axes[1, 0]

# Wang 14 only (RSR not computed for large run)
sc3 = ax.scatter(wang_num_spikes, wang_rsr, s=120, c=wang_scores, cmap=cmap,
                 norm=norm, edgecolors="black", linewidth=1.2, zorder=4)
annotate_wang(ax, wang_num_spikes, wang_rsr, wang_scores, wang_names)

ax.set_xlabel("Entropy Spike Count (mean per rollout)", fontsize=10)
ax.set_ylabel("RSR (rank-surprisal ratio, lower = more reachable)", fontsize=10)
ax.set_title("Zero-Shot Cartography — Method 2\n(RSR vs entropy spikes, from forward passes)", fontsize=11)

r_rsr, p_rsr = stats.spearmanr(wang_rsr, w_scores)
r_ns, p_ns = stats.spearmanr(wang_num_spikes, w_scores)
ax.text(0.03, 0.97,
        f"vs MATH500 (n=14):\n"
        f"  RSR ρ={r_rsr:.2f} (p={p_rsr:.2f})\n"
        f"  spikes ρ={r_ns:.2f} (p={p_ns:.2f})",
        transform=ax.transAxes, fontsize=7.5, va="top",
        bbox=dict(fc="lightyellow", ec="grey", alpha=0.85))

# Region labels
ax.text(0.02, 0.25, "EASY\n(low RSR,\nfew spikes)", transform=ax.transAxes,
        fontsize=8, color="green", alpha=0.6, va="top", style="italic")
ax.text(0.7, 0.97, "HARD\n(high RSR,\nmany spikes)", transform=ax.transAxes,
        fontsize=8, color="red", alpha=0.6, va="top", style="italic")

# ============================================================
# Panel 4: Can zero-shot predict training-dynamics position?
# ============================================================
ax = axes[1, 1]

# For the large run: scatter training variability vs zero-shot answer_entropy
if len(large_var) > 0 and len(large_answer_entropy) > 0:
    ax.scatter(large_answer_entropy, large_var, s=18, c="lightgrey",
               edgecolors="grey", linewidth=0.2, alpha=0.5, zorder=1,
               label=f"Large run (n={len(large_var)})")

# Wang 14
sc4 = ax.scatter(wang_answer_entropy,
                 [wang_td_var[i] for i in range(len(wang_names))],
                 s=120, c=wang_scores, cmap=cmap, norm=norm,
                 edgecolors="black", linewidth=1.2, zorder=4)
annotate_wang(ax, wang_answer_entropy,
              [wang_td_var[i] for i in range(len(wang_names))],
              wang_scores, wang_names)

# Correlation
if len(large_var) > 0:
    mask = np.isfinite(large_answer_entropy) & np.isfinite(large_var)
    r_all, p_all = stats.spearmanr(large_answer_entropy[mask], large_var[mask])
    ax.text(0.03, 0.97,
            f"Large run (n={mask.sum()}):\n"
            f"  answer_entropy vs training var\n"
            f"  ρ={r_all:.3f} (p={p_all:.4f})",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(fc="lightyellow", ec="grey", alpha=0.85))

ax.set_xlabel("Zero-Shot Answer Entropy", fontsize=10)
ax.set_ylabel("Training-Dynamics Variability (from acc_step_500)", fontsize=10)
ax.set_title("Zero-Shot → Training-Dynamics Transfer\n(can answer_entropy predict training variability?)", fontsize=11)
ax.legend(loc="lower right", fontsize=8)

# Add trendline for large run
if len(large_var) > 0:
    mask = np.isfinite(large_answer_entropy) & np.isfinite(large_var)
    z = np.polyfit(large_answer_entropy[mask], large_var[mask], 1)
    x_line = np.linspace(large_answer_entropy[mask].min(), large_answer_entropy[mask].max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.5, linewidth=1.5)

# --- Global colorbar ---
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("MATH500 Score (one-shot training)", fontsize=10)

fig.suptitle("Dataset Cartography: Training-Dynamics vs Zero-Shot Methods\n"
             "Wang 14 examples colored by actual one-shot MATH500 score",
             fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 0.93, 0.95])

out_dir = Path("one_shot_metrics/entropy_profiling/results/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "zero_shot_cartography.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {out_path}")

# --- Print summary ---
print(f"\n{'='*70}")
print("Correlation Summary: Each axis vs MATH500 (Wang 14)")
print(f"{'='*70}")
print(f"{'Axis':<30} {'Spearman ρ':>12} {'p-value':>10}")
print("-" * 55)
for label, vals in [
    ("TD confidence", np.array(wang_td_conf)),
    ("TD variability", np.array(wang_td_var)),
    ("ZS pass_rate", wang_pass_rate),
    ("ZS answer_entropy", wang_answer_entropy),
    ("ZS RSR", wang_rsr),
    ("ZS num_spikes", wang_num_spikes),
]:
    r, p = stats.spearmanr(vals, w_scores)
    print(f"{label:<30} {r:>12.3f} {p:>10.4f}")

# Panel 4 key stat
if len(large_var) > 0:
    mask = np.isfinite(large_answer_entropy) & np.isfinite(large_var)
    r, p = stats.spearmanr(large_answer_entropy[mask], large_var[mask])
    print(f"\n{'='*70}")
    print(f"Key question: does zero-shot answer_entropy predict training variability?")
    print(f"  Large run (n={mask.sum()}): ρ={r:.3f}, p={p:.4f}")
    r2, p2 = stats.spearmanr(large_pass_rate[mask], np.array(large_conf)[mask])
    print(f"  pass_rate vs training confidence: ρ={r2:.3f}, p={p2:.4f}")
