#!/usr/bin/env python3
"""Zero-shot cartography: mean_surprisal vs entropy_var_correct.

Plots all Wang 14 examples + pi13_repeat, colored by MATH500 score.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


def compute_entropy_var_correct(data):
    correct = [r for r in data["rollouts"] if r.get("is_correct", False)]
    if len(correct) < 2:
        return float("nan")
    ce = [np.mean(r["entropy_array"]) for r in correct]
    return np.var(ce)


def load_rsr(csv_path, name):
    import pandas as pd
    df = pd.read_csv(csv_path)
    row = df[df["example"] == name]
    if len(row) == 0:
        return {}
    return dict(row.iloc[0])


results_dir = Path("results/entropy_profiles")
pkl_files = sorted(results_dir.glob("entropy_*.pkl"))

# Also include pi13_repeat
repeat_pkl = Path("results/pi13_repeat/entropy_profiles/entropy_pi_13_repeat.pkl")

names = []
surprisals = []
evcs = []
scores = []
pass_rates = []

for pkl_path in list(pkl_files) + ([repeat_pkl] if repeat_pkl.exists() else []):
    name = pkl_path.stem.replace("entropy_", "")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    evc = compute_entropy_var_correct(data)
    example = data["example"]
    score = example.get("math500_score")
    pr = sum(r.get("is_correct", False) for r in data["rollouts"]) / len(data["rollouts"])

    # Get surprisal from RSR CSV
    if name == "pi_13_repeat":
        rsr = load_rsr("results/pi13_repeat/entropy_profiles/rsr_metrics.csv", name)
    else:
        rsr = load_rsr("results/entropy_profiles/rsr_metrics.csv", name)

    surp = rsr.get("mean_surprisal_mean")
    if surp is None:
        continue

    names.append(name)
    surprisals.append(surp)
    evcs.append(evc)
    scores.append(score if score else 74.4 if name == "pi_13_repeat" else None)
    pass_rates.append(pr)

# Separate into: has score (Wang 14 + repeat) vs no score
has_score = [(n, s, e, sc, pr) for n, s, e, sc, pr in zip(names, surprisals, evcs, scores, pass_rates) if sc is not None and not np.isnan(e)]
no_evc = [(n, s, e, sc, pr) for n, s, e, sc, pr in zip(names, surprisals, evcs, scores, pass_rates) if sc is not None and np.isnan(e)]

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Plot examples with both metrics
if has_score:
    ns, ss, es, scs, prs = zip(*has_score)
    ss = np.array(ss)
    es = np.array(es)
    scs = np.array(scs)

    scatter = ax.scatter(ss, es, c=scs, cmap="RdYlGn", s=120, edgecolors="black",
                         linewidth=1, zorder=5, vmin=45, vmax=75)
    cbar = plt.colorbar(scatter, ax=ax, label="MATH500 Score")

    for i, n in enumerate(ns):
        label = n
        if n == "pi_13_repeat":
            label = "pi_13_repeat"
        ax.annotate(label, (ss[i], es[i]), fontsize=7, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points", zorder=6)

    # Correlation
    r_s, p_s = stats.spearmanr(scs, es)
    r_s2, p_s2 = stats.spearmanr(scs, ss)
    ax.set_title(
        f"Zero-Shot Cartography: Surprisal vs Correct-Rollout Entropy Variance\n"
        f"MATH500 vs surprisal: Spearman={r_s2:.3f} (p={p_s2:.3f})  |  "
        f"MATH500 vs evc: Spearman={r_s:.3f} (p={p_s:.3f})",
        fontsize=10,
    )

# Plot examples with pass_rate=0 (no correct rollouts, so no EVC) at bottom
if no_evc:
    ns2, ss2, _, scs2, _ = zip(*no_evc)
    ss2 = np.array(ss2)
    scs2 = np.array(scs2)
    # Place at y=0 with different marker
    ax.scatter(ss2, np.zeros(len(ss2)), c=scs2, cmap="RdYlGn", s=120,
               edgecolors="black", linewidth=1, zorder=5, marker="v",
               vmin=45, vmax=75)
    for i, n in enumerate(ns2):
        ax.annotate(n + " (no correct)", (ss2[i], 0), fontsize=6, ha="left",
                    va="top", xytext=(5, -8), textcoords="offset points",
                    zorder=6, color="gray")

ax.set_xlabel("Mean Surprisal (lower = model finds reasoning more natural)", fontsize=10)
ax.set_ylabel("Entropy Variance of Correct Rollouts (higher = more diverse paths)", fontsize=10)

# Quadrant labels
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xmid = np.median(ss)
ymid = np.nanmedian(es)

ax.axvline(xmid, color="gray", linestyle="--", alpha=0.3)
ax.axhline(ymid, color="gray", linestyle="--", alpha=0.3)

# Label quadrants
props = dict(fontsize=8, alpha=0.5, ha="center", style="italic")
ax.text(xlim[0] + 0.15 * (xlim[1] - xlim[0]), ylim[1] - 0.05 * (ylim[1] - ylim[0]),
        "Low surprisal + High diversity\n(BEST for RLVR?)", **props, color="green")
ax.text(xlim[1] - 0.15 * (xlim[1] - xlim[0]), ylim[0] + 0.08 * (ylim[1] - ylim[0]),
        "High surprisal + Low diversity\n(WORST for RLVR?)", **props, color="red")

fig.tight_layout()
out_path = Path("results/figures/cartography_surprisal_vs_evc.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {out_path}")
