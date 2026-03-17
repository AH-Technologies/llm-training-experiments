#!/usr/bin/env python3
"""Correlate Dataset Cartography metrics (variability, confidence) from
acc_step_500.json with actual one-shot MATH500 scores for Wang 14 examples."""

import json
import numpy as np
from scipy import stats

# Wang 14 examples: name -> (index_in_acc_step_500, math500_score)
wang_examples = {
    "pi_1": (124, 74.0),
    "pi_2": (267, 70.6),
    "pi_4": (931, 65.6),
    "pi_7": (568, 64.0),
    "pi_11": (875, 64.0),
    "pi_13": (682, 74.4),
    "pi_16": (50, 67.0),
    "pi_17": (870, 67.2),
    "pi_605": (60, 71.8),
    "pi_606": (906, 64.4),
    "pi_1201": (847, 71.4),
    "pi_1207": (174, 54.0),
    "pi_1208": (770, 45.0),
    "pi_1209": (789, 72.2),
}

ACC_JSON = "one_shot_metrics/One-Shot-RLVR/data/acc_step_500.json"

with open(ACC_JSON) as f:
    acc_data = json.load(f)

print(f"{'Example':<12} {'Index':>6} {'MATH500':>8} {'MeanAcc':>8} {'Std':>8} {'Final':>8} {'Range':>8} {'Slope':>10}")
print("-" * 80)

names, scores, variabilities, means, finals, ranges, slopes = [], [], [], [], [], [], []

for name, (idx, score) in wang_examples.items():
    curve = acc_data[str(idx)]
    mean_acc = np.mean(curve)
    std_acc = np.std(curve)
    final_acc = curve[-1]
    rng = max(curve) - min(curve)
    slope = np.polyfit(np.arange(len(curve)), curve, 1)[0]

    names.append(name)
    scores.append(score)
    variabilities.append(std_acc)
    means.append(mean_acc)
    finals.append(final_acc)
    ranges.append(rng)
    slopes.append(slope)

    print(f"{name:<12} {idx:>6} {score:>8.1f} {mean_acc:>8.3f} {std_acc:>8.4f} {final_acc:>8.3f} {rng:>8.3f} {slope:>10.6f}")

scores = np.array(scores)
variabilities = np.array(variabilities)
means = np.array(means)
finals = np.array(finals)
ranges = np.array(ranges)
slopes = np.array(slopes)
cv = variabilities / (means + 1e-8)

print(f"\n{'='*75}")
print(f"Dataset Cartography Correlations with MATH500 (n={len(scores)})")
print(f"{'='*75}")
print(f"{'Metric':<25} {'Spearman r':>10} {'p-value':>10} {'Pearson r':>10} {'p-value':>10}")
print("-" * 70)

for label, vals in [
    ("variability (std)", variabilities),
    ("confidence (mean_acc)", means),
    ("final_acc", finals),
    ("1 - confidence", 1 - means),
    ("coeff_of_variation", cv),
    ("range (max-min)", ranges),
    ("trend (slope)", slopes),
]:
    r_s, p_s = stats.spearmanr(vals, scores)
    r_p, p_p = stats.pearsonr(vals, scores)
    sig = "***" if p_s < 0.01 else "**" if p_s < 0.05 else "*" if p_s < 0.1 else ""
    print(f"{label:<25} {r_s:>10.3f} {p_s:>10.4f} {r_p:>10.3f} {p_p:>10.4f} {sig}")

# Also print sorted by variability to see the ranking
print(f"\n{'='*75}")
print("Examples sorted by variability (Dataset Cartography ranking)")
print(f"{'='*75}")
sorted_by_var = sorted(zip(names, variabilities, scores, means), key=lambda x: x[1], reverse=True)
for i, (n, v, s, m) in enumerate(sorted_by_var):
    print(f"  {i+1:>2}. {n:<12} var={v:.4f}  confidence={m:.3f}  MATH500={s:.1f}")
