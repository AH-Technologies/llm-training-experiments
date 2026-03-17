#!/usr/bin/env python3
"""Compare all metrics between pi13 original and pi13 repeat variant."""

import pickle
import numpy as np
import random
from collections import Counter


def distinct_n(txts, n):
    all_ng = []
    for t in txts:
        w = t.split()
        all_ng.extend([tuple(w[i:i+n]) for i in range(len(w)-n+1)])
    return len(set(all_ng)) / max(len(all_ng), 1)


def compute_all_metrics(data):
    rollouts = data["rollouts"]
    texts = [r["text"] for r in rollouts]
    answers = [r.get("extracted_answer") for r in rollouts]
    correct = [r for r in rollouts if r.get("is_correct", False)]
    wrong = [r for r in rollouts if not r.get("is_correct", False)]

    m = {}
    m["num_rollouts"] = len(rollouts)
    m["pass_rate"] = len(correct) / len(rollouts)
    m["n_unique_answers"] = len(set(answers))

    # Answer distribution entropy
    counts = Counter(answers)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    m["answer_entropy"] = -sum(p * np.log(p + 1e-12) for p in probs)

    # Entropy-based metrics
    entropy_arrays = [r["entropy_array"] for r in rollouts]
    means = [np.mean(e) for e in entropy_arrays]
    medians = [np.median(e) for e in entropy_arrays]
    stds_per = [np.std(e) for e in entropy_arrays]
    maxes = [np.max(e) for e in entropy_arrays]

    m["mean_entropy_mean"] = np.mean(means)
    m["mean_entropy_std"] = np.std(means)
    m["median_entropy_mean"] = np.mean(medians)
    m["std_entropy_mean"] = np.mean(stds_per)
    m["max_entropy_mean"] = np.mean(maxes)
    m["cross_rollout_entropy_var"] = np.var(means)

    # Spike metrics
    for r in rollouts:
        e = r["entropy_array"]
        threshold = np.mean(e) + 2 * np.std(e)
        r["_n_spikes"] = int(np.sum(e > threshold))
        spike_vals = e[e > threshold]
        r["_mean_spike_mag"] = float(np.mean(spike_vals)) if len(spike_vals) > 0 else 0.0
        spike_pos = np.where(e > threshold)[0]
        r["_mean_spike_pos"] = float(np.mean(spike_pos / len(e))) if len(spike_pos) > 0 else 0.5

    m["num_spikes_mean"] = np.mean([r["_n_spikes"] for r in rollouts])
    m["mean_spike_magnitude_mean"] = np.mean([r["_mean_spike_mag"] for r in rollouts])
    m["mean_spike_position_mean"] = np.mean([r["_mean_spike_pos"] for r in rollouts])

    # High entropy ratio
    for r in rollouts:
        e = r["entropy_array"]
        mv = np.mean(e)
        r["_high_ent_ratio"] = float(np.mean(e > mv + np.std(e)))
    m["high_entropy_ratio_mean"] = np.mean([r["_high_ent_ratio"] for r in rollouts])

    # Early vs late entropy
    for r in rollouts:
        e = r["entropy_array"]
        mid = len(e) // 2
        r["_early"] = float(np.mean(e[:mid])) if mid > 0 else 0.0
        r["_late"] = float(np.mean(e[mid:])) if mid > 0 else 0.0
        r["_trend"] = r["_late"] - r["_early"]
    m["early_mean_mean"] = np.mean([r["_early"] for r in rollouts])
    m["late_mean_mean"] = np.mean([r["_late"] for r in rollouts])
    m["entropy_trend_mean"] = np.mean([r["_trend"] for r in rollouts])

    # Token count stats
    token_counts = [r["num_tokens"] for r in rollouts]
    m["num_tokens_mean"] = np.mean(token_counts)
    m["num_tokens_std"] = np.std(token_counts)

    # Text diversity
    m["distinct_1"] = distinct_n(texts, 1)
    m["distinct_2"] = distinct_n(texts, 2)
    m["distinct_3"] = distinct_n(texts, 3)
    m["distinct_4"] = distinct_n(texts, 4)

    # Jaccard
    rng = random.Random(42)
    jacs = []
    for _ in range(500):
        i = rng.randint(0, len(texts) - 1)
        j = rng.randint(0, len(texts) - 1)
        if i == j:
            continue
        s1, s2 = set(texts[i].split()), set(texts[j].split())
        if s1 | s2:
            jacs.append(len(s1 & s2) / len(s1 | s2))
    m["jaccard_mean"] = np.mean(jacs)

    # Correct-only diversity
    if len(correct) >= 2:
        ct = [r["text"] for r in correct]
        m["distinct_4_correct"] = distinct_n(ct, 4)
        cl = [len(t.split()) for t in ct]
        m["length_std_correct"] = np.std(cl)
        m["length_cv_correct"] = np.std(cl) / max(np.mean(cl), 1)
        ce = [np.mean(r["entropy_array"]) for r in correct]
        m["entropy_var_correct"] = np.var(ce)
    else:
        m["distinct_4_correct"] = float("nan")
        m["length_std_correct"] = float("nan")
        m["length_cv_correct"] = float("nan")
        m["entropy_var_correct"] = float("nan")

    # RSR metrics (load from CSV if available)
    return m


# Load RSR separately
def load_rsr(csv_path, name):
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        row = df[df["example"] == name]
        if len(row) == 0:
            return {}
        row = row.iloc[0]
        return {
            "rsr_mean": row["rsr_mean"],
            "rsr_std": row["rsr_std"],
            "mean_rank_mean": row["mean_rank_mean"],
            "mean_surprisal_mean": row["mean_surprisal_mean"],
        }
    except Exception:
        return {}


with open("results/entropy_profiles/entropy_pi_13.pkl", "rb") as f:
    orig_data = pickle.load(f)
with open("results/pi13_repeat/entropy_profiles/entropy_pi_13_repeat.pkl", "rb") as f:
    repeat_data = pickle.load(f)

m_orig = compute_all_metrics(orig_data)
m_repeat = compute_all_metrics(repeat_data)

# Add RSR
m_orig.update(load_rsr("results/entropy_profiles/rsr_metrics.csv", "pi_13"))
m_repeat.update(load_rsr("results/pi13_repeat/entropy_profiles/rsr_metrics.csv", "pi_13_repeat"))

# Print sorted by relative difference
header = "rel_diff"
print(f"{'metric':<35} {'original':>12} {'repeat':>12} {'diff':>12} {header:>10}")
print("-" * 85)

diffs = []
for k in m_orig:
    o = m_orig[k]
    r = m_repeat.get(k)
    if r is None:
        continue
    if isinstance(o, float) and (np.isnan(o) or np.isnan(r)):
        continue
    diff = r - o
    denom = abs(o) if abs(o) > 1e-8 else 1e-8
    rel = abs(diff) / denom * 100
    diffs.append((rel, k, o, r, diff))

diffs.sort(reverse=True)
for rel, k, o, r, diff in diffs:
    marker = " <<<" if rel > 20 else ""
    print(f"{k:<35} {o:>12.4f} {r:>12.4f} {diff:>+12.4f} {rel:>9.1f}%{marker}")
