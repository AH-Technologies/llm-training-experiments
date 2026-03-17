#!/usr/bin/env python3
"""Compare pi1, pi13, and pi13_repeat across all metrics."""

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
    m = {}
    m["pass_rate"] = len(correct) / len(rollouts)
    m["n_unique_answers"] = len(set(answers))
    counts = Counter(answers)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    m["answer_entropy"] = -sum(p * np.log(p + 1e-12) for p in probs)

    entropy_arrays = [r["entropy_array"] for r in rollouts]
    means = [np.mean(e) for e in entropy_arrays]
    m["mean_entropy_mean"] = np.mean(means)
    m["mean_entropy_std"] = np.std(means)
    m["cross_rollout_entropy_var"] = np.var(means)
    m["std_entropy_mean"] = np.mean([np.std(e) for e in entropy_arrays])
    m["max_entropy_mean"] = np.mean([np.max(e) for e in entropy_arrays])

    for r in rollouts:
        e = r["entropy_array"]
        threshold = np.mean(e) + 2 * np.std(e)
        r["_n_spikes"] = int(np.sum(e > threshold))
    m["num_spikes_mean"] = np.mean([r["_n_spikes"] for r in rollouts])

    for r in rollouts:
        e = r["entropy_array"]
        mid = len(e) // 2
        r["_early"] = float(np.mean(e[:mid])) if mid > 0 else 0.0
        r["_late"] = float(np.mean(e[mid:])) if mid > 0 else 0.0
        r["_trend"] = r["_late"] - r["_early"]
    m["early_mean_mean"] = np.mean([r["_early"] for r in rollouts])
    m["late_mean_mean"] = np.mean([r["_late"] for r in rollouts])
    m["entropy_trend_mean"] = np.mean([r["_trend"] for r in rollouts])

    token_counts = [r["num_tokens"] for r in rollouts]
    m["num_tokens_mean"] = np.mean(token_counts)
    m["num_tokens_std"] = np.std(token_counts)
    m["distinct_4"] = distinct_n(texts, 4)

    rng = random.Random(42)
    jacs = []
    for _ in range(500):
        i, j = rng.randint(0, len(texts) - 1), rng.randint(0, len(texts) - 1)
        if i == j:
            continue
        s1, s2 = set(texts[i].split()), set(texts[j].split())
        if s1 | s2:
            jacs.append(len(s1 & s2) / len(s1 | s2))
    m["jaccard_mean"] = np.mean(jacs)

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
    return m


def load_rsr(csv_path, name):
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        row = df[df["example"] == name].iloc[0]
        return {
            "rsr_mean": row["rsr_mean"],
            "rsr_std": row["rsr_std"],
            "mean_rank_mean": row["mean_rank_mean"],
            "mean_surprisal_mean": row["mean_surprisal_mean"],
        }
    except Exception:
        return {}


# Load all three
with open("results/entropy_profiles/entropy_pi_1.pkl", "rb") as f:
    pi1_data = pickle.load(f)
with open("results/entropy_profiles/entropy_pi_13.pkl", "rb") as f:
    pi13_data = pickle.load(f)
with open("results/pi13_repeat/entropy_profiles/entropy_pi_13_repeat.pkl", "rb") as f:
    repeat_data = pickle.load(f)

m1 = compute_all_metrics(pi1_data)
m13 = compute_all_metrics(pi13_data)
mr = compute_all_metrics(repeat_data)

m1.update(load_rsr("results/entropy_profiles/rsr_metrics.csv", "pi_1"))
m13.update(load_rsr("results/entropy_profiles/rsr_metrics.csv", "pi_13"))
mr.update(load_rsr("results/pi13_repeat/entropy_profiles/rsr_metrics.csv", "pi_13_repeat"))

# Known MATH500 scores
m1["math500_score"] = 74.0
m13["math500_score"] = 74.4
mr["math500_score"] = 74.4  # repeat trains to similar level

hdr = "{:<30} {:>12} {:>12} {:>12}  {:>14}"
print(hdr.format("metric", "pi_1", "pi_13", "pi13_repeat", "pi1_vs_repeat"))
print("-" * 90)

for k in sorted(m1.keys()):
    v1 = m1.get(k)
    v13 = m13.get(k)
    vr = mr.get(k)
    if v1 is None or vr is None or v13 is None:
        continue
    if isinstance(v1, float) and np.isnan(v1):
        continue
    if isinstance(vr, float) and np.isnan(vr):
        continue

    diff_1_r = abs(v1 - vr)
    diff_1_13 = abs(v1 - v13)
    closer = "<<< SIMILAR" if diff_1_r < diff_1_13 else ""

    row = "{:<30} {:>12.4f} {:>12.4f} {:>12.4f}  {}"
    print(row.format(k, v1, v13, vr, closer))
