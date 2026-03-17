#!/usr/bin/env python3
"""Compute reasoning path diversity metrics for rollout pickles.

Measures how diverse the model's reasoning strategies are across rollouts.
Hypothesis: examples with more diverse reasoning paths are better for RLVR.
"""

import pickle
import numpy as np
from collections import Counter
from pathlib import Path
from scipy import stats
import pandas as pd


def distinct_n(texts, n):
    """Fraction of unique n-grams across all texts."""
    all_ngrams = []
    for t in texts:
        words = t.split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        all_ngrams.extend(ngrams)
    return len(set(all_ngrams)) / max(len(all_ngrams), 1)


def pairwise_jaccard(texts, n_pairs=500, seed=42):
    """Average pairwise Jaccard similarity of word sets."""
    import random
    rng = random.Random(seed)
    jaccards = []
    for _ in range(n_pairs):
        i, j = rng.randint(0, len(texts)-1), rng.randint(0, len(texts)-1)
        if i == j:
            continue
        s1 = set(texts[i].split())
        s2 = set(texts[j].split())
        if len(s1 | s2) == 0:
            continue
        jaccards.append(len(s1 & s2) / len(s1 | s2))
    return np.mean(jaccards) if jaccards else 0.0


def answer_entropy(answers):
    """Shannon entropy of the answer distribution."""
    counts = Counter(answers)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log(p + 1e-12) for p in probs)


def compute_diversity_metrics(data):
    """Compute diversity metrics for a single example's rollouts."""
    rollouts = data["rollouts"]
    texts = [r["text"] for r in rollouts]
    answers = [r.get("extracted_answer", None) for r in rollouts]
    correct_texts = [r["text"] for r in rollouts if r.get("is_correct", False)]
    wrong_texts = [r["text"] for r in rollouts if not r.get("is_correct", False)]

    metrics = {}

    # Answer diversity
    metrics["n_unique_answers"] = len(set(answers))
    metrics["answer_entropy"] = answer_entropy(answers)
    metrics["n_unique_wrong_answers"] = len(set(a for a, r in zip(answers, rollouts) if not r.get("is_correct", False)))

    # Text diversity (all rollouts)
    metrics["jaccard_all"] = pairwise_jaccard(texts)
    for n in [1, 2, 3, 4]:
        metrics[f"distinct_{n}_all"] = distinct_n(texts, n)

    # Text diversity among CORRECT rollouts only
    if len(correct_texts) >= 2:
        metrics["jaccard_correct"] = pairwise_jaccard(correct_texts)
        for n in [1, 2, 3, 4]:
            metrics[f"distinct_{n}_correct"] = distinct_n(correct_texts, n)
        # Length diversity of correct answers
        correct_lengths = [len(t.split()) for t in correct_texts]
        metrics["length_std_correct"] = np.std(correct_lengths)
        metrics["length_cv_correct"] = np.std(correct_lengths) / max(np.mean(correct_lengths), 1)
    else:
        metrics["jaccard_correct"] = np.nan
        for n in [1, 2, 3, 4]:
            metrics[f"distinct_{n}_correct"] = np.nan
        metrics["length_std_correct"] = np.nan
        metrics["length_cv_correct"] = np.nan

    # Text diversity among WRONG rollouts
    if len(wrong_texts) >= 2:
        metrics["jaccard_wrong"] = pairwise_jaccard(wrong_texts)
    else:
        metrics["jaccard_wrong"] = np.nan

    # Length diversity (all)
    all_lengths = [len(t.split()) for t in texts]
    metrics["length_mean"] = np.mean(all_lengths)
    metrics["length_std"] = np.std(all_lengths)
    metrics["length_cv"] = np.std(all_lengths) / max(np.mean(all_lengths), 1)

    # Token-level diversity: how different are rollouts at the token level?
    # Use entropy arrays if available
    if "entropy_array" in rollouts[0]:
        entropy_arrays = [r["entropy_array"] for r in rollouts]
        # Variance of mean entropy across rollouts
        mean_entropies = [np.mean(e) for e in entropy_arrays]
        metrics["entropy_variance_across_rollouts"] = np.var(mean_entropies)

    return metrics


def main():
    results_dir = Path("results/entropy_profiles")
    pkl_files = sorted(results_dir.glob("entropy_*.pkl"))

    # Also include pi13_repeat
    repeat_pkl = Path("results/pi13_repeat/entropy_profiles/entropy_pi_13_repeat.pkl")

    all_metrics = []

    for pkl_path in list(pkl_files) + ([repeat_pkl] if repeat_pkl.exists() else []):
        name = pkl_path.stem.replace("entropy_", "")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        example = data["example"]
        metrics = compute_diversity_metrics(data)
        metrics["example"] = name
        metrics["pass_rate"] = data.get("pass_rate", sum(r.get("is_correct", False) for r in data["rollouts"]) / len(data["rollouts"]))
        metrics["math500_score"] = example.get("math500_score")
        all_metrics.append(metrics)

        print(f"{name}: {metrics['n_unique_answers']} unique answers, "
              f"answer_entropy={metrics['answer_entropy']:.2f}, "
              f"jaccard_all={metrics['jaccard_all']:.3f}, "
              f"distinct_4={metrics['distinct_4_all']:.4f}, "
              f"jaccard_correct={metrics.get('jaccard_correct', 'N/A')}")

    df = pd.DataFrame(all_metrics)

    # Correlations with MATH500
    score_col = "math500_score"
    diversity_cols = [c for c in df.columns if c not in ["example", "math500_score", "pass_rate"]]

    wang_df = df[df[score_col].notna()].copy()
    print(f"\n{'='*70}")
    print(f"Correlations with MATH500 (n={len(wang_df)})")
    print(f"{'='*70}")
    print(f"{'metric':<35} {'Spearman ρ':>10} {'p-value':>10} {'Pearson r':>10} {'p-value':>10}")
    print("-" * 80)

    corr_results = []
    for col in diversity_cols:
        valid = wang_df[[col, score_col]].dropna()
        if len(valid) < 5:
            continue
        r_s, p_s = stats.spearmanr(valid[col], valid[score_col])
        r_p, p_p = stats.pearsonr(valid[col], valid[score_col])
        sig = "***" if p_s < 0.01 else "**" if p_s < 0.05 else "*" if p_s < 0.1 else ""
        print(f"{col:<35} {r_s:>10.3f} {p_s:>10.4f} {r_p:>10.3f} {p_p:>10.4f} {sig}")
        corr_results.append({"metric": col, "spearman_rho": r_s, "spearman_p": p_s})

    # Save
    csv_path = results_dir / "reasoning_diversity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # Print pi13 vs pi13_repeat comparison
    if repeat_pkl.exists():
        print(f"\n{'='*70}")
        print("pi_13 vs pi_13_repeat comparison")
        print(f"{'='*70}")
        orig = df[df["example"] == "pi_13"].iloc[0]
        repeat = df[df["example"] == "pi_13_repeat"].iloc[0]
        for col in diversity_cols:
            if col in orig and not (isinstance(orig[col], float) and np.isnan(orig[col])):
                o = orig[col]
                r = repeat[col]
                print(f"  {col:<35} orig={o:>10.4f}  repeat={r:>10.4f}")


if __name__ == "__main__":
    main()
