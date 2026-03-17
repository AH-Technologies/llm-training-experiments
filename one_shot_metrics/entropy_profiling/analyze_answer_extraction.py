#!/usr/bin/env python3
"""Analyze answer extraction success rates across all rollout pickles."""

import pickle
from pathlib import Path
from collections import Counter

RESULT_DIRS = {
    "Wang 14": Path("results/entropy_profiles"),
    "Large 659": Path("results/large_run/entropy_profiles"),
    "Li 9": Path("results/li_run/entropy_profiles"),
}

for run_name, results_dir in RESULT_DIRS.items():
    pkl_files = sorted(results_dir.glob("entropy_*.pkl"))
    if not pkl_files:
        continue

    print(f"\n{'='*80}")
    print(f"  {run_name} ({len(pkl_files)} examples)")
    print(f"{'='*80}")
    print(f"{'Example':<30} {'Total':>6} {'Extracted':>10} {'None':>6} {'%Ext':>7} {'Correct':>8} {'%Corr':>7} {'UniqueAns':>10}")
    print("-" * 95)

    total_rollouts = 0
    total_extracted = 0
    total_none = 0
    total_correct = 0

    per_example = []

    for pkl_path in pkl_files:
        name = pkl_path.stem.replace("entropy_", "")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        rollouts = data["rollouts"]
        n = len(rollouts)

        extracted = [r for r in rollouts if r.get("extracted_answer") is not None]
        none_ans = [r for r in rollouts if r.get("extracted_answer") is None]
        correct = [r for r in rollouts if r.get("is_correct", False)]

        n_ext = len(extracted)
        n_none = len(none_ans)
        n_corr = len(correct)

        # Count unique answers (excluding None)
        answers = [r.get("extracted_answer") for r in rollouts if r.get("extracted_answer") is not None]
        unique_answers = len(set(answers))

        pct_ext = 100.0 * n_ext / n if n > 0 else 0
        pct_corr = 100.0 * n_corr / n if n > 0 else 0

        print(f"{name:<30} {n:>6} {n_ext:>10} {n_none:>6} {pct_ext:>6.1f}% {n_corr:>8} {pct_corr:>6.1f}% {unique_answers:>10}")

        total_rollouts += n
        total_extracted += n_ext
        total_none += n_none
        total_correct += n_corr

        per_example.append({
            "name": name,
            "n": n,
            "n_extracted": n_ext,
            "n_none": n_none,
            "pct_extracted": pct_ext,
            "n_correct": n_corr,
            "pct_correct": pct_corr,
            "unique_answers": unique_answers,
        })

    print("-" * 95)
    pct_ext_total = 100.0 * total_extracted / total_rollouts if total_rollouts > 0 else 0
    pct_corr_total = 100.0 * total_correct / total_rollouts if total_rollouts > 0 else 0
    print(f"{'TOTAL':<30} {total_rollouts:>6} {total_extracted:>10} {total_none:>6} {pct_ext_total:>6.1f}% {total_correct:>8} {pct_corr_total:>6.1f}%")

    # Breakdown: of the ones where extraction failed, what do the responses look like?
    print(f"\n  --- Extraction failure analysis (sampled) ---")
    # Check a few None examples for common patterns
    for pkl_path in pkl_files[:3]:  # Check first 3 examples
        name = pkl_path.stem.replace("entropy_", "")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        rollouts = data["rollouts"]
        none_rollouts = [r for r in rollouts if r.get("extracted_answer") is None]

        if not none_rollouts:
            continue

        # Analyze text patterns in failed extractions
        patterns = Counter()
        for r in none_rollouts:
            text = r.get("text", "")
            if "\\boxed" in text:
                patterns["has_boxed_but_failed"] += 1
            elif "boxed" in text.lower():
                patterns["has_boxed_lowercase"] += 1
            elif len(text.strip()) < 50:
                patterns["very_short_response"] += 1
            elif "answer" in text.lower() and ("is" in text.lower() or "=" in text.lower()):
                patterns["has_answer_no_boxed"] += 1
            else:
                patterns["other_no_boxed"] += 1

        if patterns:
            print(f"\n  {name} ({len(none_rollouts)} failed extractions):")
            for pattern, count in patterns.most_common():
                print(f"    {pattern}: {count} ({100*count/len(none_rollouts):.1f}%)")

    # Show some actual failed extraction texts
    print(f"\n  --- Sample failed extraction texts ---")
    for pkl_path in pkl_files[:2]:
        name = pkl_path.stem.replace("entropy_", "")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        none_rollouts = [r for r in data["rollouts"] if r.get("extracted_answer") is None]
        if none_rollouts:
            print(f"\n  [{name}] Example failed text (last 200 chars):")
            sample = none_rollouts[0]
            text = sample.get("text", "")
            print(f"    ...{text[-200:]}")
            if len(none_rollouts) > 1:
                text2 = none_rollouts[1].get("text", "")
                print(f"    ...{text2[-200:]}")


if __name__ == "__main__":
    pass
