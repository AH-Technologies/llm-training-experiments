#!/usr/bin/env python3
"""Standalone script for computing rollout diversity metrics from existing pickle files.

Usage:
    python compute_diversity_standalone.py --results-dir results/li_run/entropy_profiles
    python compute_diversity_standalone.py --results-dir results/large_run/entropy_profiles
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from rollout_diversity import compute_all_diversity_metrics


def main():
    parser = argparse.ArgumentParser(description="Compute rollout diversity metrics from existing pickles")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing entropy_*.pkl files (e.g. results/li_run/entropy_profiles)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: <results-dir>/diversity_metrics.csv)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pkl_files = sorted(results_dir.glob("entropy_*.pkl"))

    if not pkl_files:
        print(f"ERROR: No entropy_*.pkl files found in {results_dir}")
        return

    print(f"Found {len(pkl_files)} pickle files in {results_dir}")

    rows = []
    for pkl_path in pkl_files:
        name = pkl_path.stem.replace("entropy_", "")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        rollouts = data["rollouts"]
        example = data["example"]
        metrics = compute_all_diversity_metrics(rollouts)

        row = {"example": name}
        # Include available score columns
        for score_key in ["math500_score", "historical_variance", "avg_all"]:
            if example.get(score_key) is not None:
                row[score_key] = example[score_key]
        row["pass_rate"] = float(sum(r.get("is_correct", False) for r in rollouts) / len(rollouts))
        row.update(metrics)
        rows.append(row)

        print(f"  {name}: {len(rollouts)} rollouts, "
              f"answer_entropy={metrics['answer_entropy']:.3f}, "
              f"unique_answers={metrics['num_unique_answers']}, "
              f"clusters={metrics['num_entropy_clusters']}")

    df = pd.DataFrame(rows)

    output_path = Path(args.output) if args.output else results_dir / "diversity_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved diversity metrics to {output_path}")
    print(f"\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
