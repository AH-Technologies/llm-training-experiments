"""Filter bigmath dataset by Qwen3-8B pass@k solve rates.

Takes eval results (possibly sharded) and the raw + VERL parquets, filters
to problems with solve_rate in (0, 0.5), and outputs train/test splits.

Usage:
    python scripts/self_teach/filter_bigmath_by_eval.py
    python scripts/self_teach/filter_bigmath_by_eval.py --min-solve-rate 0.0 --max-solve-rate 0.5
    python scripts/self_teach/filter_bigmath_by_eval.py --eval-results data/bigmath/eval_results_k8_shard0.json data/bigmath/eval_results_k8_shard1.json
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the problem and provide your final answer. "
    "Put your final answer within \\boxed{}."
)


def main():
    parser = argparse.ArgumentParser(description="Filter bigmath by Qwen eval results")
    parser.add_argument(
        "--eval-results", nargs="+",
        default=["data/bigmath/eval_results_k8.json"],
        help="Path(s) to eval result JSON files (supports multiple shards)",
    )
    parser.add_argument("--data", default="data/bigmath/bigmath_filtered.parquet")
    parser.add_argument("--min-solve-rate", type=float, default=0.0,
                        help="Minimum solve rate (exclusive, >0 means at least one correct)")
    parser.add_argument("--max-solve-rate", type=float, default=0.5,
                        help="Maximum solve rate (exclusive)")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/bigmath")
    args = parser.parse_args()

    # Load and merge eval results
    all_results = []
    for path in args.eval_results:
        with open(path) as f:
            data = json.load(f)
        all_results.extend(data["results"])
        print(f"Loaded {len(data['results'])} results from {path}")
    print(f"Total eval results: {len(all_results)}")

    # Load raw parquet
    table = pq.read_table(args.data)
    questions = table.column("question").to_pylist()
    solutions = table.column("solution").to_pylist()
    sources = table.column("source").to_pylist()
    solve_rates_llama = table.column("llama8b_solve_rate").to_pylist()

    assert len(all_results) == len(questions), (
        f"Eval results ({len(all_results)}) != dataset ({len(questions)}). "
        f"Make sure all shards are included."
    )

    # Filter by Qwen solve rate
    keep_indices = []
    for i, result in enumerate(all_results):
        sr = result["solve_rate"]
        if sr > args.min_solve_rate and sr < args.max_solve_rate:
            keep_indices.append(i)

    print(f"\nFiltering to solve_rate in ({args.min_solve_rate}, {args.max_solve_rate}):")
    print(f"  Kept: {len(keep_indices)} / {len(all_results)} ({len(keep_indices)/len(all_results)*100:.1f}%)")

    # Solve rate distribution of kept problems
    kept_rates = [all_results[i]["solve_rate"] for i in keep_indices]
    print(f"\nKept problems solve rate distribution:")
    brackets = [(0.001, 0.125), (0.125, 0.25), (0.25, 0.375), (0.375, 0.5)]
    for lo, hi in brackets:
        count = sum(1 for r in kept_rates if lo <= r < hi)
        print(f"  [{lo:.3f}, {hi:.3f}): {count:>5d}")

    # Shuffle keep_indices for random train/test split
    import random
    rng = random.Random(args.seed)
    rng.shuffle(keep_indices)

    n_test = max(1, int(len(keep_indices) * args.test_fraction))
    n_train = len(keep_indices) - n_test
    train_indices = keep_indices[:n_train]
    test_indices = keep_indices[n_train:]
    print(f"\nSplit: {n_train} train / {n_test} test")

    # Build output tables
    out_dir = Path(args.out_dir)

    for split, indices in [("train", train_indices), ("test", test_indices)]:
        rows_raw = {
            "question": [questions[i] for i in indices],
            "solution": [solutions[i] for i in indices],
            "source": [sources[i] for i in indices],
            "llama8b_solve_rate": [solve_rates_llama[i] for i in indices],
            "qwen8b_solve_rate": [all_results[i]["solve_rate"] for i in indices],
        }

        rows_verl = {
            "data_source": [],
            "prompt": [],
            "ability": [],
            "reward_model": [],
            "extra_info": [],
        }
        for j, i in enumerate(indices):
            rows_verl["data_source"].append("bigmath")
            rows_verl["prompt"].append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": questions[i]},
            ])
            rows_verl["ability"].append(sources[i])
            rows_verl["reward_model"].append({"ground_truth": solutions[i]})
            rows_verl["extra_info"].append({
                "index": j,
                "source": sources[i],
                "llama8b_solve_rate": solve_rates_llama[i],
                "qwen8b_solve_rate": all_results[i]["solve_rate"],
                "interaction_kwargs": {
                    "name": "self_teach",
                    "ground_truth": solutions[i],
                    "data_source": "bigmath",
                },
            })

        pq.write_table(pa.Table.from_pydict(rows_raw), out_dir / f"bigmath_{split}.parquet")
        pq.write_table(pa.Table.from_pydict(rows_verl), out_dir / f"bigmath_{split}_verl.parquet")

    print(f"\nSaved to {out_dir}/:")
    print(f"  bigmath_train.parquet         ({n_train} rows)")
    print(f"  bigmath_test.parquet          ({n_test} rows)")
    print(f"  bigmath_train_verl.parquet    ({n_train} rows)")
    print(f"  bigmath_test_verl.parquet     ({n_test} rows)")


if __name__ == "__main__":
    main()
