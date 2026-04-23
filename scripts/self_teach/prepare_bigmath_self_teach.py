"""Download Big-Math-RL-Verified and prepare it for self-teach GRPO training.

Filters to problems in the target difficulty range based on llama8b_solve_rate
and converts to VERL format with interaction_kwargs. Outputs a single dataset
(no train/test split) for further filtering based on Qwen pass@k performance.

Usage:
    python scripts/self_teach/prepare_bigmath_self_teach.py
    python scripts/self_teach/prepare_bigmath_self_teach.py --min-solve-rate 0 --max-solve-rate 0.4
    python scripts/self_teach/prepare_bigmath_self_teach.py --max-problems 10000
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the problem and provide your final answer. "
    "Put your final answer within \\boxed{}."
)


def convert_to_verl(
    rows: list[dict],
    data_source_name: str = "bigmath",
) -> pa.Table:
    """Convert filtered rows to VERL-compatible format for self-teach."""
    out = {
        "data_source": [],
        "prompt": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
    }

    for i, row in enumerate(rows):
        question = row["problem"]
        answer = row["answer"]
        source = row.get("source", "unknown")
        solve_rate = row.get("llama8b_solve_rate", -1)

        out["data_source"].append(data_source_name)
        out["prompt"].append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ])
        out["ability"].append(source)
        out["reward_model"].append({
            "ground_truth": answer,
        })
        out["extra_info"].append({
            "index": i,
            "source": source,
            "llama8b_solve_rate": solve_rate,
            "interaction_kwargs": {
                "name": "self_teach",
                "ground_truth": answer,
                "data_source": data_source_name,
            },
        })

    return pa.Table.from_pydict(out)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Big-Math-RL-Verified for self-teach training"
    )
    parser.add_argument(
        "--min-solve-rate", type=float, default=0.0,
        help="Minimum llama8b_solve_rate (inclusive)",
    )
    parser.add_argument(
        "--max-solve-rate", type=float, default=0.4,
        help="Maximum llama8b_solve_rate (inclusive)",
    )
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="Cap total number of problems (sample randomly after filtering)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/bigmath",
        help="Output directory",
    )
    args = parser.parse_args()

    # Download dataset
    print("Loading Big-Math-RL-Verified from HuggingFace...")
    ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
    print(f"Total problems: {len(ds)}")

    # Print solve rate distribution before filtering
    solve_rates = ds["llama8b_solve_rate"]
    print(f"\nSolve rate distribution (all {len(ds)} problems):")
    brackets = [(0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in brackets:
        count = sum(1 for r in solve_rates if r is not None and lo <= r < hi)
        print(f"  [{lo:.1f}, {hi:.1f}): {count:>6d} ({count/len(ds)*100:.1f}%)")

    # Filter by solve rate
    filtered = ds.filter(
        lambda x: x["llama8b_solve_rate"] is not None and args.min_solve_rate <= x["llama8b_solve_rate"] <= args.max_solve_rate
    )
    print(f"\nAfter filtering [{args.min_solve_rate}, {args.max_solve_rate}]: {len(filtered)} problems")

    # Shuffle
    filtered = filtered.shuffle(seed=args.seed)

    # Cap if requested
    if args.max_problems and len(filtered) > args.max_problems:
        filtered = filtered.select(range(args.max_problems))
        print(f"Capped to {args.max_problems} problems")

    all_rows = [filtered[i] for i in range(len(filtered))]

    # Print source distribution
    from collections import Counter
    sources = Counter(r.get("source", "unknown") for r in all_rows)
    print(f"\nSource distribution:")
    for source, count in sources.most_common():
        print(f"  {source:30s}: {count:>5d}")

    # Convert to VERL format
    verl_table = convert_to_verl(all_rows)

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(all_rows)

    # Raw format (question/solution columns for eval script compatibility)
    raw_table = pa.Table.from_pydict({
        "question": [r["problem"] for r in all_rows],
        "solution": [r["answer"] for r in all_rows],
        "source": [r.get("source", "unknown") for r in all_rows],
        "llama8b_solve_rate": [r.get("llama8b_solve_rate", -1) for r in all_rows],
    })

    pq.write_table(raw_table, out_dir / "bigmath_filtered.parquet")
    pq.write_table(verl_table, out_dir / "bigmath_filtered_verl.parquet")

    print(f"\nSaved to {out_dir}/:")
    print(f"  bigmath_filtered.parquet      ({n} rows, raw format)")
    print(f"  bigmath_filtered_verl.parquet ({n} rows, VERL format)")

    # Summary stats
    rates = [r.get("llama8b_solve_rate", 0) for r in all_rows]
    print(f"\nSolve rate stats:")
    print(f"  mean: {sum(rates)/len(rates):.3f}")
    print(f"  min:  {min(rates):.3f}")
    print(f"  max:  {max(rates):.3f}")


if __name__ == "__main__":
    main()
