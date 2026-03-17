#!/usr/bin/env python3
"""Prepare a single-example dataset for 1-shot RLVR training.

Usage:
    python scripts/prepare_example_data.py --name ex_907 --copies 128
    python scripts/prepare_example_data.py --name ex_672 --copies 128
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd


ENTROPY_DIR = Path("one_shot_metrics/entropy_profiling/results/large_run/entropy_profiles")


def create_example_dataset(name: str, output_dir: Path, num_copies: int = 128):
    pkl_path = ENTROPY_DIR / f"entropy_{name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    example = data["example"]
    prompt_text = example["prompt_text"]
    ground_truth = example["ground_truth"]

    print(f"Example: {name}")
    print(f"Ground truth: {ground_truth}")
    print(f"Prompt: {prompt_text[:120]}...")

    verl_data = []
    for idx in range(num_copies):
        item = {
            "data_source": "math",
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "math",
            "reward_model": {
                "ground_truth": ground_truth,
                "task_type": "math",
            },
            "extra_info": {
                "index": idx,
                "source": name,
            },
        }
        verl_data.append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(verl_data)
    output_path = output_dir / f"{name}_r{num_copies}.parquet"
    df.to_parquet(output_path)

    print(f"Created {output_path} ({len(df)} rows)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare single-example dataset")
    parser.add_argument("--name", type=str, required=True, help="Example name (e.g. ex_907)")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--copies", type=int, default=128, help="Number of duplicates")
    args = parser.parse_args()

    create_example_dataset(args.name, Path(args.output_dir), args.copies)
