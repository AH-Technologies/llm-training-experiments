#!/usr/bin/env python3
"""Prepare π1 dataset for 1-shot RLVR training.

Based on Table 2 from the paper "Reinforcement Learning for Reasoning in
Large Language Models with One Training Example" (arXiv:2504.20571v3).

π1 is the example with highest historical variance score from DSR-sub.
It's a physics/algebra problem about wind pressure on a sail. The answer is 12.8.
"""

import pandas as pd
from pathlib import Path

# π1 problem from Table 2 of the paper
PI1_PROMPT = r"""The pressure $P$ exerted by wind on a sail varies jointly as the area $A$ of the sail and the cube of the wind's velocity $V$. When the velocity is $8$ miles per hour, the pressure on a sail of $2$ square feet is $4$ pounds. Find the wind velocity when the pressure on $4$ square feet of sail is $32$ pounds. Let's think step by step and output the final answer within \boxed{}."""

PI1_GROUND_TRUTH = "12.8"


def create_pi1_dataset(output_dir: Path, num_copies: int = 128):
    """Create π1 dataset with duplicates for batch size requirements.

    Args:
        output_dir: Directory to save the parquet file
        num_copies: Number of times to duplicate the example (paper uses 128)
    """
    # Create verl-compatible data format
    verl_data = []

    for idx in range(num_copies):
        item = {
            "data_source": "math",
            "prompt": [
                {
                    "role": "user",
                    "content": PI1_PROMPT
                }
            ],
            "ability": "math",
            "reward_model": {
                "ground_truth": PI1_GROUND_TRUTH,
                "task_type": "algebra",
            },
            "extra_info": {
                "index": idx,
                "source": "pi1",
                "paper_ref": "Table 2, arXiv:2504.20571v3",
            },
        }
        verl_data.append(item)

    # Save as parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(verl_data)

    output_path = output_dir / f"pi1_r{num_copies}.parquet"
    df.to_parquet(output_path)

    print(f"Created π1 dataset:")
    print(f"  Rows: {len(df)}")
    print(f"  Ground truth: {PI1_GROUND_TRUTH}")
    print(f"  Output: {output_path}")
    print(f"\nFirst row preview:")
    print(f"  data_source: {df.iloc[0]['data_source']}")
    print(f"  prompt: {df.iloc[0]['prompt'][0]['content'][:100]}...")
    print(f"  ground_truth: {df.iloc[0]['reward_model']['ground_truth']}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare π1 dataset")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--copies", type=int, default=128, help="Number of duplicates")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    create_pi1_dataset(output_dir, args.copies)