#!/usr/bin/env python3
"""Prepare π13 dataset for 1-shot RLVR training.

Based on Table 21 from the paper "Reinforcement Learning for Reasoning in
Large Language Models with One Training Example" (arXiv:2504.20571v3).

π13 is a geometry problem about circle equations. The answer is 4/3.
The paper requires duplicating the single example to batch_size (128) copies.
"""

import pandas as pd
from pathlib import Path

# π13 problem from Table 21 of the paper
PI13_PROMPT = """Given that circle $C$ passes through points $P(0,-4)$, $Q(2,0)$, and $R(3,-1)$.
$(1)$ Find the equation of circle $C$;
$(2)$ If line $l: mx+y-1=0$ intersects circle $C$ at points $A$ and $B$, and $|AB|=4$, find the value of $m$. Let's think step by step and output the final answer within \\boxed{}."""

PI13_GROUND_TRUTH = "4/3"


def create_pi13_dataset(output_dir: Path, num_copies: int = 128):
    """Create π13 dataset with duplicates for batch size requirements.

    Args:
        output_dir: Directory to save the parquet file
        num_copies: Number of times to duplicate the example (paper uses 128)
    """
    # Create verl-compatible data format
    verl_data = []

    for idx in range(num_copies):
        item = {
            "data_source": "math",  # Use "math" so reward function uses flexible comparison
            "prompt": [
                {
                    "role": "user",
                    "content": PI13_PROMPT
                }
            ],
            "ability": "math",
            "reward_model": {
                "ground_truth": PI13_GROUND_TRUTH,
                "task_type": "geometry",
            },
            "extra_info": {
                "index": idx,
                "source": "pi13",
                "paper_ref": "Table 21, arXiv:2504.20571v3",
            },
        }
        verl_data.append(item)

    # Save as parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(verl_data)

    output_path = output_dir / f"pi13_r{num_copies}.parquet"
    df.to_parquet(output_path)

    print(f"Created π13 dataset:")
    print(f"  Rows: {len(df)}")
    print(f"  Ground truth: {PI13_GROUND_TRUTH}")
    print(f"  Output: {output_path}")
    print(f"\nFirst row preview:")
    print(f"  data_source: {df.iloc[0]['data_source']}")
    print(f"  prompt: {df.iloc[0]['prompt'][0]['content'][:100]}...")
    print(f"  ground_truth: {df.iloc[0]['reward_model']['ground_truth']}")

    return output_path


def verify_existing_file(filepath: Path):
    """Verify an existing pi13 parquet file."""
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return False

    df = pd.read_parquet(filepath)
    print(f"Verifying {filepath}:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    if len(df) > 0:
        first_row = df.iloc[0]
        print(f"  First row data_source: {first_row.get('data_source', 'N/A')}")

        if 'reward_model' in first_row:
            rm = first_row['reward_model']
            if isinstance(rm, dict):
                print(f"  Ground truth: {rm.get('ground_truth', 'N/A')}")

        if 'prompt' in first_row:
            prompt = first_row['prompt']
            if isinstance(prompt, list) and len(prompt) > 0:
                content = prompt[0].get('content', '')
                print(f"  Prompt preview: {content[:100]}...")

    return len(df) == 128


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare π13 dataset")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--copies", type=int, default=128, help="Number of duplicates")
    parser.add_argument("--verify", action="store_true", help="Verify existing file instead of creating")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.verify:
        filepath = output_dir / f"pi13_r{args.copies}.parquet"
        verify_existing_file(filepath)
    else:
        create_pi13_dataset(output_dir, args.copies)
