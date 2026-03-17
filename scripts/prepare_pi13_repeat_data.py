#!/usr/bin/env python3
"""Prepare π13 dataset with 'repeat the question' suffix for 1-shot RLVR training.

Same problem as pi13 but with a suffix instructing the model to first repeat
the question word by word before solving. This variant was found to reach
high validation scores faster during training.
"""

import pandas as pd
from pathlib import Path

PI13_REPEAT_PROMPT = """Given that circle $C$ passes through points $P(0,-4)$, $Q(2,0)$, and $R(3,-1)$.
$(1)$ Find the equation of circle $C$;
$(2)$ If line $l: mx+y-1=0$ intersects circle $C$ at points $A$ and $B$, and $|AB|=4$, find the value of $m$. First, repeat the question word by word, and then let's think step by step to solve the problem. Output the final answer within \\boxed{}."""

PI13_GROUND_TRUTH = "4/3"


def create_pi13_repeat_dataset(output_dir: Path, num_copies: int = 128):
    verl_data = []
    for idx in range(num_copies):
        item = {
            "data_source": "math",
            "prompt": [{"role": "user", "content": PI13_REPEAT_PROMPT}],
            "ability": "math",
            "reward_model": {
                "ground_truth": PI13_GROUND_TRUTH,
                "task_type": "geometry",
            },
            "extra_info": {
                "index": idx,
                "source": "pi13_repeat",
                "paper_ref": "Table 21, arXiv:2504.20571v3 (repeat suffix variant)",
            },
        }
        verl_data.append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(verl_data)
    output_path = output_dir / f"pi13_repeat_r{num_copies}.parquet"
    df.to_parquet(output_path)

    print(f"Created π13 repeat dataset:")
    print(f"  Rows: {len(df)}")
    print(f"  Ground truth: {PI13_GROUND_TRUTH}")
    print(f"  Output: {output_path}")
    print(f"\nPrompt:\n{PI13_REPEAT_PROMPT}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare π13 repeat dataset")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--copies", type=int, default=128)
    args = parser.parse_args()
    create_pi13_repeat_dataset(Path(args.output_dir), args.copies)
