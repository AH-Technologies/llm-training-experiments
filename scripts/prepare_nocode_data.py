#!/usr/bin/env python3
"""Prepare pi1/pi13 datasets with no-code suffix for RLVR training."""

import pandas as pd
from pathlib import Path

NOCODE_SUFFIX = " You do not have access to a Python interpreter or any code execution environment. Solve this problem using mathematical reasoning only."

PI1_PROMPT_BASE = r"""The pressure $P$ exerted by wind on a sail varies jointly as the area $A$ of the sail and the cube of the wind's velocity $V$. When the velocity is $8$ miles per hour, the pressure on a sail of $2$ square feet is $4$ pounds. Find the wind velocity when the pressure on $4$ square feet of sail is $32$ pounds."""

PI13_PROMPT_BASE = """Given that circle $C$ passes through points $P(0,-4)$, $Q(2,0)$, and $R(3,-1)$.
$(1)$ Find the equation of circle $C$;
$(2)$ If line $l: mx+y-1=0$ intersects circle $C$ at points $A$ and $B$, and $|AB|=4$, find the value of $m$."""

COT_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."

EXAMPLES = {
    "pi1_nocode": {
        "prompt": PI1_PROMPT_BASE + NOCODE_SUFFIX + COT_SUFFIX,
        "ground_truth": "12.8",
        "task_type": "algebra",
        "source": "pi1_nocode",
    },
    "pi13_nocode": {
        "prompt": PI13_PROMPT_BASE + NOCODE_SUFFIX + COT_SUFFIX,
        "ground_truth": "4/3",
        "task_type": "geometry",
        "source": "pi13_nocode",
    },
}


def create_nocode_dataset(name: str, output_dir: Path, num_copies: int = 128):
    ex = EXAMPLES[name]
    verl_data = []
    for idx in range(num_copies):
        item = {
            "data_source": "math",
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "ability": "math",
            "reward_model": {
                "ground_truth": ex["ground_truth"],
                "task_type": ex["task_type"],
            },
            "extra_info": {
                "index": idx,
                "source": ex["source"],
            },
        }
        verl_data.append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(verl_data)
    output_path = output_dir / f"{name}_r{num_copies}.parquet"
    df.to_parquet(output_path)

    print(f"Created {name} dataset:")
    print(f"  Rows: {len(df)}")
    print(f"  Ground truth: {ex['ground_truth']}")
    print(f"  Output: {output_path}")
    print(f"  Prompt preview: {ex['prompt'][:150]}...")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, choices=list(EXAMPLES.keys()))
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--copies", type=int, default=128)
    args = parser.parse_args()
    create_nocode_dataset(args.name, Path(args.output_dir), args.copies)
