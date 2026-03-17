#!/usr/bin/env python3
"""Prepare validation dataset with both normal and no-code prompt variants.

Creates 4 variants per question:
1. math_train_cot    — standard "Let's think step by step..." suffix
2. math_nocode_cot   — no-code instruction + "Let's think step by step..."
3. math_qwen_cot     — Qwen-style system message CoT
4. math_no_cot       — raw question only

This lets us track whether the model learns to follow the no-code instruction
while maintaining (or improving) its math reasoning ability.
"""

import pandas as pd
from pathlib import Path
import copy

TRAIN_COT_SUFFIX = r" Let's think step by step and output the final answer within \boxed{}."
NOCODE_SUFFIX = " You do not have access to a Python interpreter or any code execution environment. Solve this problem using mathematical reasoning only."
QWEN_COT_SYSTEM_MSG = r"Please reason step by step, and put your final answer within \boxed{}."

VARIANTS = {
    "math_train_cot": lambda q: [{"role": "user", "content": q + TRAIN_COT_SUFFIX}],
    "math_nocode_cot": lambda q: [{"role": "user", "content": q + NOCODE_SUFFIX + TRAIN_COT_SUFFIX}],
    "math_qwen_cot": lambda q: [
        {"role": "system", "content": QWEN_COT_SYSTEM_MSG},
        {"role": "user", "content": q},
    ],
    "math_no_cot": lambda q: [{"role": "user", "content": q}],
}


def get_user_content(prompt_msgs) -> str:
    if not isinstance(prompt_msgs, list):
        prompt_msgs = [dict(m) for m in prompt_msgs]
    user_msgs = [m for m in prompt_msgs if m.get("role") == "user"]
    return user_msgs[-1]["content"] if user_msgs else prompt_msgs[-1]["content"]


def create_nocode_val(input_path: Path, output_path: Path):
    df = pd.read_parquet(input_path)
    print(f"Input: {input_path} ({len(df)} rows)")

    rows = []
    for _, row in df.iterrows():
        original_content = get_user_content(row["prompt"])

        for data_source, build_prompt in VARIANTS.items():
            new_row = {}
            for col in df.columns:
                val = row[col]
                if hasattr(val, 'as_py'):
                    new_row[col] = val.as_py()
                elif hasattr(val, 'tolist'):
                    new_row[col] = val.tolist()
                elif isinstance(val, (dict, list)):
                    new_row[col] = copy.deepcopy(val)
                else:
                    new_row[col] = val

            new_row["prompt"] = build_prompt(original_content)
            new_row["data_source"] = data_source
            rows.append(new_row)

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path)

    print(f"Output: {output_path} ({len(out_df)} rows)")
    for src in VARIANTS:
        print(f"  {src}: {len(out_df[out_df['data_source'] == src])} rows")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./data/math500.parquet")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / "math500_nocode_val.parquet"
    create_nocode_val(input_path, output_path)
