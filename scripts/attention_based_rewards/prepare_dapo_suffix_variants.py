#!/usr/bin/env python3
"""Generate DAPO-Math-17k parquet with step-by-step suffix appended to user messages.

Reads:  attention_based_rewards/data/dapo_math_17k.parquet
Writes: attention_based_rewards/data/dapo_math_17k_step_suffix.parquet

Usage:
    srun --time=00:10:00 --account=nn12068k --partition=accel \
        python scripts/attention_based_rewards/prepare_dapo_suffix_variants.py
"""

import copy
from pathlib import Path

import pandas as pd

STEP_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."

INPUT_PATH = Path("attention_based_rewards/data/dapo_math_17k.parquet")
OUTPUT_PATH = Path("attention_based_rewards/data/dapo_math_17k_step_suffix.parquet")


def main():
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    modified_prompts = []
    for prompt in df["prompt"]:
        prompt = copy.deepcopy(prompt)
        # Append suffix to last user message
        for msg in reversed(prompt):
            if msg["role"] == "user":
                msg["content"] = msg["content"] + STEP_SUFFIX
                break
        modified_prompts.append(prompt)

    df["prompt"] = modified_prompts
    df.to_parquet(OUTPUT_PATH)

    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"Sample user message (first 200 chars):")
    print(f"  {df.iloc[0]['prompt'][1]['content'][:200]}")

    # Verify suffix is present
    sample = df.iloc[0]["prompt"][1]["content"]
    assert sample.endswith(STEP_SUFFIX.strip()), "Suffix not found at end of user message!"
    print("Verification passed.")


if __name__ == "__main__":
    main()
