#!/usr/bin/env python3
"""Prepare DAPO-Math-17k as verl-format parquet for GRPO training.

Downloads open-r1/DAPO-Math-17k-Processed (en config, ~14.1K English problems)
and converts to verl format with our system prompt.

Produces:
  - attention_based_rewards/data/dapo_math_17k.parquet

Usage:
    srun --time=00:10:00 --account=nn12068k --partition=accel \
        python scripts/attention_based_rewards/prepare_dapo_verl.py
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path

# System prompt from Attention Illuminates paper
SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


def main():
    print("Loading DAPO-Math-17k-Processed (en config)...")
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    print(f"  {len(ds)} English problems loaded")

    # Inspect schema
    print(f"  Columns: {ds.column_names}")
    print(f"  Sample prompt type: {type(ds[0]['prompt'])}")
    print(f"  Sample: {str(ds[0]['prompt'])[:200]}")

    verl_data = []
    for i, example in enumerate(ds):
        # Extract problem text from prompt field
        # DAPO-Math-17k-Processed may have prompt as string or list
        prompt_field = example["prompt"]
        if isinstance(prompt_field, list):
            # Already in chat format — extract user message
            problem_text = None
            for msg in prompt_field:
                if msg["role"] == "user":
                    problem_text = msg["content"]
                    break
            if problem_text is None:
                problem_text = prompt_field[-1]["content"]
        else:
            problem_text = prompt_field

        # Build chat-format prompt with our system prompt
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ]

        # Ground truth from reward_model field
        ground_truth = example["reward_model"]["ground_truth"]

        item = {
            "data_source": example.get("data_source", "math_dapo"),
            "prompt": prompt,
            "ability": example.get("ability", "MATH"),
            "reward_model": {
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "index": i,
                "source": "dapo_math_17k",
            },
        }
        verl_data.append(item)

    output_dir = Path("attention_based_rewards/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(verl_data)
    output_path = output_dir / "dapo_math_17k.parquet"
    df.to_parquet(output_path)

    print(f"\nSaved {len(df)} examples to {output_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample prompt: {df.iloc[0]['prompt'][1]['content'][:100]}...")
    print(f"  Sample answer: {df.iloc[0]['reward_model']['ground_truth']}")

    # Verify by reading back
    df_check = pd.read_parquet(output_path)
    print(f"\nVerification: read back {len(df_check)} rows, columns={list(df_check.columns)}")
    assert len(df_check) == len(ds), "Row count mismatch!"
    print("Done!")


if __name__ == "__main__":
    main()
