#!/usr/bin/env python3
"""Prepare DAPO-Math-17k data for attention-rhythm GRPO training.

Creates a symlink to the existing parquet from attention_based_rewards,
or downloads and prepares it fresh if not available.
"""

import os
import sys

PROJECT_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments"
SRC = os.path.join(PROJECT_DIR, "attention_based_rewards/data/dapo_math_17k.parquet")
DST = os.path.join(PROJECT_DIR, "attention_sparks_thinking/data/dapo_math_17k.parquet")


def main():
    os.makedirs(os.path.dirname(DST), exist_ok=True)

    if os.path.exists(SRC):
        if not os.path.exists(DST):
            os.symlink(SRC, DST)
            print(f"Symlinked {DST} -> {SRC}")
        else:
            print(f"Data already exists at {DST}")
        return

    # Prepare fresh if source doesn't exist
    print(f"Source not found at {SRC}, preparing fresh...")

    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("BytedTsinghua/DAPO-Math-17k", split="train")

    SYSTEM_PROMPT = (
        "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
        "You first think about the reasoning process as an internal monologue and then provide "
        "the user with the answer. Respond in the following format:\n"
        "<think>\n...\n</think>\n<answer>\n...\n</answer>"
    )

    rows = []
    for i, item in enumerate(ds):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ]
        rows.append({
            "data_source": "math_dapo",
            "prompt": prompt,
            "ability": "MATH",
            "reward_model": {"ground_truth": item["answer"]},
            "extra_info": {"index": i, "source": "dapo_math_17k"},
        })

    df = pd.DataFrame(rows)
    df.to_parquet(DST)
    print(f"Saved {len(df)} examples to {DST}")


if __name__ == "__main__":
    main()
