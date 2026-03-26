#!/usr/bin/env python3
"""Prepare AIME 2025 and AMC 2023 evaluation datasets.

Downloads from HuggingFace and saves as verl-format parquet.
Sources:
  - MathArena/aime_2025 (30 problems, AIME I+II)
  - AI-MO/aimo-validation-amc (83 problems, AMC12 2022-2023)
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path


SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


def prepare_aime_2025(output_dir: Path):
    """Prepare AIME 2025 (30 competition math problems)."""
    print("Loading MathArena/aime_2025...")
    ds = load_dataset("MathArena/aime_2025", split="train")
    print(f"  {len(ds)} problems")

    verl_data = []
    for i, ex in enumerate(ds):
        item = {
            "data_source": "aime_2025",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["problem"]},
            ],
            "ability": "math",
            "reward_model": {
                "ground_truth": str(int(ex["answer"])),
            },
            "extra_info": {
                "index": i,
                "problem_idx": ex["problem_idx"],
                "problem_type": ex.get("problem_type", []),
                "source": "aime_2025",
            },
        }
        verl_data.append(item)

    df = pd.DataFrame(verl_data)
    out_path = output_dir / "aime_2025.parquet"
    df.to_parquet(out_path)
    print(f"  Saved {len(df)} problems to {out_path}")
    print(f"  Sample: {df.iloc[0]['prompt'][1]['content'][:100]}...")
    print(f"  Answer: {df.iloc[0]['reward_model']['ground_truth']}")
    return out_path


def prepare_amc23(output_dir: Path):
    """Prepare AMC 2022-2023 (83 AMC12 problems)."""
    print("\nLoading AI-MO/aimo-validation-amc...")
    ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
    print(f"  {len(ds)} problems")

    verl_data = []
    for i, ex in enumerate(ds):
        item = {
            "data_source": "amc_2023",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["problem"]},
            ],
            "ability": "math",
            "reward_model": {
                "ground_truth": str(int(ex["answer"])),
            },
            "extra_info": {
                "index": i,
                "url": ex.get("url", ""),
                "source": "amc_2023",
            },
        }
        verl_data.append(item)

    df = pd.DataFrame(verl_data)
    out_path = output_dir / "amc_2023.parquet"
    df.to_parquet(out_path)
    print(f"  Saved {len(df)} problems to {out_path}")
    print(f"  Sample: {df.iloc[0]['prompt'][1]['content'][:100]}...")
    print(f"  Answer: {df.iloc[0]['reward_model']['ground_truth']}")
    return out_path


if __name__ == "__main__":
    output_dir = Path("attention_based_rewards/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_aime_2025(output_dir)
    prepare_amc23(output_dir)
