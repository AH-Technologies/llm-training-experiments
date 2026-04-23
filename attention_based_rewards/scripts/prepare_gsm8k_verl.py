#!/usr/bin/env python3
"""Prepare GSM8K train split as verl-format parquet for GRPO training.

Produces:
  - attention_based_rewards/data/gsm8k_train.parquet  (7,473 examples)

Format matches verl's expected schema (same as existing pi13 data):
  - data_source: str
  - prompt: list[dict]  (chat messages)
  - reward_model: dict   (contains ground_truth)
  - extra_info: dict
"""

import re
import pandas as pd
from datasets import load_dataset
from pathlib import Path


# Thinking Sparks system prompt (from paper A.1)
SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide "
    "the user with the answer. Respond in the following format:\n"
    "<think>\\n...\\n</think>\\n<answer>\\n...\\n</answer>"
)


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the numeric answer from GSM8K's '#### <number>' format."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


def prepare_gsm8k_train(output_dir: Path):
    """Create GSM8K train parquet in verl format."""
    print("Loading GSM8K train split...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"  {len(ds)} examples")

    verl_data = []
    for i, example in enumerate(ds):
        question = example["question"]
        answer = extract_gsm8k_answer(example["answer"])

        item = {
            "data_source": "gsm8k",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "ability": "math",
            "reward_model": {
                "ground_truth": answer,
            },
            "extra_info": {
                "index": i,
                "source": "gsm8k_train",
            },
        }
        verl_data.append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(verl_data)
    output_path = output_dir / "gsm8k_train.parquet"
    df.to_parquet(output_path)

    print(f"Saved {len(df)} examples to {output_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample prompt: {df.iloc[0]['prompt'][1]['content'][:100]}...")
    print(f"  Sample answer: {df.iloc[0]['reward_model']['ground_truth']}")

    return output_path


def prepare_gsm8k_test(output_dir: Path):
    """Create GSM8K test parquet in verl format (for final evaluation)."""
    print("\nLoading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  {len(ds)} examples")

    verl_data = []
    for i, example in enumerate(ds):
        question = example["question"]
        answer = extract_gsm8k_answer(example["answer"])

        item = {
            "data_source": "gsm8k",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "ability": "math",
            "reward_model": {
                "ground_truth": answer,
            },
            "extra_info": {
                "index": i,
                "source": "gsm8k_test",
            },
        }
        verl_data.append(item)

    df = pd.DataFrame(verl_data)
    output_path = output_dir / "gsm8k_test.parquet"
    df.to_parquet(output_path)

    print(f"Saved {len(df)} examples to {output_path}")
    return output_path


if __name__ == "__main__":
    output_dir = Path("attention_based_rewards/data")
    prepare_gsm8k_train(output_dir)
    prepare_gsm8k_test(output_dir)
