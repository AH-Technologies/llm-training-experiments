"""Download DAPO-Math-17K and prepare train/test splits for self-teach GRPO training.

DAPO-Math-17K contains ~17.9K competition-level math problems from AoPS with
integer-only answers. The HuggingFace version has each problem duplicated 100x
(for rollout training), so we deduplicate first.

The original prompts use "Answer: $Answer" format; we replace with our standard
system prompt and \\boxed{} format.

Usage:
    python scripts/prepare_dapo_self_teach.py
    python scripts/prepare_dapo_self_teach.py --max-problems 5000
    python scripts/prepare_dapo_self_teach.py --test-fraction 0.1
"""

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the problem and provide your final answer. "
    "Put your final answer within \\boxed{}."
)

# The original DAPO prompt prefix that we strip to get the raw math problem
DAPO_PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n"
)


def extract_question(prompt_messages: list[dict]) -> str:
    """Extract the raw math problem from DAPO's prompt format."""
    # DAPO uses a single user message with a prefix + the actual problem
    user_content = prompt_messages[0]["content"]
    if user_content.startswith(DAPO_PROMPT_PREFIX):
        return user_content[len(DAPO_PROMPT_PREFIX):]
    # Fallback: try stripping up to the double newline after instructions
    parts = user_content.split("\n\n", 1)
    if len(parts) == 2:
        return parts[1]
    return user_content


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DAPO-Math-17K for self-teach training"
    )
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="Cap total number of problems (sample randomly after shuffling)",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.1,
        help="Fraction of data to use for test split",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/dapo",
        help="Output directory",
    )
    args = parser.parse_args()

    # Download dataset
    print("Loading DAPO-Math-17K from HuggingFace...")
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    print(f"Total rows (with duplicates): {len(ds)}")

    # Deduplicate by extra_info.index
    seen = set()
    unique_rows = []
    for i in range(len(ds)):
        row = ds[i]
        uid = row["extra_info"]["index"]
        if uid not in seen:
            seen.add(uid)
            unique_rows.append(row)
    print(f"Unique problems after dedup: {len(unique_rows)}")

    # Extract questions and answers
    problems = []
    for row in unique_rows:
        question = extract_question(row["prompt"])
        answer = row["reward_model"]["ground_truth"]
        problems.append({
            "question": question,
            "answer": answer,
            "ability": row.get("ability", "MATH"),
            "original_id": row["extra_info"]["index"],
        })

    print(f"\nSample extracted question (first 200 chars):")
    print(f"  {problems[0]['question'][:200]}...")
    print(f"  Answer: {problems[0]['answer']}")

    # Shuffle
    rng = random.Random(args.seed)
    rng.shuffle(problems)

    # Cap if requested
    if args.max_problems and len(problems) > args.max_problems:
        problems = problems[:args.max_problems]
        print(f"Capped to {args.max_problems} problems")

    # Train/test split
    n_test = max(1, int(len(problems) * args.test_fraction))
    n_train = len(problems) - n_test
    splits = {
        "train": problems[:n_train],
        "test": problems[n_train:],
    }
    print(f"\nSplit: {n_train} train / {n_test} test")

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_problems in splits.items():
        # Raw format (for eval scripts)
        raw_table = pa.Table.from_pydict({
            "question": [p["question"] for p in split_problems],
            "solution": [p["answer"] for p in split_problems],
            "source": [p["ability"] for p in split_problems],
        })

        # VERL format (for training)
        verl_data = {
            "data_source": [],
            "prompt": [],
            "ability": [],
            "reward_model": [],
            "extra_info": [],
        }
        for i, p in enumerate(split_problems):
            verl_data["data_source"].append("dapo")
            verl_data["prompt"].append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p["question"]},
            ])
            verl_data["ability"].append(p["ability"])
            verl_data["reward_model"].append({
                "ground_truth": p["answer"],
            })
            verl_data["extra_info"].append({
                "index": i,
                "source": p["ability"],
                "original_id": p["original_id"],
                "interaction_kwargs": {
                    "name": "self_teach",
                    "ground_truth": p["answer"],
                    "data_source": "dapo",
                },
            })

        verl_table = pa.Table.from_pydict(verl_data)

        pq.write_table(raw_table, out_dir / f"dapo_{split_name}.parquet")
        pq.write_table(verl_table, out_dir / f"dapo_{split_name}_verl.parquet")

    print(f"\nSaved to {out_dir}/:")
    print(f"  dapo_train.parquet         ({n_train} rows, raw format)")
    print(f"  dapo_test.parquet          ({n_test} rows, raw format)")
    print(f"  dapo_train_verl.parquet    ({n_train} rows, VERL format)")
    print(f"  dapo_test_verl.parquet     ({n_test} rows, VERL format)")


if __name__ == "__main__":
    main()
