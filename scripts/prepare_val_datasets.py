"""Download and prepare validation datasets in VERL format.

Prepares three validation sets:
  - AIME 2025 (MathArena/aime_2025) — 30 problems
  - AMC (AI-MO/aimo-validation-amc) — 83 problems
  - MATH500 (already exists as data/math500.parquet)

Outputs individual parquets and a combined validation parquet with data_source
tags so metrics can be tracked per-dataset in W&B.

Usage:
    python scripts/prepare_val_datasets.py
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the problem and provide your final answer. "
    "Put your final answer within \\boxed{}."
)


def load_aime_2025():
    """Load AIME 2025 dataset."""
    ds = load_dataset("MathArena/aime_2025", split="train")
    rows = []
    for r in ds:
        rows.append({
            "question": r["problem"],
            "answer": str(int(r["answer"])),
            "data_source": "aime_2025",
        })
    print(f"AIME 2025: {len(rows)} problems")
    return rows


def load_amc():
    """Load AMC validation dataset."""
    ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
    rows = []
    for r in ds:
        ans = r["answer"]
        if ans == int(ans):
            ans = str(int(ans))
        else:
            ans = str(ans)
        rows.append({
            "question": r["problem"],
            "answer": ans,
            "data_source": "amc",
        })
    print(f"AMC: {len(rows)} problems")
    return rows


def load_math500(path="data/math500.parquet"):
    """Load existing MATH500 parquet."""
    table = pq.read_table(path)
    cols = table.column_names
    rows = []

    for i in range(table.num_rows):
        if "prompt" in cols and "reward_model" in cols:
            # Already in VERL format
            prompt = table.column("prompt")[i].as_py()
            question = ""
            for msg in prompt:
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                    break
            rm = table.column("reward_model")[i].as_py()
            answer = rm.get("ground_truth", "") if isinstance(rm, dict) else ""
        elif "question" in cols:
            question = table.column("question")[i].as_py()
            answer = table.column("solution")[i].as_py() if "solution" in cols else ""
        else:
            raise ValueError(f"Unknown MATH500 format, columns: {cols}")

        rows.append({
            "question": question,
            "answer": str(answer),
            "data_source": "math500",
        })
    print(f"MATH500: {len(rows)} problems")
    return rows


def to_verl_table(rows: list[dict]) -> pa.Table:
    """Convert rows to VERL parquet format."""
    verl = {
        "data_source": [],
        "prompt": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
    }
    for i, r in enumerate(rows):
        verl["data_source"].append(r["data_source"])
        verl["prompt"].append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["question"]},
        ])
        verl["ability"].append("math")
        verl["reward_model"].append({"ground_truth": r["answer"]})
        verl["extra_info"].append({"index": i})

    return pa.Table.from_pydict(verl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--math500-path", default="data/math500.parquet")
    parser.add_argument("--out-dir", default="data/val")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all datasets
    aime = load_aime_2025()
    amc = load_amc()
    math500 = load_math500(args.math500_path)

    # Save individual datasets
    for name, rows in [("aime_2025", aime), ("amc", amc), ("math500", math500)]:
        table = to_verl_table(rows)
        path = out_dir / f"{name}_verl.parquet"
        pq.write_table(table, path)
        print(f"  Saved {path} ({len(rows)} rows)")

    # Save combined validation set
    all_rows = aime + amc + math500
    combined = to_verl_table(all_rows)
    combined_path = out_dir / "val_combined_verl.parquet"
    pq.write_table(combined, combined_path)
    print(f"\nSaved combined: {combined_path} ({len(all_rows)} rows)")

    # Summary
    from collections import Counter
    sources = Counter(r["data_source"] for r in all_rows)
    print("\nCombined validation set:")
    for src, count in sources.most_common():
        print(f"  {src:15s}: {count}")


if __name__ == "__main__":
    main()
