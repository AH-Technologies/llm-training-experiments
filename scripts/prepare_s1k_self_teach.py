"""Convert s1K train/test parquet to verl format for self-teach training.

Converts from s1K columns (question, solution, source_type, ...) to verl
format (data_source, prompt, ability, reward_model, extra_info) with
interaction_kwargs embedded for the self-teach pipeline.

Usage:
    python scripts/prepare_s1k_self_teach.py
    python scripts/prepare_s1k_self_teach.py --train data/s1K/s1k_train.parquet --test data/s1K/s1k_test.parquet
"""

import argparse
import json

import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the problem and provide your final answer. "
    "Put your final answer within \\boxed{}."
)


def convert_table(table: pa.Table) -> pa.Table:
    """Convert s1K table to verl-compatible format."""
    d = table.to_pydict()
    n = table.num_rows

    rows = {
        "data_source": [],
        "prompt": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
    }

    for i in range(n):
        question = d["question"][i]
        solution = d["solution"][i]
        source_type = d["source_type"][i] or "unknown"
        cot_type = d["cot_type"][i] or "unknown"
        metadata = d["metadata"][i] or "{}"

        rows["data_source"].append("s1k")
        rows["prompt"].append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ])
        rows["ability"].append(cot_type)
        rows["reward_model"].append({
            "ground_truth": solution,
            "task_type": cot_type,
        })
        rows["extra_info"].append({
            "index": i,
            "source": source_type,
            "metadata": metadata,
            "interaction_kwargs": {
                "name": "self_teach",
                "ground_truth": solution,
                "data_source": "s1k",
            },
        })

    return pa.Table.from_pydict(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/s1K/s1k_train.parquet")
    parser.add_argument("--test", default="data/s1K/s1k_test.parquet")
    parser.add_argument("--out-dir", default="data/s1K")
    args = parser.parse_args()

    for split, path in [("train", args.train), ("test", args.test)]:
        table = pq.read_table(path)
        converted = convert_table(table)
        out_path = f"{args.out_dir}/s1k_{split}_verl.parquet"
        pq.write_table(converted, out_path)
        print(f"{split}: {table.num_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
