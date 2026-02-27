"""Add interaction_kwargs to training data for the self-teach multi-turn pipeline.

Takes an existing parquet file and adds the 'interaction_kwargs' column
needed by VERL's multi-turn interaction system.

Usage:
    python scripts/prepare_self_teach_data.py \
        --input data/pi13_r128.parquet \
        --output data/pi13_r128_self_teach.parquet
"""

import argparse

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Prepare data for self-teach training")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    args = parser.parse_args()

    table = pq.read_table(args.input)
    rows = table.to_pydict()
    n = len(rows["data_source"])

    # Embed interaction_kwargs INSIDE extra_info, because VERL's rl_dataset.py
    # reads it from extra_info["interaction_kwargs"] (not as a top-level column).
    # See: verl/utils/dataset/rl_dataset.py:350 and
    #      verl/experimental/agent_loop/tool_agent_loop.py:150
    for i in range(n):
        ground_truth = rows["reward_model"][i].get("ground_truth", "")
        data_source = rows["data_source"][i]
        if rows["extra_info"][i] is None:
            rows["extra_info"][i] = {}
        rows["extra_info"][i]["interaction_kwargs"] = {
            "name": "self_teach",
            "ground_truth": ground_truth,
            "data_source": data_source,
        }

    out_table = pa.Table.from_pydict(rows)
    pq.write_table(out_table, args.output)
    print(f"Wrote {n} rows to {args.output}")
    print(f"  Embedded interaction_kwargs inside extra_info with name='self_teach'")


if __name__ == "__main__":
    main()
