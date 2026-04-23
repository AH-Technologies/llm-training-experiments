"""Subset selection for s1K pruning.

Two strategies:
- random_select: uniform random without replacement, seeded
- skill_abundance_select: paper's rank-by-total-skill-count (§4, App. C)
"""

from __future__ import annotations

import argparse
import random
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


def random_select(pool_size: int, n: int, seed: int) -> list[int]:
    """Return n distinct indices in [0, pool_size) chosen uniformly, seeded.

    Order of returned indices is `random.sample`'s order (deterministic
    given the seed) and is the order used downstream.
    """
    if n > pool_size:
        raise ValueError(f"Cannot select {n} from pool of size {pool_size}")
    rng = random.Random(seed)
    return rng.sample(range(pool_size), n)


def skill_abundance_select(skills_path: str, n: int) -> list[int]:
    """Paper's selection rule: rank by total skill count DESC, break ties by
    index ASC, take top n. Reads the skills parquet produced by
    `s1.pruning.tag_skills.tag_dataset`.
    """
    table = pq.read_table(skills_path, columns=["index", "skill_count"])
    indices = table.column("index").to_pylist()
    counts = table.column("skill_count").to_pylist()
    pool_size = len(indices)
    if n > pool_size:
        raise ValueError(f"Cannot select {n} from pool of size {pool_size}")
    # Sort by (skill_count DESC, index ASC). Stable Python sort lets us do it
    # in two passes OR one key-function pass.
    ranked = sorted(
        zip(counts, indices), key=lambda pair: (-pair[0], pair[1])
    )
    return [idx for _, idx in ranked[:n]]


def write_subset(input_path: str, output_path: str, indices: list[int]) -> None:
    """Write a parquet containing only the given row indices from input_path,
    preserving the order of `indices`.
    """
    table = pq.read_table(input_path)
    # pa.Table.take preserves order; accepts any iterable of int indices
    subset = table.take(pa.array(indices, type=pa.int64()))
    pq.write_table(subset, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a pruned subset of s1K for SFT."
    )
    parser.add_argument(
        "--strategy",
        choices=("random", "skill_abundance"),
        required=True,
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", default="data/s1K/s1k.parquet")
    parser.add_argument(
        "--skills",
        default="data/s1K/s1k_skills.parquet",
        help="Required for --strategy skill_abundance.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.strategy == "random":
        table = pq.read_table(args.input)
        indices = random_select(pool_size=table.num_rows, n=args.n, seed=args.seed)
    else:  # skill_abundance
        indices = skill_abundance_select(args.skills, n=args.n)

    write_subset(args.input, args.output, indices)
    print(f"Wrote {len(indices)} rows ({args.strategy}, n={args.n}) to {args.output}")


if __name__ == "__main__":
    main()
