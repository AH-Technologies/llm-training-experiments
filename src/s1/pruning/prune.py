"""Subset selection for s1K pruning.

Selection modes:
- random_select: uniform random without replacement, seeded
- skill_abundance_select: paper's rank-by-total-skill-count (Alibaba §4, App. C)
- screen_select: thirds-based screening — score by a registered strategy,
  sort, split into top/middle/bottom, return one third
"""

from __future__ import annotations

import argparse
import random

import pyarrow as pa
import pyarrow.parquet as pq

from s1.pruning import strategies


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


_POSITIONS = ("top", "middle", "bottom")


def partition_thirds(scores: list[float]) -> dict[str, list[int]]:
    """Sort indices by score DESC (ties broken by index ASC), split into thirds.

    Returns {"top": [...], "middle": [...], "bottom": [...]}. Each list has
    floor(len(scores)/3) entries. Any leftover (1 or 2 of the lowest-scored
    indices) is dropped — the three positions are always equal-sized.
    """
    n = len(scores)
    third = n // 3
    ranked = sorted(range(n), key=lambda i: (-scores[i], i))
    return {
        "top": ranked[:third],
        "middle": ranked[third : 2 * third],
        "bottom": ranked[2 * third : 3 * third],
    }


def screen_select(
    strategy: str,
    position: str,
    n: int,
    input_path: str,
    skills_path: str | None,
) -> list[int]:
    """Score every row by `strategy`, split into thirds, return the requested
    position's first n indices.

    The full third has floor(pool_size/3) entries; n must not exceed that.
    """
    if position not in _POSITIONS:
        raise ValueError(f"position must be one of {_POSITIONS}, got {position!r}")
    s1k_table = pq.read_table(input_path)
    skills_table = pq.read_table(skills_path) if skills_path else None
    score_fn = strategies.get(strategy)
    scores = score_fn(s1k_table, skills_table)
    if len(scores) != s1k_table.num_rows:
        raise ValueError(
            f"strategy '{strategy}' returned {len(scores)} scores for {s1k_table.num_rows} rows"
        )
    third = partition_thirds(scores)[position]
    if n > len(third):
        raise ValueError(
            f"Cannot select {n} from a third of size {len(third)} (pool={len(scores)})"
        )
    return third[:n]


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
        required=True,
        help=(
            "'random', 'skill_abundance', or any registered ranking strategy "
            f"(see s1.pruning.strategies; current: {strategies.names()}). "
            "When the strategy is a ranking one, --position selects the third."
        ),
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", default="data/s1K/s1k.parquet")
    parser.add_argument(
        "--skills",
        default="data/s1K/s1k_skills.parquet",
        help="Required for --strategy skill_abundance and for skill-based ranking strategies.",
    )
    parser.add_argument(
        "--position",
        choices=_POSITIONS,
        default=None,
        help="For ranking strategies: which third to return (top/middle/bottom).",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.strategy == "random":
        table = pq.read_table(args.input)
        indices = random_select(pool_size=table.num_rows, n=args.n, seed=args.seed)
    elif args.strategy == "skill_abundance":
        indices = skill_abundance_select(args.skills, n=args.n)
    else:
        if args.position is None:
            raise SystemExit(
                f"--strategy '{args.strategy}' is a ranking strategy; "
                "pass --position {top,middle,bottom}."
            )
        indices = screen_select(
            strategy=args.strategy,
            position=args.position,
            n=args.n,
            input_path=args.input,
            skills_path=args.skills,
        )

    write_subset(args.input, args.output, indices)
    tag = f"{args.strategy}/{args.position}" if args.position else args.strategy
    print(f"Wrote {len(indices)} rows ({tag}, n={args.n}) to {args.output}")


if __name__ == "__main__":
    main()
