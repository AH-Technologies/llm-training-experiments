"""Subset selection for s1K pruning.

Two strategies:
- random_select: uniform random without replacement, seeded
- skill_abundance_select: paper's rank-by-total-skill-count (§4, App. C)
"""

from __future__ import annotations

import random
from typing import Iterable

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
