"""Subset selection for s1K pruning.

Two strategies:
- random_select: uniform random without replacement, seeded
- skill_abundance_select: paper's rank-by-total-skill-count (§4, App. C)
"""

from __future__ import annotations

import random
from typing import Iterable


def random_select(pool_size: int, n: int, seed: int) -> list[int]:
    """Return n distinct indices in [0, pool_size) chosen uniformly, seeded.

    Order of returned indices is `random.sample`'s order (deterministic
    given the seed) and is the order used downstream.
    """
    if n > pool_size:
        raise ValueError(f"Cannot select {n} from pool of size {pool_size}")
    rng = random.Random(seed)
    return rng.sample(range(pool_size), n)
