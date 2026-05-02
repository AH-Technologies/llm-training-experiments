"""Greedy facility-location selection over s1K skill tags.

The `skill_count` strategy ranks rows by how many salient skills they cover.
That picks rows individually, ignoring how the chosen subset *as a whole*
covers the skill space. `skill_coverage` ranks rows by their marginal
contribution to a facility-location objective:

    f(S) = Σ_{i ∈ pool} max_{j ∈ S} jaccard(skills_i, skills_j)

— "every example in the pool has a similar friend in the chosen set." This
is the canonical diversity / coverage objective in the data-selection
literature (DEITA, QDIT, MIG, SMART). Skill sets per row are the union of
the five paper-native category lists in `s1k_skills.parquet`, with the
category as a prefix so e.g. `algebra:linear_equations` doesn't collide
with `geometry:linear_equations`.

Output of `compute_ranking` is the full lazy-greedy selection order of the
1000 rows. The strategy in `strategies.py` reads this ranking and emits
scores `-selection_order`, so top-third = first 333 picked = best for
skill coverage.

The ranking is deterministic and fast (a few seconds on 1000 rows). Cached
to `data/s1K/s1k_skill_coverage_rank.parquet` so repeat runs are instant.
"""

from __future__ import annotations

import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from s1.pruning.tag_skills import SKILL_CATEGORIES


_DEFAULT_RANK_PATH = "data/s1K/s1k_skill_coverage_rank.parquet"


def _gather_skill_sets(skills_table: pa.Table) -> list[set[str]]:
    """Return list[set[str]] of category-prefixed skills per row."""
    n = skills_table.num_rows
    sets: list[set[str]] = [set() for _ in range(n)]
    for cat in SKILL_CATEGORIES:
        col = skills_table.column(f"skills_{cat}").to_pylist()
        for i, lst in enumerate(col):
            for s in lst or ():
                sets[i].add(f"{cat}:{s}")
    return sets


def _jaccard_matrix(sets: list[set[str]]) -> np.ndarray:
    """Symmetric N×N Jaccard similarity matrix. 1.0 on the diagonal.

    Two empty sets default to similarity 0 — they share no skills, so they
    should not be treated as redundant. This matters for s1K rows where the
    judge returned `None` across all five categories: we want the greedy
    selector to still distribute over them by other margins, not collapse
    them onto one another.
    """
    n = len(sets)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim[i, i] = 1.0
        ai = sets[i]
        for j in range(i + 1, n):
            aj = sets[j]
            if not ai or not aj:
                s = 0.0
            else:
                inter = len(ai & aj)
                if inter == 0:
                    s = 0.0
                else:
                    s = inter / len(ai | aj)
            sim[i, j] = s
            sim[j, i] = s
    return sim


def _greedy_facility_location(sim: np.ndarray, k: int) -> list[int]:
    """Greedy maximisation of f(S) = Σ_i max_{j ∈ S} sim[i, j], |S| = k.

    Each iteration picks the row with the largest marginal gain over the
    currently-best similarity vector. Ties are broken by lowest index
    (numpy `argmax` behaviour) for determinism. O(n²) per iteration with
    numpy broadcasting; on n = 1000 this is a few seconds total.
    """
    n = sim.shape[0]
    best = np.zeros(n, dtype=np.float32)
    chosen = np.zeros(n, dtype=bool)
    order: list[int] = []
    for _ in range(min(k, n)):
        gains = np.maximum(sim - best[:, None], 0).sum(axis=0)
        gains[chosen] = -np.inf
        c = int(gains.argmax())
        order.append(c)
        chosen[c] = True
        best = np.maximum(best, sim[:, c])
    return order


def compute_ranking(skills_table: pa.Table) -> list[int]:
    """Full greedy facility-location ranking of s1K rows by skill coverage."""
    sets = _gather_skill_sets(skills_table)
    sim = _jaccard_matrix(sets)
    return _greedy_facility_location(sim, k=skills_table.num_rows)


def load_or_compute_ranking(
    skills_table: pa.Table,
    cache_path: str | None = None,
) -> list[int]:
    """Return the cached ranking, recomputing+writing it if the cache is
    missing or row-misaligned with `skills_table`. Falls back to module-level
    `_DEFAULT_RANK_PATH` when `cache_path` is None — resolved at call time so
    tests can monkeypatch the constant.
    """
    if cache_path is None:
        cache_path = _DEFAULT_RANK_PATH
    if os.path.exists(cache_path):
        cached = pq.read_table(cache_path)
        if cached.num_rows == skills_table.num_rows:
            return [int(x) for x in cached.column("selection_order").to_pylist()]
    rank = compute_ranking(skills_table)
    out = pa.table({"selection_order": rank})
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    pq.write_table(out, cache_path)
    return rank
