"""Per-sample scoring strategies for thirds-based screening.

Each strategy takes the raw s1K parquet (and optionally the skills parquet)
and returns one float per row. The screening sweep sorts by this score
descending and splits the dataset into top/middle/bottom thirds.

Adding a new strategy:
1. Define a function `score_<name>(s1k_table, skills_table) -> list[float]`.
2. Decorate it with `@register("<name>")`.
3. Add `("<name>", "top")`, `("<name>", "middle")`, `("<name>", "bottom")` to
   the SCREEN_CELLS array in scripts/submit_prune_screen.slurm.

The scoring should be cheap. If a strategy needs precomputation (e.g. base-model
NLL or sentence embeddings), build a separate parquet first and load it here —
do not run expensive jobs from inside score_*.
"""

from __future__ import annotations

from typing import Callable

import pyarrow as pa


ScoreFn = Callable[[pa.Table, pa.Table | None], list[float]]
_REGISTRY: dict[str, ScoreFn] = {}


def register(name: str) -> Callable[[ScoreFn], ScoreFn]:
    def deco(fn: ScoreFn) -> ScoreFn:
        if name in _REGISTRY:
            raise ValueError(f"Strategy '{name}' already registered")
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> ScoreFn:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown strategy '{name}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


# ── Built-in strategies ───────────────────────────────────────────────────

@register("response_length")
def score_response_length(s1k_table: pa.Table, skills_table: pa.Table | None) -> list[float]:
    """Character length of the first thinking trajectory.

    Cheap proxy for response complexity / reasoning-trace difficulty.
    """
    trajectories = s1k_table.column("thinking_trajectories").to_pylist()
    return [float(len(t[0])) if t else 0.0 for t in trajectories]


@register("skill_count")
def score_skill_count(s1k_table: pa.Table, skills_table: pa.Table | None) -> list[float]:
    """Total salient math skill count across the 5 paper categories.

    Requires the skills parquet (data/s1K/s1k_skills.parquet) produced by
    s1.pruning.tag_skills. This is the same signal `skill_abundance_select`
    uses, exposed here for the thirds-split screening regime.
    """
    if skills_table is None:
        raise ValueError(
            "score_skill_count requires --skills <s1k_skills.parquet>; pass it via the prune CLI."
        )
    return [float(c) for c in skills_table.column("skill_count").to_pylist()]
