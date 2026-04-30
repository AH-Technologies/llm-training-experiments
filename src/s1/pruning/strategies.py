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


_BASE_NLL_PATH = "data/s1K/s1k_base_nll.parquet"


def _load_base_nll(s1k_table: pa.Table) -> pa.Table:
    """Load s1k_base_nll.parquet and assert row alignment with s1K."""
    import pyarrow.parquet as pq
    try:
        nll = pq.read_table(_BASE_NLL_PATH)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{_BASE_NLL_PATH} not found. Run "
            "`python -m s1.pruning.compute_base_nll` to precompute base-model "
            "NLL signals before using base_loss / base_logprob_mean strategies."
        ) from e
    if nll.num_rows != s1k_table.num_rows:
        raise ValueError(
            f"{_BASE_NLL_PATH} has {nll.num_rows} rows but s1K has {s1k_table.num_rows}; "
            "regenerate the NLL parquet against the current s1K."
        )
    return nll


@register("base_loss")
def score_base_loss(s1k_table: pa.Table, skills_table: pa.Table | None) -> list[float]:
    """Total negative log-likelihood of the SFT response under the untrained
    base model. High score = response is far from what the base model would
    naturally generate; the model has the most to learn from this example.

    Top of the screening partition picks the most "informative" / hardest
    examples; bottom picks the ones the base model already roughly produces.
    Scores are length-confounded by design — long traces accumulate more
    total NLL — see `base_logprob_mean` for the per-token version.
    """
    nll = _load_base_nll(s1k_table)
    return [float(x) for x in nll.column("total_nll").to_pylist()]


@register("base_logprob_mean")
def score_base_logprob_mean(s1k_table: pa.Table, skills_table: pa.Table | None) -> list[float]:
    """Per-token mean log-probability of the SFT response under the untrained
    base model. Score = -mean_nll, so HIGH = each token is highly probable
    given context = the trace reads as smoothly extending from the base
    model's natural distribution.

    Top of the screening partition picks the most "coherent" / sustained-
    confidence reasoning traces; bottom picks the most surprising-token-by-
    token traces (unusual notation, stylistic outliers, dense unfamiliar
    formalism). Length-normalised by per-token averaging — distinct from
    `base_loss` even though both come from the same forward pass.
    """
    nll = _load_base_nll(s1k_table)
    return [-float(x) for x in nll.column("mean_nll").to_pylist()]
