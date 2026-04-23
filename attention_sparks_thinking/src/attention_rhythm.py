"""Core attention rhythm metrics: WAAD, FAI, and gamma computation.

Pure PyTorch, no veRL dependency. Implements the coupled rhythm credit
assignment from "Thinking Sparks!" (Park et al., 2025).
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def compute_waad(
    A_bar_loc: torch.Tensor,
    response_start: int,
    W: int = 10,
) -> torch.Tensor:
    """Compute Windowed Average Attention Distance for response tokens.

    Args:
        A_bar_loc: (seq_len, seq_len) aggregated local attention matrix
        response_start: index where response tokens begin
        W: clipping window (default 10)

    Returns:
        (num_response_tokens,) tensor of WAAD values
    """
    seq_len = A_bar_loc.shape[0]
    positions = torch.arange(seq_len, device=A_bar_loc.device)
    # dist[t, s] = t - s (how far back position t looks at position s)
    dist = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
    dist_clipped = dist.clamp(min=0, max=W).float()

    # WAAD[t] = sum_s A_bar_loc[t,s] * min(t-s, W)
    waad = (A_bar_loc * dist_clipped).sum(dim=-1)  # (seq_len,)
    return waad[response_start:]


def compute_fai(
    A_bar_glob: torch.Tensor,
    response_start: int,
    H_lo: int = 10,
    H_hi: int = 50,
) -> torch.Tensor:
    """Compute Future Attention Influence for response tokens.

    Args:
        A_bar_glob: (seq_len, seq_len) aggregated global attention matrix
        response_start: index where response tokens begin
        H_lo, H_hi: horizon bounds for influence calculation

    Returns:
        (num_response_tokens,) tensor of FAI values
    """
    seq_len = A_bar_glob.shape[0]
    num_response = seq_len - response_start
    fai = torch.zeros(num_response, device=A_bar_glob.device)

    for idx, s in enumerate(range(response_start, seq_len)):
        t_lo = max(s + H_lo, response_start)
        t_hi = min(s + H_hi, seq_len - 1)
        if t_lo > t_hi:
            fai[idx] = 0.0
            continue
        fai[idx] = A_bar_glob[t_lo : t_hi + 1, s].mean()

    return fai


def compute_fai_vectorized(
    A_bar_glob: torch.Tensor,
    response_start: int,
    H_lo: int = 10,
    H_hi: int = 50,
) -> torch.Tensor:
    """Vectorized FAI computation (faster for long sequences)."""
    seq_len = A_bar_glob.shape[0]
    num_response = seq_len - response_start

    # For each source position s, build a mask of valid target positions t
    s_positions = torch.arange(response_start, seq_len, device=A_bar_glob.device)
    t_positions = torch.arange(seq_len, device=A_bar_glob.device)

    # mask[idx, t] = True if t is a valid target for source s_positions[idx]
    # Valid: t >= response_start, t >= s + H_lo, t <= min(seq_len-1, s + H_hi)
    s_exp = s_positions.unsqueeze(1)  # (num_response, 1)
    t_exp = t_positions.unsqueeze(0)  # (1, seq_len)

    mask = (t_exp >= s_exp + H_lo) & (t_exp <= torch.clamp(s_exp + H_hi, max=seq_len - 1))
    mask = mask & (t_exp >= response_start)

    # Gather attention values: A_bar_glob[t, s] for valid (t, s)
    # We need column s from rows t
    attn_cols = A_bar_glob[:, response_start:seq_len]  # (seq_len, num_response)
    attn_cols = attn_cols.T  # (num_response, seq_len) — row idx is source token offset

    masked_attn = attn_cols * mask.float()
    counts = mask.float().sum(dim=1).clamp(min=1)
    fai = masked_attn.sum(dim=1) / counts

    return fai


def compute_gamma(
    waad: torch.Tensor,
    fai: torch.Tensor,
    q: float = 0.4,
    gamma_amp: float = 1.5,
    alpha: float = 0.5,
    k: int = 3,
) -> tuple[torch.Tensor, dict]:
    """Compute per-token scaling coefficients using coupled rhythm credit.

    Args:
        waad: (num_response_tokens,) WAAD values
        fai: (num_response_tokens,) FAI values
        q: quantile for top selection (default 0.4 = top 40%)
        gamma_amp: amplification factor (default 1.5)
        alpha: back-allocation fraction (default 0.5)
        k: neighborhood lookback for dominated anchor check (default 3)

    Returns:
        gamma: (num_response_tokens,) scaling coefficients
        stats: dict with diagnostic info
    """
    n = waad.shape[0]
    assert n == fai.shape[0], f"WAAD length {n} != FAI length {fai.shape[0]}"

    if n < 3:
        return torch.ones(n, device=waad.device), {"n_tokens": n, "n_amplified": 0}

    # Step 5a: delta[t] = |WAAD[t] - WAAD[t+1]|
    delta = torch.zeros(n, device=waad.device)
    delta[:-1] = (waad[:-1] - waad[1:]).abs()

    # T_loc = top q fraction by delta
    delta_threshold = torch.quantile(delta, 1.0 - q)
    T_loc = delta >= delta_threshold

    # Step 5b: T_glob = top q fraction by FAI
    fai_threshold = torch.quantile(fai, 1.0 - q)
    T_glob = fai >= fai_threshold

    # Step 5c: Identify locally-dominated anchors
    tau_waad = torch.median(waad)
    tau_delta = torch.median(delta)

    D = torch.zeros(n, dtype=torch.bool, device=waad.device)
    intro_tokens = torch.zeros(n, dtype=torch.bool, device=waad.device)

    for t in range(n):
        if not T_glob[t]:
            continue
        if waad[t] > tau_waad:
            continue
        # Check if any of the k preceding tokens has a high delta
        start = max(0, t - k)
        if start == t:
            continue
        neighborhood_delta = delta[start:t]
        if neighborhood_delta.max() >= tau_delta:
            D[t] = True
            # intro(t) = argmax of delta in {t-k, ..., t-1}
            intro_idx = start + neighborhood_delta.argmax()
            intro_tokens[intro_idx] = True

    # Step 5d: Compute gamma
    gamma = torch.ones(n, device=waad.device)
    boost = gamma_amp - 1.0

    # Regular anchors: in T_glob but not in D
    regular_anchor = T_glob & ~D
    gamma[regular_anchor] = 1.0 + boost

    # Dominated anchors
    gamma[D] = 1.0 + (1.0 - alpha) * boost

    # Intro tokens (preplan tokens that get back-allocated credit)
    gamma[intro_tokens] = 1.0 + alpha * boost

    # Stats
    n_amplified = (gamma > 1.0).sum().item()
    frac_amplified = n_amplified / n if n > 0 else 0.0

    stats = {
        "n_tokens": n,
        "n_amplified": n_amplified,
        "frac_amplified": frac_amplified,
        "gamma_mean": gamma.mean().item(),
        "gamma_std": gamma.std().item(),
        "n_regular_anchor": regular_anchor.sum().item(),
        "n_dominated_anchor": D.sum().item(),
        "n_intro": intro_tokens.sum().item(),
        "n_T_loc": T_loc.sum().item(),
        "n_T_glob": T_glob.sum().item(),
    }

    if frac_amplified > 0.6:
        logger.warning(f"High gamma coverage: {frac_amplified:.1%} tokens amplified (>60%). Thresholds may be too loose.")
    elif frac_amplified < 0.1:
        logger.warning(f"Low gamma coverage: {frac_amplified:.1%} tokens amplified (<10%). Thresholds may be too tight.")

    return gamma, stats
