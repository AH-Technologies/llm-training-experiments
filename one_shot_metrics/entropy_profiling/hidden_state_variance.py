#!/usr/bin/env python3
"""Compute hidden state variance metrics across rollouts.

Measures internal computational diversity: do rollouts take different paths
through the model's layers, or converge on similar internal representations?

Key hypothesis (from the README):
- High variance in early layers → model can't consistently parse the problem (too hard)
- High variance in late layers → model understands but is uncertain about computation (sweet spot)
"""

import numpy as np


def compute_hidden_state_metrics(all_hidden_states: np.ndarray) -> dict:
    """Compute variance and cosine distance metrics from hidden states.

    Args:
        all_hidden_states: Array of shape (n_rollouts, n_layers, hidden_dim).
            Final-token hidden states per layer per rollout.

    Returns:
        Dict with hidden state variance metrics.
    """
    n_rollouts, n_layers, hidden_dim = all_hidden_states.shape

    if n_rollouts < 2:
        return _empty_metrics(n_layers)

    # Per-layer metrics
    layer_variances = np.zeros(n_layers)
    layer_cosine_dists = np.zeros(n_layers)

    for layer in range(n_layers):
        states = all_hidden_states[:, layer, :]  # (n_rollouts, hidden_dim)

        # Mean per-dimension variance across rollouts
        layer_variances[layer] = float(np.mean(np.var(states, axis=0)))

        # Mean pairwise cosine distance
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        normalized = states / (norms + 1e-8)
        cos_sim = normalized @ normalized.T  # (n_rollouts, n_rollouts)
        # Upper triangle (exclude diagonal)
        mask = np.triu(np.ones((n_rollouts, n_rollouts), dtype=bool), k=1)
        layer_cosine_dists[layer] = float(1.0 - np.mean(cos_sim[mask]))

    # Split layers into thirds
    third = n_layers // 3
    early_layers = slice(0, third)
    mid_layers = slice(third, 2 * third)
    late_layers = slice(2 * third, n_layers)

    # Aggregate variance metrics
    hs_var_early = float(np.mean(layer_variances[early_layers]))
    hs_var_mid = float(np.mean(layer_variances[mid_layers]))
    hs_var_late = float(np.mean(layer_variances[late_layers]))
    hs_var_total = float(np.mean(layer_variances))

    # Late-to-early ratio (high = understands but uncertain about computation)
    hs_var_late_early_ratio = hs_var_late / (hs_var_early + 1e-12)

    # Peak variance layer (normalized 0-1)
    hs_peak_layer = float(np.argmax(layer_variances) / max(n_layers - 1, 1))

    # Cosine distance aggregates
    hs_cosine_early = float(np.mean(layer_cosine_dists[early_layers]))
    hs_cosine_mid = float(np.mean(layer_cosine_dists[mid_layers]))
    hs_cosine_late = float(np.mean(layer_cosine_dists[late_layers]))
    hs_cosine_total = float(np.mean(layer_cosine_dists))
    hs_cosine_late_early_ratio = hs_cosine_late / (hs_cosine_early + 1e-12)

    return {
        "hs_var_early": hs_var_early,
        "hs_var_mid": hs_var_mid,
        "hs_var_late": hs_var_late,
        "hs_var_total": hs_var_total,
        "hs_var_late_early_ratio": hs_var_late_early_ratio,
        "hs_peak_layer": hs_peak_layer,
        "hs_cosine_early": hs_cosine_early,
        "hs_cosine_mid": hs_cosine_mid,
        "hs_cosine_late": hs_cosine_late,
        "hs_cosine_total": hs_cosine_total,
        "hs_cosine_late_early_ratio": hs_cosine_late_early_ratio,
        # Per-layer profiles for plotting
        "hs_layer_variances": layer_variances.tolist(),
        "hs_layer_cosine_dists": layer_cosine_dists.tolist(),
    }


def _empty_metrics(n_layers: int) -> dict:
    """Return zeroed metrics when there aren't enough rollouts."""
    return {
        "hs_var_early": 0.0,
        "hs_var_mid": 0.0,
        "hs_var_late": 0.0,
        "hs_var_total": 0.0,
        "hs_var_late_early_ratio": 0.0,
        "hs_peak_layer": 0.0,
        "hs_cosine_early": 0.0,
        "hs_cosine_mid": 0.0,
        "hs_cosine_late": 0.0,
        "hs_cosine_total": 0.0,
        "hs_cosine_late_early_ratio": 0.0,
        "hs_layer_variances": [0.0] * n_layers,
        "hs_layer_cosine_dists": [0.0] * n_layers,
    }


# Scalar metric keys (excludes per-layer profiles)
SCALAR_METRIC_KEYS = [
    "hs_var_early", "hs_var_mid", "hs_var_late", "hs_var_total",
    "hs_var_late_early_ratio", "hs_peak_layer",
    "hs_cosine_early", "hs_cosine_mid", "hs_cosine_late",
    "hs_cosine_total", "hs_cosine_late_early_ratio",
]
