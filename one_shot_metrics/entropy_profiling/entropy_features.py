#!/usr/bin/env python3
"""Extract summary features from per-token entropy arrays."""

import numpy as np

from rollout_diversity import compute_all_diversity_metrics


def extract_single_rollout_features(entropy_array: np.ndarray) -> dict:
    """Extract features from a single rollout's entropy curve.

    Args:
        entropy_array: 1D array of per-token entropy values (nats)

    Returns:
        Dict of feature name -> value
    """
    if len(entropy_array) == 0:
        return {k: 0.0 for k in [
            "mean_entropy", "median_entropy", "std_entropy", "max_entropy",
            "num_spikes", "mean_spike_magnitude", "mean_spike_position",
            "high_entropy_ratio", "early_mean", "late_mean", "entropy_trend",
            "num_tokens",
        ]}

    n = len(entropy_array)
    mean = float(np.mean(entropy_array))
    median = float(np.median(entropy_array))
    std = float(np.std(entropy_array))
    max_ent = float(np.max(entropy_array))

    # Spike detection: adaptive threshold = median + 1.5 * IQR
    q1 = np.percentile(entropy_array, 25)
    q3 = np.percentile(entropy_array, 75)
    iqr = q3 - q1
    spike_threshold = median + 1.5 * iqr

    spike_mask = entropy_array > spike_threshold
    num_spikes = int(np.sum(spike_mask))

    if num_spikes > 0:
        spike_values = entropy_array[spike_mask]
        spike_positions = np.where(spike_mask)[0] / max(n - 1, 1)  # normalized 0-1
        mean_spike_magnitude = float(np.mean(spike_values - spike_threshold))
        mean_spike_position = float(np.mean(spike_positions))
    else:
        mean_spike_magnitude = 0.0
        mean_spike_position = 0.0

    # High-entropy ratio (fraction above global threshold)
    # Use a reasonable absolute threshold: log(100) ≈ 4.6 nats
    high_threshold = 4.0
    high_entropy_ratio = float(np.mean(entropy_array > high_threshold))

    # Trend: early vs late entropy
    quarter = max(n // 4, 1)
    early_mean = float(np.mean(entropy_array[:quarter]))
    late_mean = float(np.mean(entropy_array[-quarter:]))
    entropy_trend = late_mean - early_mean

    return {
        "mean_entropy": mean,
        "median_entropy": median,
        "std_entropy": std,
        "max_entropy": max_ent,
        "num_spikes": num_spikes,
        "mean_spike_magnitude": mean_spike_magnitude,
        "mean_spike_position": mean_spike_position,
        "high_entropy_ratio": high_entropy_ratio,
        "early_mean": early_mean,
        "late_mean": late_mean,
        "entropy_trend": entropy_trend,
        "num_tokens": n,
    }


def extract_example_features(rollouts: list[dict]) -> dict:
    """Extract aggregated features across all rollouts for one example.

    Args:
        rollouts: List of rollout dicts with 'entropy_array' key

    Returns:
        Dict with mean/std of each feature + cross-rollout variance
    """
    per_rollout = [extract_single_rollout_features(r["entropy_array"]) for r in rollouts]

    feature_names = list(per_rollout[0].keys())
    aggregated = {}

    for feat in feature_names:
        values = np.array([r[feat] for r in per_rollout])
        aggregated[f"{feat}_mean"] = float(np.mean(values))
        aggregated[f"{feat}_std"] = float(np.std(values))

    # Cross-rollout entropy variance: how much the entropy curves differ
    # Compute pairwise variance of mean entropies
    mean_entropies = np.array([r["mean_entropy"] for r in per_rollout])
    aggregated["cross_rollout_entropy_var"] = float(np.var(mean_entropies))

    # Pass rate (if available)
    correct = [r.get("is_correct", False) for r in rollouts]
    aggregated["pass_rate"] = float(np.mean(correct))

    # Rollout diversity metrics
    diversity = compute_all_diversity_metrics(rollouts)
    aggregated.update(diversity)

    return aggregated


def extract_all_features(all_results: dict) -> dict:
    """Extract features for all examples.

    Args:
        all_results: Dict mapping example name -> {example, rollouts, pass_rate}

    Returns:
        Dict mapping example name -> aggregated features
    """
    features = {}
    for name, result in all_results.items():
        features[name] = extract_example_features(result["rollouts"])
    return features
