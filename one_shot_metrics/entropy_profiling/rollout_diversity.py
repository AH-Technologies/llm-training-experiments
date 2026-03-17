#!/usr/bin/env python3
"""Rollout diversity metrics: measure how diverse a model's rollouts are across
answer outcomes, token sequences, entropy curve shapes, and reasoning structures."""

import hashlib
from collections import Counter
from itertools import combinations

import numpy as np


def compute_answer_diversity(rollouts: list[dict]) -> dict:
    """Count distinct final answers and compute Shannon entropy of answer distribution.

    Args:
        rollouts: List of rollout dicts with 'extracted_answer' key.

    Returns:
        {num_unique_answers, answer_entropy}
    """
    answers = [r.get("extracted_answer") or "__NONE__" for r in rollouts]
    counts = Counter(answers)
    num_unique = len(counts)

    # Shannon entropy in nats
    total = len(answers)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * np.log(p)

    return {
        "num_unique_answers": num_unique,
        "answer_entropy": float(entropy),
    }


def _compute_bleu4(ref_tokens: list, hyp_tokens: list) -> float:
    """Compute sentence-level BLEU-4 score (no smoothing, inline n-gram implementation)."""
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) > 0 else 0.0

    log_avg = 0.0
    for n in range(1, 5):
        # Build n-gram counts
        ref_ngrams = Counter()
        for i in range(len(ref_tokens) - n + 1):
            ref_ngrams[tuple(ref_tokens[i:i + n])] += 1

        hyp_ngrams = Counter()
        for i in range(len(hyp_tokens) - n + 1):
            hyp_ngrams[tuple(hyp_tokens[i:i + n])] += 1

        # Clipped counts
        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = max(len(hyp_tokens) - n + 1, 1)

        precision = clipped / total if total > 0 else 0.0
        if precision == 0:
            return 0.0
        log_avg += np.log(precision) / 4.0

    return float(bp * np.exp(log_avg))


def compute_token_sequence_divergence(rollouts: list[dict], max_pairs: int = 64, seed: int = 42) -> dict:
    """Measure how early and how much rollout token sequences diverge.

    Args:
        rollouts: List of rollout dicts with 'token_ids' key.
        max_pairs: Maximum number of pairs to sample when C(n,2) > max_pairs.
        seed: Random seed for pair subsampling.

    Returns:
        {mean_common_prefix_len, mean_pairwise_bleu}
    """
    n = len(rollouts)
    if n < 2:
        return {"mean_common_prefix_len": 1.0, "mean_pairwise_bleu": 1.0}

    token_seqs = [r.get("token_ids", []) for r in rollouts]

    # Generate pairs (subsample if too many)
    all_pairs = list(combinations(range(n), 2))
    rng = np.random.RandomState(seed)
    if len(all_pairs) > max_pairs:
        indices = rng.choice(len(all_pairs), max_pairs, replace=False)
        pairs = [all_pairs[i] for i in indices]
    else:
        pairs = all_pairs

    prefix_lens = []
    bleu_scores = []

    for i, j in pairs:
        seq_i, seq_j = token_seqs[i], token_seqs[j]

        # Common prefix length (normalized by mean sequence length)
        common = 0
        min_len = min(len(seq_i), len(seq_j))
        for k in range(min_len):
            if seq_i[k] == seq_j[k]:
                common += 1
            else:
                break
        mean_len = (len(seq_i) + len(seq_j)) / 2.0
        prefix_lens.append(common / mean_len if mean_len > 0 else 0.0)

        # BLEU-4
        bleu_scores.append(_compute_bleu4(seq_i, seq_j))

    return {
        "mean_common_prefix_len": float(np.mean(prefix_lens)),
        "mean_pairwise_bleu": float(np.mean(bleu_scores)),
    }


def _interpolate_entropy(entropy_array: np.ndarray, num_bins: int = 100) -> np.ndarray:
    """Interpolate entropy array to fixed number of bins."""
    if len(entropy_array) <= 1:
        return np.zeros(num_bins)
    x_orig = np.linspace(0, 1, len(entropy_array))
    x_new = np.linspace(0, 1, num_bins)
    return np.interp(x_new, x_orig, entropy_array)


def _silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette score (manual implementation, no sklearn)."""
    n = len(data)
    if n < 2:
        return 0.0

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    silhouettes = np.zeros(n)
    for i in range(n):
        own_cluster = labels[i]
        own_mask = labels == own_cluster

        # a(i): mean distance to own cluster members
        if own_mask.sum() > 1:
            dists_own = np.linalg.norm(data[own_mask] - data[i], axis=1)
            a_i = dists_own.sum() / (own_mask.sum() - 1)
        else:
            a_i = 0.0

        # b(i): min mean distance to other clusters
        b_i = np.inf
        for label in unique_labels:
            if label == own_cluster:
                continue
            other_mask = labels == label
            dists_other = np.linalg.norm(data[other_mask] - data[i], axis=1)
            mean_dist = dists_other.mean()
            b_i = min(b_i, mean_dist)

        if b_i == np.inf:
            silhouettes[i] = 0.0
        else:
            silhouettes[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    return float(np.mean(silhouettes))


def compute_entropy_curve_clustering(rollouts: list[dict], num_bins: int = 100) -> dict:
    """Cluster entropy curves to find distinct trajectory shapes.

    Args:
        rollouts: List of rollout dicts with 'entropy_array' key.
        num_bins: Number of bins for curve interpolation.

    Returns:
        {num_entropy_clusters, positional_entropy_variance}
    """
    from scipy.cluster.vq import kmeans2

    n = len(rollouts)
    curves = np.array([_interpolate_entropy(r["entropy_array"], num_bins) for r in rollouts])

    # Positional entropy variance: mean variance across positions
    positional_var = float(np.mean(np.var(curves, axis=0)))

    if n < 8:
        return {
            "num_entropy_clusters": 1,
            "positional_entropy_variance": positional_var,
        }

    # Try k=2..min(8, n//4), pick best silhouette
    best_k = 1
    best_silhouette = -1.0

    max_k = min(8, n // 4)
    for k in range(2, max_k + 1):
        try:
            centroids, labels = kmeans2(curves, k, minit="++", iter=20, seed=42)
            # Check all clusters have at least 1 member
            if len(np.unique(labels)) < k:
                continue
            score = _silhouette_score(curves, labels)
            if score > best_silhouette:
                best_silhouette = score
                best_k = k
        except Exception:
            continue

    return {
        "num_entropy_clusters": best_k,
        "positional_entropy_variance": positional_var,
    }


def compute_reasoning_structure_diversity(rollouts: list[dict]) -> dict:
    """Analyze structural diversity of reasoning across rollouts.

    Args:
        rollouts: List of rollout dicts with 'text' key.

    Returns:
        {num_unique_structures, mean_num_steps, std_num_steps}
    """
    step_counts = []
    skeletons = []

    for r in rollouts:
        text = r.get("text", "")
        # Split on double newlines, count non-empty segments as steps
        segments = [s.strip() for s in text.split("\n\n") if s.strip()]
        num_steps = len(segments)
        step_counts.append(num_steps)

        # Skeleton: bucket each segment length (S<50, M<200, L>=200), hash the tuple
        size_labels = []
        for seg in segments:
            seg_len = len(seg)
            if seg_len < 50:
                size_labels.append("S")
            elif seg_len < 200:
                size_labels.append("M")
            else:
                size_labels.append("L")
        skeleton = hashlib.md5(str(tuple(size_labels)).encode()).hexdigest()
        skeletons.append(skeleton)

    step_counts = np.array(step_counts, dtype=float)
    num_unique = len(set(skeletons))

    return {
        "num_unique_structures": num_unique,
        "mean_num_steps": float(np.mean(step_counts)) if len(step_counts) > 0 else 0.0,
        "std_num_steps": float(np.std(step_counts)) if len(step_counts) > 0 else 0.0,
    }


def compute_all_diversity_metrics(rollouts: list[dict], max_pairs: int = 64) -> dict:
    """Compute all rollout diversity metrics.

    Args:
        rollouts: List of rollout dicts.
        max_pairs: Maximum pairs for token divergence computation.

    Returns:
        Dict with all 9 diversity metrics.
    """
    metrics = {}
    metrics.update(compute_answer_diversity(rollouts))
    metrics.update(compute_token_sequence_divergence(rollouts, max_pairs=max_pairs))
    metrics.update(compute_entropy_curve_clustering(rollouts))
    metrics.update(compute_reasoning_structure_diversity(rollouts))
    return metrics
