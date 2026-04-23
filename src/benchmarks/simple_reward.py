"""
Simple reward function for GRPO benchmarking.

Returns a dummy reward to minimize compute overhead during benchmarking.
"""


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    """
    Simple reward: 1.0 if response contains the answer, 0.0 otherwise.

    This is intentionally simple to minimize reward computation overhead
    during benchmarking - we're measuring training throughput, not reward quality.
    """
    if ground_truth in solution_str:
        return 1.0
    return 0.0
