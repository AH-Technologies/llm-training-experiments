"""Shannon entropy over the answer distribution.

Given k rollouts with extracted answers, compute the entropy of the
distribution over distinct answer strings. Higher entropy means more
diverse model outputs.

Also reports related sub-metrics:
- unique_answer_count: number of distinct answers (including correct)
- unique_wrong_answer_count: distinct answers excluding the correct one
- pass_rate: fraction of rollouts that are correct
"""

import math
from collections import Counter

from . import ExampleRollouts, register_metric


@register_metric("answer_entropy")
def answer_entropy(rollouts: ExampleRollouts) -> dict[str, float]:
    answers = [a if a is not None else "<no_answer>" for a in rollouts.extracted_answers]
    k = len(answers)

    # Count frequencies
    counter = Counter(answers)
    unique_count = len(counter)

    # Shannon entropy: H = -sum(p * log2(p))
    entropy = 0.0
    for count in counter.values():
        p = count / k
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalized entropy (0 to 1, comparable across different k)
    max_entropy = math.log2(unique_count) if unique_count > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Wrong-answer specific metrics
    correct_count = sum(rollouts.is_correct)
    pass_rate = correct_count / k if k > 0 else 0.0

    wrong_answers = [a for a, c in zip(answers, rollouts.is_correct) if not c]
    unique_wrong_count = len(set(wrong_answers))

    # Entropy over wrong answers only
    wrong_counter = Counter(wrong_answers)
    wrong_entropy = 0.0
    if wrong_answers:
        for count in wrong_counter.values():
            p = count / len(wrong_answers)
            if p > 0:
                wrong_entropy -= p * math.log2(p)

    return {
        "answer_entropy": entropy,
        "answer_entropy_normalized": normalized_entropy,
        "wrong_answer_entropy": wrong_entropy,
        "unique_answer_count": float(unique_count),
        "unique_wrong_answer_count": float(unique_wrong_count),
        "pass_rate": pass_rate,
    }
