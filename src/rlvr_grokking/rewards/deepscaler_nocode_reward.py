"""Reward function with code penalty for no-code RLVR experiments.

Same as deepscaler_reward.py but applies a -0.5 penalty when the model
generates Python code (```python, import, def, print()).
"""

from rlvr_grokking.rewards.deepscaler_reward import (
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy,
)

CODE_MARKERS = ["```python", "```Python", "import ", "from sympy", "def ", "print("]
CODE_PENALTY = -0.5


def _has_code(text: str) -> bool:
    return any(marker in text for marker in CODE_MARKERS)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict = None,
) -> float:
    """Compute reward with code penalty.

    Returns:
        1.0 for correct without code
        0.5 for correct with code (1.0 + CODE_PENALTY)
        0.0 for incorrect without code
        -0.5 for incorrect with code (0.0 + CODE_PENALTY)
    """
    if not ground_truth:
        return CODE_PENALTY if _has_code(solution_str) else 0.0

    ground_truth = str(ground_truth)

    if '\\boxed' in ground_truth:
        ground_truth = extract_answer(ground_truth)
        if ground_truth is None:
            return CODE_PENALTY if _has_code(solution_str) else 0.0

    given_answer = extract_answer(solution_str)
    if given_answer is None:
        base_score = 0.0
    else:
        is_correct = grade_answer_mathd(given_answer, ground_truth) \
            or grade_answer_sympy(given_answer, ground_truth)
        base_score = 1.0 if is_correct else 0.0

    if _has_code(solution_str):
        return base_score + CODE_PENALTY

    return base_score
