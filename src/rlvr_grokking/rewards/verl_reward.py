"""Reward function compatible with verl's RewardManager interface.

verl expects a function with signature:
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

Uses veRL's built-in prime_math module for robust SymPy-based math grading,
which handles LaTeX expressions, fractions, symbolic equivalence, etc.
"""

import os
import json
import logging
from typing import Any
from datetime import datetime

from verl.utils.reward_score.prime_math import (
    grade_answer,
    match_answer,
    _last_boxed_only_string,
)
from verl.utils.reward_score.prime_math.grader import math_equal

# Setup logging
_log_dir = os.environ.get("REWARD_LOG_DIR", "./logs/rewards")
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, f"reward_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

# Global counter for logging
_reward_counter = {"count": 0, "correct": 0, "total_reward": 0.0}

# Try to import wandb for logging
try:
    import wandb
    _wandb_available = wandb.run is not None
except ImportError:
    _wandb_available = False
    wandb = None


def _log_reward(
    data_source: str,
    ground_truth: Any,
    extracted: str | None,
    reward: float,
    match_type: str,
    response_preview: str,
):
    """Log reward computation details to file and wandb."""
    _reward_counter["count"] += 1
    _reward_counter["total_reward"] += reward
    if reward > 0.5:
        _reward_counter["correct"] += 1

    log_entry = {
        "step": _reward_counter["count"],
        "data_source": data_source,
        "ground_truth": str(ground_truth),
        "extracted_answer": extracted,
        "reward": reward,
        "match_type": match_type,
        "response_preview": response_preview[:200] if response_preview else "",
        "running_accuracy": _reward_counter["correct"] / _reward_counter["count"],
        "running_avg_reward": _reward_counter["total_reward"] / _reward_counter["count"],
    }

    # Log to file
    try:
        with open(_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

    # Log to wandb if available (every 100 samples to avoid overhead)
    if _wandb_available and _reward_counter["count"] % 100 == 0:
        try:
            wandb.log({
                "reward/running_accuracy": log_entry["running_accuracy"],
                "reward/running_avg_reward": log_entry["running_avg_reward"],
                "reward/total_samples": _reward_counter["count"],
                "reward/last_reward": reward,
            })
        except Exception:
            pass

    # Print periodic summary
    if _reward_counter["count"] % 500 == 0:
        print(f"[Reward] Samples: {_reward_counter['count']}, "
              f"Accuracy: {log_entry['running_accuracy']:.3f}, "
              f"Avg Reward: {log_entry['running_avg_reward']:.3f}")


def extract_answer(response: str) -> str | None:
    """Extract answer from model response.

    Priority:
    1. Last \\boxed{...} in response
    2. "answer is" / "answer:" pattern via prime_math's match_answer
    No aggressive fallbacks (no "last number in response").
    """
    # 1. Try \boxed{} — the standard math format
    boxed = _last_boxed_only_string(response)
    if boxed is not None:
        return boxed

    # 2. Use prime_math's match_answer for "answer is", "answer:", etc.
    is_matched, extracted = match_answer(response)
    if is_matched and extracted:
        return extracted

    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> float:
    """Compute reward for a generated solution.

    Uses veRL's prime_math for robust SymPy-based math grading that handles
    LaTeX expressions, fractions, symbolic equivalence, intervals, etc.

    Args:
        data_source: Identifier for the dataset
        solution_str: The model's generated response
        ground_truth: The correct answer
        extra_info: Optional metadata

    Returns:
        1.0 for correct answer, 0.0 for incorrect or no valid answer extracted.
    """
    ground_truth_str = str(ground_truth)

    # Extract answer
    extracted = extract_answer(solution_str)

    if extracted is None:
        _log_reward(
            data_source=data_source,
            ground_truth=ground_truth,
            extracted=None,
            reward=0.0,
            match_type="no_extraction",
            response_preview=solution_str,
        )
        return 0.0

    # Grade using prime_math's grade_answer (SymPy-based)
    is_correct = grade_answer(extracted, ground_truth_str)
    match_type = "grade_answer"

    # If grade_answer didn't match, try math_equal as fallback
    # (handles pi, percentages, intervals, matrices)
    if not is_correct:
        try:
            is_correct = math_equal(extracted, ground_truth_str, timeout=True)
            if is_correct:
                match_type = "math_equal"
        except Exception:
            pass

    reward = 1.0 if is_correct else 0.0

    _log_reward(
        data_source=data_source,
        ground_truth=ground_truth,
        extracted=extracted,
        reward=reward,
        match_type=match_type if is_correct else "no_match",
        response_preview=solution_str,
    )

    return reward
