"""Reward function compatible with verl's RewardManager interface.

verl expects a function with signature:
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

This module provides that interface for math problem evaluation.
"""

import re
import os
import json
import logging
from typing import Any
from fractions import Fraction
from datetime import datetime

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
    normalized_extracted: str,
    normalized_truth: str,
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
        "normalized_extracted": normalized_extracted,
        "normalized_truth": normalized_truth,
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


def extract_boxed_answer(response: str) -> str | None:
    """Extract answer from \\boxed{...} format.

    Handles nested braces properly. Also handles \\boxed without braces.
    """
    # Try \boxed{...} first
    pattern = r'\\boxed\s*\{'
    match = re.search(pattern, response)
    if match:
        start = match.end()
        brace_count = 1
        pos = start

        while pos < len(response) and brace_count > 0:
            if response[pos] == '{':
                brace_count += 1
            elif response[pos] == '}':
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            return response[start:pos-1].strip()

    # Try \boxed without braces (e.g., \boxed{4/3} sometimes written as \boxed 4/3)
    pattern2 = r'\\boxed\s+([^\s\\]+)'
    match2 = re.search(pattern2, response)
    if match2:
        return match2.group(1).strip()

    return None


def extract_frac(s: str) -> tuple[str, str] | None:
    """Extract numerator and denominator from \\frac{a}{b} pattern."""
    match = re.search(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', s)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # Handle nested braces
    pattern = r'\\frac\s*\{'
    match = re.search(pattern, s)
    if not match:
        return None

    start = match.end()
    brace_count = 1
    pos = start

    while pos < len(s) and brace_count > 0:
        if s[pos] == '{':
            brace_count += 1
        elif s[pos] == '}':
            brace_count -= 1
        pos += 1

    if brace_count != 0:
        return None

    num = s[start:pos-1].strip()

    # Find second argument
    rest = s[pos:]
    match2 = re.search(r'\s*\{', rest)
    if not match2:
        return None

    start2 = match2.end()
    brace_count = 1
    pos2 = start2

    while pos2 < len(rest) and brace_count > 0:
        if rest[pos2] == '{':
            brace_count += 1
        elif rest[pos2] == '}':
            brace_count -= 1
        pos2 += 1

    if brace_count != 0:
        return None

    denom = rest[start2:pos2-1].strip()
    return num, denom


def normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison.

    - Remove whitespace
    - Normalize fractions (\\frac{a}{b} -> a/b)
    - Remove \\left and \\right
    - Normalize common LaTeX commands
    - Handle dfrac, tfrac, etc.
    """
    if answer is None:
        return ""

    s = answer.strip()

    # Remove \left and \right
    s = re.sub(r'\\left\s*', '', s)
    s = re.sub(r'\\right\s*', '', s)

    # Remove \displaystyle
    s = re.sub(r'\\displaystyle\s*', '', s)

    # Handle \dfrac, \tfrac same as \frac
    s = re.sub(r'\\[dt]frac', r'\\frac', s)

    # Convert \frac{a}{b} to a/b
    frac_result = extract_frac(s)
    if frac_result:
        num, denom = frac_result
        # Recursively normalize the numerator and denominator
        num_norm = normalize_math_answer(num)
        denom_norm = normalize_math_answer(denom)
        s = f"{num_norm}/{denom_norm}"
    else:
        # Simple regex replacement for non-nested fracs
        s = re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'\1/\2', s)

    # Remove spaces
    s = re.sub(r'\s+', '', s)

    # Normalize some common forms
    s = s.replace('\\cdot', '*')
    s = s.replace('\\times', '*')
    s = s.replace('\\div', '/')
    s = s.replace('\\pm', '±')

    # Remove unnecessary parentheses around simple expressions
    if s.startswith('(') and s.endswith(')'):
        inner = s[1:-1]
        if '(' not in inner and ')' not in inner:
            s = inner

    return s


def try_parse_number(s: str) -> float | None:
    """Try to parse a string as a number, handling fractions and decimals."""
    if not s:
        return None

    s = s.strip()

    # Remove parentheses
    s = s.replace('(', '').replace(')', '')

    # Handle negative numbers with various notations
    s = s.replace('−', '-')  # Unicode minus

    try:
        # Try direct float conversion
        return float(s)
    except ValueError:
        pass

    # Try fraction a/b
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    return num / denom
            except ValueError:
                pass

    # Try Python's Fraction for exact arithmetic
    try:
        frac = Fraction(s)
        return float(frac)
    except (ValueError, ZeroDivisionError):
        pass

    return None


def answers_match(extracted: str, ground_truth: str, tolerance: float = 1e-6) -> tuple[bool, str]:
    """Compare two answers, returning (match, match_type).

    Returns:
        (True/False, match_type) where match_type is one of:
        - "exact_string": exact string match after normalization
        - "numeric": numeric values match within tolerance
        - "no_match": answers don't match
        - "parse_error": couldn't parse one or both answers
    """
    # Normalize both
    norm_ext = normalize_math_answer(extracted)
    norm_truth = normalize_math_answer(ground_truth)

    # Exact string match
    if norm_ext == norm_truth:
        return True, "exact_string"

    # Try numeric comparison
    ext_val = try_parse_number(norm_ext)
    truth_val = try_parse_number(norm_truth)

    if ext_val is None or truth_val is None:
        return False, "parse_error"

    # Check if values match within tolerance
    if abs(ext_val - truth_val) < tolerance:
        return True, "numeric"

    # Also check relative tolerance for larger numbers
    if truth_val != 0 and abs((ext_val - truth_val) / truth_val) < tolerance:
        return True, "numeric_relative"

    return False, "no_match"


def extract_answer(response: str) -> str | None:
    """Extract the final answer from model response.

    Looks for patterns like:
    - \\boxed{X} (for math problems)
    - "Answer: X" or "answer: X"
    - "the answer is X"
    - "= X" at the end
    - Just the number if response is simple
    - Last number/fraction in the response
    """
    response = response.strip()

    # Try <answer>...</answer> first (our prompted format)
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        # Try to extract a boxed answer within the answer tags
        boxed_in_answer = extract_boxed_answer(answer_content)
        if boxed_in_answer:
            return boxed_in_answer
        # Otherwise return the content (strip trailing punctuation)
        answer_content = re.sub(r'[.\s]+$', '', answer_content)
        if answer_content:
            return answer_content

    # Try \boxed{} (for math problems) - primary format for models not using <answer> tags
    boxed = extract_boxed_answer(response)
    if boxed:
        return boxed

    # Try "the answer is X" pattern
    match = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+([^\n,.]+)", response)
    if match:
        return match.group(1).strip()

    # Try "Answer: X" pattern (handles fractions)
    match = re.search(r"[Aa]nswer[:\s]+([^\n,]+?)(?:\.|$|\n)", response)
    if match:
        ans = match.group(1).strip()
        if ans:
            return ans

    # Try "= X" pattern at end (handles fractions)
    match = re.search(r"=\s*([^\n=]+?)\s*$", response)
    if match:
        ans = match.group(1).strip()
        # Clean up common suffixes
        ans = re.sub(r'\.$', '', ans)
        if ans:
            return ans

    # Try to find a fraction pattern like 4/3 or \frac{4}{3}
    frac_match = re.search(r'\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}', response)
    if frac_match:
        return frac_match.group(0)

    # Try simple fraction a/b
    frac_match = re.search(r'(-?\d+)\s*/\s*(-?\d+)', response)
    if frac_match:
        return frac_match.group(0)

    # Try just a number (integer or decimal)
    match = re.search(r"^(-?\d+\.?\d*)$", response)
    if match:
        return match.group(1)

    # Last number in response as fallback (including decimals)
    matches = re.findall(r"(-?\d+\.?\d*)", response)
    if matches:
        return matches[-1]

    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> float:
    """Compute reward for a generated solution.

    This is the main entry point for verl's reward computation.

    Args:
        data_source: Identifier for the dataset (e.g., "pi13", "math500")
        solution_str: The model's generated response
        ground_truth: The correct answer (from reward_model.ground_truth)
        extra_info: Optional metadata

    Returns:
        Reward value:
        - 1.0 for correct answer
        - 0.0 for incorrect or no valid answer extracted
    """
    # Extract the answer from the response
    extracted = extract_answer(solution_str)

    if extracted is None:
        _log_reward(
            data_source=data_source,
            ground_truth=ground_truth,
            extracted=None,
            normalized_extracted="",
            normalized_truth=normalize_math_answer(str(ground_truth)),
            reward=0.0,
            match_type="no_extraction",
            response_preview=solution_str,
        )
        return 0.0

    # Use flexible comparison for math problems
    normalized_extracted = normalize_math_answer(extracted)
    normalized_truth = normalize_math_answer(str(ground_truth))

    match, match_type = answers_match(extracted, str(ground_truth))
    reward = 1.0 if match else 0.0

    _log_reward(
        data_source=data_source,
        ground_truth=ground_truth,
        extracted=extracted,
        normalized_extracted=normalized_extracted,
        normalized_truth=normalized_truth,
        reward=reward,
        match_type=match_type,
        response_preview=solution_str,
    )

    return reward


# Alternative reward functions for different scoring schemes


def compute_score_binary(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> float:
    """Binary reward: 1.0 for correct, 0.0 for incorrect.

    No partial credit - stricter than compute_score.
    """
    extracted = extract_answer(solution_str)

    if extracted is None:
        return 0.0

    try:
        answer = int(extracted)
        return 1.0 if answer == int(ground_truth) else 0.0
    except (ValueError, TypeError):
        return 0.0


def compute_score_with_format_bonus(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> float:
    """Reward with bonus for good formatting.

    - 1.0 for correct answer with \\boxed{} format
    - 0.8 for correct answer without explicit format
    - 0.1 for wrong but formatted answer
    - 0.0 for no answer
    """
    response = solution_str.strip()

    # Check for \boxed{} format
    boxed = extract_boxed_answer(response)

    if boxed:
        match, _ = answers_match(boxed, str(ground_truth))
        if match:
            return 1.0  # Correct with good format
        else:
            return 0.1  # Wrong but good format

    # Fallback to any extraction
    extracted = extract_answer(response)
    if extracted is None:
        return 0.0

    match, _ = answers_match(extracted, str(ground_truth))
    if match:
        return 0.8  # Correct but not explicitly formatted
    else:
        return 0.05  # Wrong and poorly formatted
