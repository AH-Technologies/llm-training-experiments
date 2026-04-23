#!/usr/bin/env python3
"""Monkey-patch RLHFDataset.__getitem__ to append a random suffix to user messages.

This enables the "random_suffix" experimental condition where each time a prompt
is sampled, a randomly chosen prompting strategy suffix is appended.

Usage:
    from attention_based_rewards.scripts.suffix_patch import apply_random_suffix_patch
    apply_random_suffix_patch()  # Call BEFORE importing verl's main
"""

import logging
import random

logger = logging.getLogger(__name__)

# 36 diverse prompting strategy suffixes
SUFFIXES = [
    "Let's think step by step and output the final answer within \\boxed{}.",
    "Break this problem down into smaller parts and solve each one.",
    "Think carefully and show your work step by step.",
    "First, identify what we need to find. Then solve systematically.",
    "Approach this problem methodically and verify your answer.",
    "Let's reason through this carefully before giving the answer.",
    "Consider multiple approaches and pick the best one.",
    "Start with what you know and work towards the solution.",
    "Explain your reasoning clearly as you solve this.",
    "Let's solve this using a structured approach.",
    "Think about this problem from first principles.",
    "What are the key relationships here? Use them to solve.",
    "Work through this problem one step at a time.",
    "Let's be systematic. What information do we have?",
    "Solve this carefully, checking each step as you go.",
    "Take a deep breath and solve this step by step.",
    "Before solving, plan your approach. Then execute.",
    "Think like a mathematician. What's the elegant solution?",
    "Identify the pattern and use it to find the answer.",
    "Let's use logical reasoning to solve this problem.",
    "Consider what tools or formulas apply here.",
    "Start from the basics and build up to the solution.",
    "What would a careful problem-solver do here?",
    "Let's organize our thoughts and solve this systematically.",
    "Focus on the key insight needed to solve this.",
    "Walk me through your reasoning as you solve this.",
    "Check your work as you go. What's the answer?",
    "Think about edge cases and special conditions.",
    "Simplify the problem first, then solve.",
    "What's the most efficient way to solve this?",
    "Let's verify our approach makes sense, then solve.",
    "Use mathematical reasoning to find the answer.",
    "Decompose this into manageable subproblems.",
    "Reason carefully about each step of the solution.",
    "Apply the relevant mathematical concepts step by step.",
    "Show your complete reasoning, then box the final answer.",
]


def apply_random_suffix_patch(train_file_pattern: str = "dapo_math_17k"):
    """Monkey-patch RLHFDataset.__getitem__ to append a random suffix.

    Only applies the suffix to datasets whose data_files contain
    train_file_pattern, so validation sets are left untouched.
    """
    from verl.utils.dataset.rl_dataset import RLHFDataset

    original_getitem = RLHFDataset.__getitem__

    def _is_train_dataset(dataset):
        """Check if this dataset instance is a training set."""
        for f in getattr(dataset, "data_files", []):
            if train_file_pattern in str(f):
                return True
        return False

    def _patched_getitem(self, item):
        row_dict = original_getitem(self, item)
        if not _is_train_dataset(self):
            return row_dict
        messages = row_dict["raw_prompt"]
        for msg in reversed(messages):
            if msg["role"] == "user":
                suffix = random.choice(SUFFIXES)
                msg["content"] = msg["content"] + " " + suffix
                break
        return row_dict

    RLHFDataset.__getitem__ = _patched_getitem
    logger.info(f"Applied random suffix patch with {len(SUFFIXES)} suffixes (train only, pattern='{train_file_pattern}')")
