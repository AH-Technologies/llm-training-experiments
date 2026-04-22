"""Prompt templates for the 3-turn student-teacher self-play pipeline.

All templates use XML tags to clearly separate different sections of the prompt,
making it easier for models to distinguish between problem statements, student
attempts, ground truth answers, and instructions.

Templates are composed from reusable building blocks so that each piece of
content is defined exactly once.
"""

# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------

_CONTEXT_BLOCK = (
    "<problem_statement>\n"
    "{question}\n"
    "</problem_statement>\n\n"
    "<student_attempt>\n"
    "{student_attempt}\n"
    "</student_attempt>\n\n"
)

_GT_BLOCK = (
    "<ground_truth_answer>\n"
    "{ground_truth}\n"
    "</ground_truth_answer>\n\n"
)

# --- Instructions (vary by filtered / blind) ---

_INSTRUCTION_SIGHTED = (
    "You are a teacher reviewing a student's {correctness}attempt at the problem above. "
    "Analyze the student's approach step by step, comparing it against the "
    "ground truth. Then provide your feedback to the student inside "
    "<feedback></feedback> tags."
)

_INSTRUCTION_BLIND = (
    "You are a teacher reviewing a student's {correctness}attempt at the problem above. "
    "Based on your own reasoning, analyze the student's approach step by step. "
    "Then provide your feedback to the student inside <feedback></feedback> tags."
)

# --- Rules ---

_RULE_NO_REVEAL_GT = "- Do NOT reveal the ground truth or final answer in your feedback."
_RULE_NO_REVEAL = "- Do NOT reveal the final answer in your feedback."

_RULE_CORRECT_CASE = (
    "- If the student's answer is correct, acknowledge this and suggest "
    "improvements to clarity or rigor."
)

_RULE_INCORRECT_CASE = (
    "- If the student's answer is incorrect, guide them toward fixing their "
    "mistake without giving the solution away."
)

_RULE_INCORRECT_ONLY = (
    "- Guide the student toward fixing their mistake without giving the "
    "solution away."
)


# ------------------------------------------------------------------
# Template composition helper
# ------------------------------------------------------------------

def _build_template(*, blind: bool, filtered: bool) -> str:
    parts = [_CONTEXT_BLOCK]

    if not blind:
        parts.append(_GT_BLOCK)

    instruction = _INSTRUCTION_BLIND if blind else _INSTRUCTION_SIGHTED
    correctness = "incorrect " if filtered else ""
    parts.append(instruction.format(correctness=correctness))

    no_reveal = _RULE_NO_REVEAL if blind else _RULE_NO_REVEAL_GT
    if filtered:
        rules = f"\n\nRules:\n{no_reveal}\n{_RULE_INCORRECT_ONLY}"
    else:
        rules = f"\n\nRules:\n{no_reveal}\n{_RULE_CORRECT_CASE}\n{_RULE_INCORRECT_CASE}"
    parts.append(rules)

    return "".join(parts)


# ------------------------------------------------------------------
# Public templates
# ------------------------------------------------------------------

TEACHER_PROMPT_TEMPLATE = _build_template(blind=False, filtered=False)
TEACHER_PROMPT_TEMPLATE_FILTERED = _build_template(blind=False, filtered=True)
BLIND_TEACHER_PROMPT_TEMPLATE = _build_template(blind=True, filtered=False)
BLIND_TEACHER_PROMPT_TEMPLATE_FILTERED = _build_template(blind=True, filtered=True)

# ------------------------------------------------------------------
# Student₂ prompt (single-turn with explicit context)
# ------------------------------------------------------------------

STUDENT2_PROMPT_TEMPLATE = (
    "<problem_statement>\n"
    "{question}\n"
    "</problem_statement>\n\n"
    "<your_first_attempt>\n"
    "{first_attempt}\n"
    "</your_first_attempt>\n\n"
    "<feedback_from_teacher>\n"
    "{feedback}\n"
    "</feedback_from_teacher>\n\n"
    "Use the feedback from your teacher to improve your first attempt at the problem.\n"
    "Put your final answer within \\boxed{{}}."
)
