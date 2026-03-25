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

# --- Instructions (vary by sighted / blind) ---

_INSTRUCTION_SIGHTED = (
    "You are a teacher reviewing a student's {correctness}attempt at the problem above. "
    "You have access to the ground truth answer.\n\n"
    "First, in a <scratchpad>, analyze the student's approach step by step, comparing "
    "it against the ground truth. Work through the full solution if needed.\n\n"
    "Then, in a <feedback> section, write clear guidance for the student. "
    "The student will ONLY see the <feedback> section, not the scratchpad."
)

_INSTRUCTION_BLIND = (
    "You are a teacher reviewing a student's {correctness}attempt at the problem above. "
    "Based on your own reasoning, analyze the student's approach.\n\n"
    "First, in a <scratchpad>, work through the problem yourself and compare "
    "with the student's approach.\n\n"
    "Then, in a <feedback> section, write clear guidance for the student. "
    "The student will ONLY see the <feedback> section, not the scratchpad."
)

# --- Rules ---

_RULES_UNFILTERED = (
    "\n\nRules:\n"
    "- Put all solution details, answer references, and working in <scratchpad>.\n"
    "- The <feedback> section must guide the student's reasoning without stating the answer.\n"
    "- If the student's answer is correct, acknowledge this and suggest improvements to clarity or rigor.\n"
    "- If the student's answer is incorrect, guide them toward fixing their mistake."
)

_RULES_FILTERED = (
    "\n\nRules:\n"
    "- Put all solution details, answer references, and working in <scratchpad>.\n"
    "- The <feedback> section must guide the student's reasoning without stating the answer.\n"
    "- Guide the student toward fixing their mistake."
)

_OUTPUT_FORMAT = (
    "\n\nRespond in this format:\n"
    "<scratchpad>\n[your private analysis]\n</scratchpad>\n"
    "<feedback>\n[student-facing guidance]\n</feedback>"
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

    parts.append(_RULES_FILTERED if filtered else _RULES_UNFILTERED)
    parts.append(_OUTPUT_FORMAT)

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
