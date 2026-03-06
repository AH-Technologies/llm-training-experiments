"""Prompt templates for the 3-turn student-teacher self-play pipeline.

All templates use XML tags to clearly separate different sections of the prompt,
making it easier for models to distinguish between problem statements, student
attempts, ground truth answers, and instructions.
"""

TEACHER_PROMPT_TEMPLATE = (
    "<problem_statement>\n"
    "{question}\n"
    "</problem_statement>\n\n"
    "<first_attempt>\n"
    "{student_attempt}\n"
    "</first_attempt>\n\n"
    "<ground_truth_answer>\n"
    "{ground_truth}\n"
    "</ground_truth_answer>\n\n"
    "<task>\n"
    "Pinpoint the step in which I am making a mistake. "
    "Provide the most informative piece of information for me to succeed "
    "on the next try. DO NOT tell me the final answer.\n"
    "</task>"
)

