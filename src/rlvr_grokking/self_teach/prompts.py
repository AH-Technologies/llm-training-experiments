"""Prompt templates for the 3-turn student-teacher self-play pipeline."""

TEACHER_PROMPT_TEMPLATE = (
    "Problem Statement\n"
    "I am trying to solve this question.\n"
    "{question}\n"
    "Here is my current attempt:\n"
    "{student_attempt}\n"
    "Here is the ground-truth math solution:\n"
    "{ground_truth}\n"
    "Pinpoint the step in which I am making a mistake. "
    "Provide the most informative piece of information for me to succeed "
    "on the next try, without telling me the final answer."
)
