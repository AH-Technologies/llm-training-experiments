"""Prompt templates for the 3-turn student-teacher self-play pipeline."""

STUDENT1_SYSTEM = (
    "You are a math problem solver. "
    "Think step by step and put your final answer in \\boxed{}."
)

TEACHER_PROMPT = (
    "You are now a teacher reviewing a student's work.\n\n"
    "The student was asked: {question}\n\n"
    "The student answered:\n{student_answer}\n\n"
    "Provide constructive feedback to help the student improve their answer. "
    "Do NOT reveal the correct answer. Focus on the reasoning process."
)

STUDENT2_PROMPT = (
    "Your teacher has reviewed your work and provided feedback:\n\n"
    "{feedback}\n\n"
    "Based on this feedback, revise your answer. "
    "Think step by step and put your final answer in \\boxed{{}}."
)

