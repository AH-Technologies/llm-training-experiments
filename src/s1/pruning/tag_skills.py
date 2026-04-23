"""Skill identification for s1K pruning (Alibaba polymath-RL §4, App. C).

Paper-native: 5 math categories with Prealgebra+Algebra+IntermediateAlgebra
merged into `algebra` per §5 to eliminate overlaps. Uses the paper's
Table 8 prompt template verbatim, one call per (problem, category) pair.

Judge model is Gemini 2.5 Flash-Lite (paper uses Qwen2.5-72B); same
OpenAI-compatible API pattern as `src/self_teach/leakage_judge.py`.
"""

from __future__ import annotations

SKILL_CATEGORIES: tuple[str, ...] = (
    "algebra",
    "number_theory",
    "geometry",
    "precalculus",
    "probability",
)


# Paper Table 8 template, verbatim. [CATEGORY] and [QUESTION] substituted.
PROMPT_TEMPLATE = """\
Here is a reasoning problem, and your job is to identify the concepts and \
skills in the scope of {category} that are related to solve the problem.
Please separate the concepts or skills with :, and if there is no skills \
or concepts identified, please answer with None. Please put your answer \
within <answer></answer>.
For example: compute derivatives is the skill in precalculus.
Question:
{question}"""


def build_prompt(category: str, question: str) -> str:
    """Format the paper's Table 8 prompt for one (problem, category) call."""
    return PROMPT_TEMPLATE.format(category=category, question=question)
