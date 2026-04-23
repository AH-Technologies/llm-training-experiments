"""Tests for s1.pruning."""

from s1.pruning.tag_skills import (
    SKILL_CATEGORIES,
    PROMPT_TEMPLATE,
    build_prompt,
)


class TestSkillTaxonomy:
    def test_five_paper_native_categories(self):
        # Paper §4: Prealgebra+Algebra+IntermediateAlgebra merged into Algebra;
        # plus Number Theory, Geometry, Precalculus, Probability.
        assert SKILL_CATEGORIES == (
            "algebra",
            "number_theory",
            "geometry",
            "precalculus",
            "probability",
        )

    def test_categories_are_immutable_tuple(self):
        assert isinstance(SKILL_CATEGORIES, tuple)


class TestPromptBuilder:
    def test_prompt_contains_category_and_question(self):
        prompt = build_prompt(category="algebra", question="Solve for x: 2x+3=7")
        assert "algebra" in prompt
        assert "Solve for x: 2x+3=7" in prompt

    def test_prompt_has_answer_tags_instruction(self):
        prompt = build_prompt(category="geometry", question="Q")
        assert "<answer>" in prompt
        assert "</answer>" in prompt

    def test_prompt_specifies_colon_separator(self):
        prompt = build_prompt(category="geometry", question="Q")
        assert ":" in prompt  # paper: "separate the concepts or skills with :"
        assert "None" in prompt  # paper: "if there is no skills ... answer with None"
