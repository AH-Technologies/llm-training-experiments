"""Tests for self-teach prompt templates."""
import re
from src.self_teach.prompts import (
    TEACHER_PROMPT_TEMPLATE,
    TEACHER_PROMPT_TEMPLATE_FILTERED,
    BLIND_TEACHER_PROMPT_TEMPLATE,
    BLIND_TEACHER_PROMPT_TEMPLATE_FILTERED,
    STUDENT2_PROMPT_TEMPLATE,
)
from src.self_teach.trainer import extract_feedback


class TestTeacherPromptTemplates:
    """Verify teacher prompts include scratchpad/feedback structure."""

    def test_sighted_unfiltered_has_scratchpad_instruction(self):
        prompt = TEACHER_PROMPT_TEMPLATE.format(
            question="What is 2+2?",
            student_attempt="I think it's 5.",
            ground_truth="4",
        )
        assert "<scratchpad>" in prompt
        assert "<feedback>" in prompt
        assert "Do NOT reveal" not in prompt  # old rule removed

    def test_sighted_filtered_has_scratchpad_instruction(self):
        prompt = TEACHER_PROMPT_TEMPLATE_FILTERED.format(
            question="What is 2+2?",
            student_attempt="I think it's 5.",
            ground_truth="4",
        )
        assert "<scratchpad>" in prompt
        assert "<feedback>" in prompt

    def test_blind_unfiltered_has_scratchpad_instruction(self):
        prompt = BLIND_TEACHER_PROMPT_TEMPLATE.format(
            question="What is 2+2?",
            student_attempt="I think it's 5.",
        )
        assert "<scratchpad>" in prompt
        assert "<feedback>" in prompt
        assert "{ground_truth}" not in prompt

    def test_blind_filtered_has_scratchpad_instruction(self):
        prompt = BLIND_TEACHER_PROMPT_TEMPLATE_FILTERED.format(
            question="What is 2+2?",
            student_attempt="I think it's 5.",
        )
        assert "<scratchpad>" in prompt
        assert "<feedback>" in prompt

    def test_ground_truth_block_in_sighted_only(self):
        sighted = TEACHER_PROMPT_TEMPLATE.format(
            question="Q", student_attempt="A", ground_truth="GT"
        )
        blind = BLIND_TEACHER_PROMPT_TEMPLATE.format(
            question="Q", student_attempt="A"
        )
        assert "<ground_truth_answer>" in sighted
        assert "<ground_truth_answer>" not in blind


class TestExtractFeedback:
    """Verify feedback extraction ignores scratchpad."""

    def test_extracts_feedback_only(self):
        text = (
            "<scratchpad>\nThe answer is 42. Student got it wrong.\n</scratchpad>\n"
            "<feedback>\nCheck your arithmetic in step 3.\n</feedback>"
        )
        result = extract_feedback(text)
        assert "42" not in result
        assert "Check your arithmetic" in result

    def test_fallback_when_no_tags(self):
        text = "Some unstructured response."
        result = extract_feedback(text)
        assert result == text

    def test_ignores_think_blocks(self):
        text = (
            "<think>internal reasoning</think>\n"
            "<scratchpad>The answer is 7.</scratchpad>\n"
            "<feedback>Reconsider step 2.</feedback>"
        )
        result = extract_feedback(text)
        assert "Reconsider step 2" in result
        assert "7" not in result


class TestStudent2Prompt:
    """Student2 prompt should be unchanged."""

    def test_student2_template_structure(self):
        prompt = STUDENT2_PROMPT_TEMPLATE.format(
            question="What is 2+2?",
            first_attempt="5",
            feedback="Check your addition.",
        )
        assert "<problem_statement>" in prompt
        assert "<your_first_attempt>" in prompt
        assert "<feedback_from_teacher>" in prompt
        assert "\\boxed{}" in prompt
