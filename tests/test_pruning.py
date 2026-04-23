"""Tests for s1.pruning."""

from s1.pruning.tag_skills import (
    SKILL_CATEGORIES,
    PROMPT_TEMPLATE,
    build_prompt,
    parse_skills_response,
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


class TestParseSkillsResponse:
    def test_extracts_skills_between_answer_tags(self):
        raw = "<answer>solve linear equations: factor polynomials</answer>"
        assert parse_skills_response(raw) == [
            "solve linear equations",
            "factor polynomials",
        ]

    def test_none_returns_empty_list(self):
        assert parse_skills_response("<answer>None</answer>") == []

    def test_none_case_insensitive(self):
        assert parse_skills_response("<answer>none</answer>") == []
        assert parse_skills_response("<answer> NONE </answer>") == []

    def test_lowercases_skills(self):
        raw = "<answer>Solve Quadratics : Factor Polynomials</answer>"
        assert parse_skills_response(raw) == [
            "solve quadratics",
            "factor polynomials",
        ]

    def test_strips_whitespace_around_skills(self):
        raw = "<answer>  skill one  :  skill two  </answer>"
        assert parse_skills_response(raw) == ["skill one", "skill two"]

    def test_drops_empty_tokens(self):
        raw = "<answer>one: : two</answer>"
        assert parse_skills_response(raw) == ["one", "two"]

    def test_drops_literal_none_tokens_in_list(self):
        raw = "<answer>one : None : two</answer>"
        assert parse_skills_response(raw) == ["one", "two"]

    def test_missing_tags_returns_empty(self):
        assert parse_skills_response("no tags here") == []

    def test_ignores_text_outside_tags(self):
        raw = "Sure! <answer>skill a: skill b</answer> Hope that helps."
        assert parse_skills_response(raw) == ["skill a", "skill b"]

    def test_multiline_content(self):
        raw = "<answer>skill one:\n skill two\n:skill three</answer>"
        assert parse_skills_response(raw) == [
            "skill one",
            "skill two",
            "skill three",
        ]


import json
from unittest.mock import patch, MagicMock

from s1.pruning.tag_skills import SkillIdentifier


def _mock_response(content: str):
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestSkillIdentifier:
    def setup_method(self):
        self.identifier = SkillIdentifier(
            api_base="https://api.example.com/v1",
            api_key="test-key",
            model="gemini-2.5-flash-lite",
            max_workers=2,
            timeout=5.0,
        )

    @patch("s1.pruning.tag_skills.urlopen")
    def test_identify_single_returns_parsed_skills(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(
            "<answer>solve linear eq: factor polynomials</answer>"
        )
        skills = self.identifier._identify_single(
            question="2x+3=7", category="algebra"
        )
        assert skills == ["solve linear eq", "factor polynomials"]

    @patch("s1.pruning.tag_skills.urlopen")
    def test_identify_single_none_returns_empty(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response("<answer>None</answer>")
        assert self.identifier._identify_single("Q", "algebra") == []

    @patch("s1.pruning.tag_skills.urlopen")
    def test_api_error_returns_empty_and_increments_error_count(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("timeout")
        assert self.identifier._identify_single("Q", "algebra") == []
        assert self.identifier._error_count == 1

    @patch("s1.pruning.tag_skills.urlopen")
    def test_payload_uses_temperature_zero(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response("<answer>None</answer>")
        self.identifier._identify_single("Q", "algebra")
        # Capture the Request that was passed to urlopen
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["temperature"] == 0.0
        assert body["model"] == "gemini-2.5-flash-lite"

    @patch("s1.pruning.tag_skills.urlopen")
    def test_submit_collect_batch(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _mock_response("<answer>skill a</answer>"),
            _mock_response("<answer>None</answer>"),
            _mock_response("<answer>skill c: skill d</answer>"),
        ]
        self.identifier.submit_batch([
            ("Q1", "algebra"),
            ("Q2", "geometry"),
            ("Q3", "number_theory"),
        ])
        results = self.identifier.collect_results()
        assert results == [
            ["skill a"],
            [],
            ["skill c", "skill d"],
        ]
