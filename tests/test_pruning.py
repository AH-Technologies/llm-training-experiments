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


import pyarrow as pa
import pyarrow.parquet as pq

from s1.pruning.tag_skills import tag_dataset


class FakeIdentifier:
    """Test double: returns deterministic skill lists keyed by category."""

    def __init__(self, per_category_skills: dict[str, list[str]]):
        self.per_category_skills = per_category_skills
        self._last_calls: list[tuple[str, str]] = []
        self._last_results: list[list[str]] = []

    def submit_batch(self, calls):
        self._last_calls = list(calls)
        self._last_results = [
            list(self.per_category_skills.get(category, []))
            for _, category in calls
        ]

    def collect_results(self):
        return self._last_results


class TestTagDataset:
    def test_writes_parquet_with_expected_schema(self, tmp_path):
        input_path = tmp_path / "s1k.parquet"
        output_path = tmp_path / "skills.parquet"
        input_table = pa.table({
            "question": ["Q1", "Q2", "Q3"],
            "cot_type": ["math", "math", "science"],
            "source_type": ["AIME", "Omni-MATH", "GPQA"],
        })
        pq.write_table(input_table, input_path)

        identifier = FakeIdentifier({
            "algebra": ["solve eq"],
            "number_theory": [],
            "geometry": [],
            "precalculus": ["derivative"],
            "probability": [],
        })

        tag_dataset(
            input_path=str(input_path),
            output_path=str(output_path),
            identifier=identifier,
        )

        out = pq.read_table(output_path).to_pydict()
        assert out["index"] == [0, 1, 2]
        assert out["question"] == ["Q1", "Q2", "Q3"]
        assert out["cot_type"] == ["math", "math", "science"]
        assert out["source_type"] == ["AIME", "Omni-MATH", "GPQA"]
        assert out["skills_algebra"] == [["solve eq"]] * 3
        assert out["skills_number_theory"] == [[]] * 3
        assert out["skills_precalculus"] == [["derivative"]] * 3
        # skill_count = sum of per-category list lengths = 1 + 1 = 2 for each row
        assert out["skill_count"] == [2, 2, 2]

    def test_issues_five_calls_per_row(self, tmp_path):
        input_path = tmp_path / "s1k.parquet"
        output_path = tmp_path / "skills.parquet"
        pq.write_table(
            pa.table({
                "question": ["Qa", "Qb"],
                "cot_type": ["math", "math"],
                "source_type": ["s", "s"],
            }),
            input_path,
        )
        identifier = FakeIdentifier({c: [] for c in SKILL_CATEGORIES})
        tag_dataset(
            input_path=str(input_path),
            output_path=str(output_path),
            identifier=identifier,
        )
        # 2 rows × 5 categories = 10 calls
        assert len(identifier._last_calls) == 10
        # Calls ordered as (Qa,algebra), (Qa,number_theory), ..., (Qb,...)
        expected = [(q, c) for q in ("Qa", "Qb") for c in SKILL_CATEGORIES]
        assert identifier._last_calls == expected


from s1.pruning.prune import random_select, skill_abundance_select


class TestRandomSelect:
    def test_returns_n_distinct_indices(self):
        selected = random_select(pool_size=100, n=10, seed=42)
        assert len(selected) == 10
        assert len(set(selected)) == 10

    def test_indices_in_range(self):
        selected = random_select(pool_size=100, n=10, seed=42)
        assert all(0 <= i < 100 for i in selected)

    def test_deterministic_given_seed(self):
        a = random_select(pool_size=1000, n=100, seed=42)
        b = random_select(pool_size=1000, n=100, seed=42)
        assert a == b

    def test_different_seeds_give_different_selections(self):
        a = random_select(pool_size=1000, n=100, seed=42)
        b = random_select(pool_size=1000, n=100, seed=43)
        assert a != b

    def test_n_equals_pool_returns_all_indices(self):
        selected = random_select(pool_size=10, n=10, seed=42)
        assert sorted(selected) == list(range(10))

    def test_n_greater_than_pool_raises(self):
        import pytest
        with pytest.raises(ValueError):
            random_select(pool_size=10, n=11, seed=42)


class TestSkillAbundanceSelect:
    def _write_skills(self, tmp_path, skill_counts: list[int]):
        path = tmp_path / "skills.parquet"
        pq.write_table(
            pa.table({
                "index": list(range(len(skill_counts))),
                "skill_count": skill_counts,
            }),
            path,
        )
        return str(path)

    def test_picks_top_by_skill_count(self, tmp_path):
        skills_path = self._write_skills(tmp_path, [1, 5, 3, 7, 2])
        selected = skill_abundance_select(skills_path, n=2)
        assert selected == [3, 1]  # indices sorted by count desc

    def test_ties_broken_by_index_ascending(self, tmp_path):
        skills_path = self._write_skills(tmp_path, [4, 4, 4, 4, 1])
        selected = skill_abundance_select(skills_path, n=3)
        assert selected == [0, 1, 2]

    def test_deterministic(self, tmp_path):
        skills_path = self._write_skills(tmp_path, [3, 1, 4, 1, 5, 9, 2, 6])
        a = skill_abundance_select(skills_path, n=4)
        b = skill_abundance_select(skills_path, n=4)
        assert a == b

    def test_n_equals_pool_returns_all(self, tmp_path):
        skills_path = self._write_skills(tmp_path, [2, 5, 1])
        selected = skill_abundance_select(skills_path, n=3)
        assert sorted(selected) == [0, 1, 2]

    def test_n_greater_than_pool_raises(self, tmp_path):
        import pytest
        skills_path = self._write_skills(tmp_path, [2, 5])
        with pytest.raises(ValueError):
            skill_abundance_select(skills_path, n=3)
