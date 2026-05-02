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

    def test_raises_on_widespread_api_failure(self, tmp_path):
        input_path = tmp_path / "s1k.parquet"
        output_path = tmp_path / "skills.parquet"
        pq.write_table(
            pa.table({
                "question": [f"Q{i}" for i in range(4)],
                "cot_type": ["math"] * 4,
                "source_type": ["s"] * 4,
            }),
            input_path,
        )

        class FailingIdentifier(FakeIdentifier):
            def submit_batch(self, calls):
                super().submit_batch(calls)
                # Simulate: every call returned empty list, and most errored
                self._error_count = len(calls)  # >50% threshold
                self._call_count = len(calls)

        identifier = FailingIdentifier({c: [] for c in SKILL_CATEGORIES})

        import pytest
        with pytest.raises(RuntimeError, match="Skill tagging failed"):
            tag_dataset(
                input_path=str(input_path),
                output_path=str(output_path),
                identifier=identifier,
            )


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


from s1.pruning.prune import write_subset


class TestWriteSubset:
    def _write_input(self, tmp_path):
        path = tmp_path / "s1k.parquet"
        pq.write_table(
            pa.table({
                "question": [f"Q{i}" for i in range(10)],
                "solution": [f"S{i}" for i in range(10)],
                "thinking_trajectories": [[f"T{i}"] for i in range(10)],
            }),
            path,
        )
        return str(path)

    def test_writes_subset_with_same_schema(self, tmp_path):
        input_path = self._write_input(tmp_path)
        output_path = tmp_path / "subset.parquet"
        write_subset(
            input_path=input_path,
            output_path=str(output_path),
            indices=[0, 3, 7],
        )
        out = pq.read_table(output_path)
        assert out.column_names == ["question", "solution", "thinking_trajectories"]
        assert out.column("question").to_pylist() == ["Q0", "Q3", "Q7"]
        assert out.column("solution").to_pylist() == ["S0", "S3", "S7"]

    def test_preserves_index_order(self, tmp_path):
        input_path = self._write_input(tmp_path)
        output_path = tmp_path / "subset.parquet"
        # Intentionally unsorted input indices
        write_subset(
            input_path=input_path,
            output_path=str(output_path),
            indices=[7, 0, 3],
        )
        out = pq.read_table(output_path)
        assert out.column("question").to_pylist() == ["Q7", "Q0", "Q3"]


from s1.pruning.skill_coverage import (
    _gather_skill_sets,
    _greedy_facility_location,
    _jaccard_matrix,
    compute_ranking,
    load_or_compute_ranking,
)


def _skills_table(rows: list[dict[str, list[str]]]) -> pa.Table:
    """Build an s1k_skills.parquet-shaped table from per-row category dicts."""
    columns = {"index": list(range(len(rows)))}
    for cat in SKILL_CATEGORIES:
        columns[f"skills_{cat}"] = [r.get(cat, []) for r in rows]
    columns["skill_count"] = [
        sum(len(r.get(c, [])) for c in SKILL_CATEGORIES) for r in rows
    ]
    return pa.table(columns)


class TestGatherSkillSets:
    def test_categories_prefixed_to_avoid_collision(self):
        table = _skills_table([
            {"algebra": ["linear_eq"], "geometry": ["linear_eq"]},
        ])
        sets = _gather_skill_sets(table)
        assert sets[0] == {"algebra:linear_eq", "geometry:linear_eq"}

    def test_empty_row_is_empty_set(self):
        table = _skills_table([{}])
        assert _gather_skill_sets(table) == [set()]

    def test_handles_none_category_lists(self):
        # Constructed manually because _skills_table substitutes [] for missing.
        columns = {"index": [0]}
        for cat in SKILL_CATEGORIES:
            columns[f"skills_{cat}"] = [None]
        columns["skill_count"] = [0]
        table = pa.table(columns)
        assert _gather_skill_sets(table) == [set()]


class TestJaccardMatrix:
    def test_diagonal_is_one(self):
        sets = [{"a"}, {"b"}, {"c"}]
        sim = _jaccard_matrix(sets)
        for i in range(3):
            assert sim[i, i] == 1.0

    def test_disjoint_sets_give_zero(self):
        sim = _jaccard_matrix([{"a"}, {"b"}])
        assert sim[0, 1] == 0.0

    def test_identical_sets_give_one(self):
        sim = _jaccard_matrix([{"a", "b"}, {"a", "b"}])
        assert sim[0, 1] == 1.0

    def test_partial_overlap(self):
        # |{a,b} ∩ {b,c}| / |{a,b,c}| = 1/3
        sim = _jaccard_matrix([{"a", "b"}, {"b", "c"}])
        assert abs(float(sim[0, 1]) - 1.0 / 3.0) < 1e-6

    def test_two_empty_sets_give_zero_not_nan(self):
        # Avoid degenerate Jaccard 0/0 → NaN.
        sim = _jaccard_matrix([set(), set()])
        assert sim[0, 1] == 0.0

    def test_matrix_is_symmetric(self):
        sim = _jaccard_matrix([{"a", "b"}, {"b"}, {"c"}])
        for i in range(3):
            for j in range(3):
                assert sim[i, j] == sim[j, i]


class TestGreedyFacilityLocation:
    def test_returns_distinct_indices(self):
        import numpy as np
        sim = _jaccard_matrix([{"a"}, {"b"}, {"a", "b"}, {"c"}])
        order = _greedy_facility_location(sim, k=4)
        assert sorted(order) == [0, 1, 2, 3]

    def test_first_pick_maximises_total_similarity(self):
        # Row 2 = {a,b} has the highest Jaccard sum to the rest, so greedy
        # should pick it first (it covers the broadest "concept space").
        sim = _jaccard_matrix([{"a"}, {"b"}, {"a", "b"}, {"c"}])
        order = _greedy_facility_location(sim, k=1)
        assert order == [2]

    def test_full_ranking_length(self):
        sim = _jaccard_matrix([{"a"}, {"b"}, {"c"}, {"d"}])
        order = _greedy_facility_location(sim, k=4)
        assert len(order) == 4

    def test_k_larger_than_n_caps_at_n(self):
        sim = _jaccard_matrix([{"a"}, {"b"}])
        order = _greedy_facility_location(sim, k=10)
        assert sorted(order) == [0, 1]

    def test_deterministic_on_ties(self):
        # All-disjoint singletons → all marginal gains identical at every
        # iteration. argmax tiebreak picks the lowest index.
        sim = _jaccard_matrix([{"a"}, {"b"}, {"c"}])
        order = _greedy_facility_location(sim, k=3)
        assert order == [0, 1, 2]


class TestSkillCoverageRanking:
    def test_compute_ranking_returns_full_permutation(self):
        table = _skills_table([
            {"algebra": ["a"]},
            {"algebra": ["b"]},
            {"algebra": ["a", "b"]},
            {"geometry": ["c"]},
        ])
        rank = compute_ranking(table)
        assert sorted(rank) == [0, 1, 2, 3]

    def test_load_or_compute_caches_to_parquet(self, tmp_path):
        table = _skills_table([
            {"algebra": ["a"]},
            {"algebra": ["b"]},
            {"algebra": ["a", "b"]},
        ])
        cache = tmp_path / "rank.parquet"
        rank = load_or_compute_ranking(table, cache_path=str(cache))
        assert cache.exists()
        cached_table = pq.read_table(cache)
        assert cached_table.column_names == ["selection_order"]
        assert [int(x) for x in cached_table.column("selection_order").to_pylist()] == rank

    def test_load_or_compute_uses_cache_when_aligned(self, tmp_path):
        table = _skills_table([
            {"algebra": ["a"]},
            {"algebra": ["b"]},
        ])
        cache = tmp_path / "rank.parquet"
        # Seed the cache with a hand-written ranking unrelated to the actual
        # greedy result. load_or_compute should trust the cache when row count
        # matches.
        pq.write_table(pa.table({"selection_order": [1, 0]}), cache)
        rank = load_or_compute_ranking(table, cache_path=str(cache))
        assert rank == [1, 0]

    def test_load_or_compute_recomputes_on_row_mismatch(self, tmp_path):
        table = _skills_table([
            {"algebra": ["a"]},
            {"algebra": ["b"]},
            {"algebra": ["c"]},
        ])
        cache = tmp_path / "rank.parquet"
        pq.write_table(pa.table({"selection_order": [0, 1]}), cache)  # 2 rows
        rank = load_or_compute_ranking(table, cache_path=str(cache))
        assert sorted(rank) == [0, 1, 2]


from s1.pruning.strategies import (
    _BASE_NLL_PATH,
    _BASE_NLL_RESPONSE_ONLY_PATH,
    score_ifd,
    score_skill_coverage,
)


def _s1k_table(n: int) -> pa.Table:
    return pa.table({
        "question": [f"Q{i}" for i in range(n)],
        "solution": [f"S{i}" for i in range(n)],
        "thinking_trajectories": [[f"T{i}"] for i in range(n)],
    })


class TestStrategyIfd:
    def _seed_nll(self, monkeypatch, tmp_path, cond_means, uncond_means):
        cond = tmp_path / "cond.parquet"
        uncond = tmp_path / "uncond.parquet"
        n = len(cond_means)
        pq.write_table(pa.table({
            "index": list(range(n)),
            "total_nll": [m * 10 for m in cond_means],
            "mean_nll": cond_means,
            "response_tokens": [10] * n,
        }), cond)
        pq.write_table(pa.table({
            "index": list(range(n)),
            "total_nll": [m * 10 for m in uncond_means],
            "mean_nll": uncond_means,
            "response_tokens": [10] * n,
        }), uncond)
        monkeypatch.setattr("s1.pruning.strategies._BASE_NLL_PATH", str(cond))
        monkeypatch.setattr(
            "s1.pruning.strategies._BASE_NLL_RESPONSE_ONLY_PATH", str(uncond)
        )

    def test_score_is_mean_nll_difference(self, monkeypatch, tmp_path):
        self._seed_nll(monkeypatch, tmp_path, [2.0, 3.0, 1.5], [1.0, 3.5, 1.5])
        scores = score_ifd(_s1k_table(3), None)
        # log(IFD) = mean_nll(A|Q) - mean_nll(A)
        assert scores == [1.0, -0.5, 0.0]

    def test_higher_score_means_question_helps_less(self, monkeypatch, tmp_path):
        # Row 0: Q strongly helps (cond << uncond) → low IFD → low score
        # Row 1: Q barely helps (cond ≈ uncond) → high IFD ≈ 1 → high score
        self._seed_nll(monkeypatch, tmp_path, [0.5, 2.0], [3.0, 2.1])
        scores = score_ifd(_s1k_table(2), None)
        assert scores[1] > scores[0]

    def test_raises_when_response_only_missing(self, monkeypatch, tmp_path):
        cond = tmp_path / "cond.parquet"
        pq.write_table(pa.table({
            "index": [0],
            "total_nll": [1.0],
            "mean_nll": [0.1],
            "response_tokens": [10],
        }), cond)
        monkeypatch.setattr("s1.pruning.strategies._BASE_NLL_PATH", str(cond))
        monkeypatch.setattr(
            "s1.pruning.strategies._BASE_NLL_RESPONSE_ONLY_PATH",
            str(tmp_path / "missing.parquet"),
        )
        import pytest
        with pytest.raises(FileNotFoundError, match="response_only"):
            score_ifd(_s1k_table(1), None)


class TestStrategySkillCoverage:
    def _patch_cache(self, monkeypatch, tmp_path):
        cache = tmp_path / "rank.parquet"
        monkeypatch.setattr(
            "s1.pruning.skill_coverage._DEFAULT_RANK_PATH", str(cache)
        )
        return cache

    def test_score_shape_matches_s1k_rows(self, monkeypatch, tmp_path):
        self._patch_cache(monkeypatch, tmp_path)
        skills = _skills_table([{"algebra": ["a"]}, {"geometry": ["b"]}, {}])
        s1k = _s1k_table(3)
        scores = score_skill_coverage(s1k, skills)
        assert len(scores) == 3

    def test_first_picked_row_has_highest_score(self, monkeypatch, tmp_path):
        cache = self._patch_cache(monkeypatch, tmp_path)
        # Seed a known ranking; the strategy should map selection_order → score.
        pq.write_table(pa.table({"selection_order": [2, 0, 1]}), cache)
        skills = _skills_table([{"algebra": ["a"]}, {"algebra": ["b"]}, {"algebra": ["c"]}])
        scores = score_skill_coverage(_s1k_table(3), skills)
        # Row 2 picked first → highest score; row 1 picked last → lowest.
        assert scores[2] > scores[0] > scores[1]

    def test_raises_when_skills_table_none(self, monkeypatch, tmp_path):
        self._patch_cache(monkeypatch, tmp_path)
        import pytest
        with pytest.raises(ValueError, match="skills"):
            score_skill_coverage(_s1k_table(3), None)

    def test_raises_on_row_mismatch(self, monkeypatch, tmp_path):
        cache = self._patch_cache(monkeypatch, tmp_path)
        # Cache built for 2 rows but s1k has 3 — should error after recompute
        # produces 2 entries against a 3-row s1k_table.
        pq.write_table(pa.table({"selection_order": [1, 0]}), cache)
        skills = _skills_table([{"algebra": ["a"]}, {"algebra": ["b"]}])
        import pytest
        with pytest.raises(ValueError, match="skill_coverage"):
            score_skill_coverage(_s1k_table(3), skills)
