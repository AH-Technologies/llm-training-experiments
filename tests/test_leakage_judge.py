"""Tests for leakage detection."""

import json
from unittest.mock import patch, MagicMock

from src.self_teach.leakage_judge import HeuristicLeakageDetector, LLMJudgeLeakageDetector


class TestHeuristicLeakageDetector:
    def setup_method(self):
        self.detector = HeuristicLeakageDetector()

    def test_detects_boxed_leak(self):
        self.detector.submit_batch(
            feedbacks=["The answer is \\boxed{42}."],
            ground_truths=["42"],
            questions=["What is 6*7?"],
        )
        assert self.detector.collect_results() == [True]

    def test_no_leak_clean_feedback(self):
        self.detector.submit_batch(
            feedbacks=["Check your multiplication in step 3."],
            ground_truths=["42"],
            questions=["What is 6*7?"],
        )
        assert self.detector.collect_results() == [False]

    def test_batch_mixed(self):
        self.detector.submit_batch(
            feedbacks=[
                "Good approach but \\boxed{7} is wrong.",
                "Try again with a different method.",
                "The result is \\boxed{42}.",
            ],
            ground_truths=["42", "42", "42"],
            questions=["Q1", "Q2", "Q3"],
        )
        assert self.detector.collect_results() == [True, False, True]

    def test_empty_batch(self):
        self.detector.submit_batch(
            feedbacks=[],
            ground_truths=[],
            questions=[],
        )
        assert self.detector.collect_results() == []


class TestLLMJudgeLeakageDetector:
    def setup_method(self):
        self.detector = LLMJudgeLeakageDetector(
            api_base="https://api.example.com/v1",
            api_key="test-key",
            model="test-model",
            max_workers=2,
            timeout=5.0,
        )

    def _mock_urlopen_response(self, content: str):
        """Create a mock response that returns the given content as the LLM judge verdict."""
        response_body = json.dumps({
            "choices": [{"message": {"content": content}}]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("src.self_teach.leakage_judge.urlopen")
    def test_detects_leak_true(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen_response("true")
        self.detector.submit_batch(
            feedbacks=["The answer is 42."],
            ground_truths=["42"],
            questions=["What is 6*7?"],
        )
        assert self.detector.collect_results() == [True]

    @patch("src.self_teach.leakage_judge.urlopen")
    def test_detects_no_leak_false(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen_response("false")
        self.detector.submit_batch(
            feedbacks=["Check step 3."],
            ground_truths=["42"],
            questions=["What is 6*7?"],
        )
        assert self.detector.collect_results() == [False]

    @patch("src.self_teach.leakage_judge.urlopen")
    def test_handles_whitespace_in_response(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen_response("  True \n")
        self.detector.submit_batch(
            feedbacks=["The answer is 42."],
            ground_truths=["42"],
            questions=["What is 6*7?"],
        )
        assert self.detector.collect_results() == [True]

    @patch("src.self_teach.leakage_judge.urlopen")
    def test_batch_multiple(self, mock_urlopen):
        responses = [
            self._mock_urlopen_response("true"),
            self._mock_urlopen_response("false"),
        ]
        mock_urlopen.side_effect = responses
        self.detector.submit_batch(
            feedbacks=["Leak here.", "No leak."],
            ground_truths=["42", "42"],
            questions=["Q1", "Q2"],
        )
        assert self.detector.collect_results() == [True, False]

    @patch("src.self_teach.leakage_judge.urlopen")
    def test_api_error_defaults_to_no_leak(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("API timeout")
        self.detector.submit_batch(
            feedbacks=["Some feedback."],
            ground_truths=["42"],
            questions=["Q1"],
        )
        # On API error, default to no leak (fail-open to avoid false penalties)
        assert self.detector.collect_results() == [False]

    def test_prompt_contains_question_and_ground_truth(self):
        """Verify the judge prompt includes all required context."""
        prompt = self.detector._build_judge_prompt(
            feedback="Check your work.",
            ground_truth="42",
            question="What is 6*7?",
        )
        assert "What is 6*7?" in prompt
        assert "42" in prompt
        assert "Check your work." in prompt
        assert "true or false" in prompt
