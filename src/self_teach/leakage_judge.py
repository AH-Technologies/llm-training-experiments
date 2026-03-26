"""Answer leakage detection for teacher feedback.

Two implementations:
- HeuristicLeakageDetector: checks for \\boxed{ in feedback text (original method).
- LLMJudgeLeakageDetector: calls an OpenAI-compatible API for semantic detection.

Both share a two-phase interface: submit_batch() (non-blocking) then
collect_results() (blocks until ready). This lets the trainer fire judge
calls during A2 generation and collect results before reward computation.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from urllib.request import urlopen, Request


class LeakageDetector(ABC):
    """Base class for answer leakage detection."""

    @abstractmethod
    def submit_batch(
        self,
        feedbacks: list[str],
        ground_truths: list[str],
        questions: list[str],
    ) -> None:
        """Submit a batch for detection (may be async)."""

    @abstractmethod
    def collect_results(self) -> list[bool]:
        """Collect results. Blocks until all results are ready.

        Returns list of bools — True means leak detected.
        """


class HeuristicLeakageDetector(LeakageDetector):
    """Original heuristic: checks for \\boxed{ inside feedback text."""

    def __init__(self):
        self._results: list[bool] = []

    def submit_batch(self, feedbacks, ground_truths, questions):
        self._results = ["\\boxed{" in feedback for feedback in feedbacks]

    def collect_results(self):
        return self._results


JUDGE_PROMPT_TEMPLATE = """\
You are evaluating whether a teacher's feedback leaks the final answer \
to a math problem.

<problem_statement>
{question}
</problem_statement>

<ground_truth_answer>
{ground_truth}
</ground_truth_answer>

<teacher_feedback>
{feedback}
</teacher_feedback>

Does the feedback reveal the final answer to the student?

Note: The ground truth value may appear in the feedback in a context \
unrelated to the answer (e.g., "step 2", "two variables", "consider \
the 2 cases"). This is NOT leakage. Leakage means the feedback \
explicitly states, reveals, or gives away the final answer value.

Respond with exactly: true or false"""


class LLMJudgeLeakageDetector(LeakageDetector):
    """Calls an OpenAI-compatible API via ThreadPoolExecutor."""

    def __init__(
        self,
        api_base: str,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash-lite",
        max_workers: int = 32,
        timeout: float = 30.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_JUDGE_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future] = []
        self._error_count: int = 0
        self._call_count: int = 0

    def _build_judge_prompt(self, feedback: str, ground_truth: str, question: str) -> str:
        return JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            feedback=feedback,
        )

    def _judge_single(self, feedback: str, ground_truth: str, question: str) -> bool:
        """Call the API for a single feedback. Returns True if leak detected."""
        prompt = self._build_judge_prompt(feedback, ground_truth, question)
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.0,
        }).encode()

        req = Request(
            f"{self.api_base}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
            content = body["choices"][0]["message"]["content"]
            self._call_count += 1
            return content.strip().lower() == "true"
        except Exception as e:
            # Fail-open: if API errors, don't penalize the teacher
            self._call_count += 1
            self._error_count += 1
            return False

    def submit_batch(self, feedbacks, ground_truths, questions):
        self._futures = [
            self.executor.submit(self._judge_single, f, gt, q)
            for f, gt, q in zip(feedbacks, ground_truths, questions)
        ]

    def collect_results(self) -> list[bool]:
        results = [f.result() for f in self._futures]
        if self._error_count > 0:
            print(
                f"[LLM Judge] WARNING: {self._error_count}/{self._call_count} "
                f"API calls failed (treated as no-leak)"
            )
        return results
