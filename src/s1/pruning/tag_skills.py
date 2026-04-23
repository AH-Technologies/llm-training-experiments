"""Skill identification for s1K pruning (Alibaba polymath-RL §4, App. C).

Paper-native: 5 math categories with Prealgebra+Algebra+IntermediateAlgebra
merged into `algebra` per §5 to eliminate overlaps. Uses the paper's
Table 8 prompt template verbatim, one call per (problem, category) pair.

Judge model is Gemini 2.5 Flash-Lite (paper uses Qwen2.5-72B); same
OpenAI-compatible API pattern as `src/self_teach/leakage_judge.py`.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from urllib.request import Request, urlopen

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


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_skills_response(raw: str) -> list[str]:
    """Extract the skill list from a judge response.

    Paper's format: `<answer>skill1 : skill2 : ...</answer>` or `<answer>None</answer>`.
    Returns a lowercased, whitespace-stripped list with empty/None tokens removed.
    """
    m = _ANSWER_RE.search(raw)
    if not m:
        return []
    inside = m.group(1).strip()
    if inside.lower() == "none":
        return []
    skills: list[str] = []
    for part in inside.split(":"):
        token = part.strip().lower()
        if not token or token == "none":
            continue
        skills.append(token)
    return skills


class SkillIdentifier:
    """Calls an OpenAI-compatible chat API once per (question, category) pair
    and returns parsed skill lists. Fail-open: API errors → empty skill list.
    """

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

    def _identify_single(self, question: str, category: str) -> list[str]:
        prompt = build_prompt(category=category, question=question)
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
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
            return parse_skills_response(content)
        except Exception:
            self._call_count += 1
            self._error_count += 1
            return []

    def submit_batch(self, calls: list[tuple[str, str]]) -> None:
        """Submit (question, category) pairs asynchronously."""
        self._futures = [
            self.executor.submit(self._identify_single, q, c) for q, c in calls
        ]

    def collect_results(self) -> list[list[str]]:
        results = [f.result() for f in self._futures]
        if self._error_count > 0:
            print(
                f"[SkillIdentifier] WARNING: {self._error_count}/{self._call_count} "
                f"API calls failed (treated as empty skill list)"
            )
        return results
