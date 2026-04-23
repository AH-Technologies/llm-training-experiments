"""Skill identification for s1K pruning (Alibaba polymath-RL §4, App. C).

Paper-native: 5 math categories with Prealgebra+Algebra+IntermediateAlgebra
merged into `algebra` per §5 to eliminate overlaps. Uses the paper's
Table 8 prompt template verbatim, one call per (problem, category) pair.

Judge model is Gemini 2.5 Flash (paper uses Qwen2.5-72B); same
OpenAI-compatible API pattern as `src/self_teach/leakage_judge.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import random as _random
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

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
        model: str = "gemini-2.5-flash",
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
            "max_tokens": 800,
            "temperature": 0.0,
            # Classification task — disable Gemini's hidden reasoning so the
            # `<answer>...</answer>` response is not truncated mid-token.
            # Harmless on non-Gemini models (unknown field is ignored).
            "reasoning_effort": "none",
        }).encode()
        # Retry with exponential backoff + jitter on 429s and 5xx. Gemini Flash
        # rate-limits bursts; this keeps the job moving instead of silently
        # dropping a call on the first 429.
        for attempt in range(5):
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
            except HTTPError as e:
                if e.code == 429 or 500 <= e.code < 600:
                    if attempt < 4:
                        time.sleep(2 ** attempt + _random.random())
                        continue
                break
            except Exception:
                break
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


def tag_dataset(input_path: str, output_path: str, identifier) -> None:
    """Tag every row of the input parquet with skills across all 5 categories.

    The `identifier` is any object exposing `submit_batch(calls)` and
    `collect_results()` — production callers pass a `SkillIdentifier`, tests
    pass a fake.
    """
    table = pq.read_table(input_path)
    questions = table.column("question").to_pylist()
    cot_type = table.column("cot_type").to_pylist()
    source_type = table.column("source_type").to_pylist()

    # Build (question, category) call list — row-major over rows, then categories
    calls = [(q, c) for q in questions for c in SKILL_CATEGORIES]
    identifier.submit_batch(calls)
    flat_results = identifier.collect_results()

    # Hard-abort on widespread API failure to prevent silent data corruption
    # (all-zero skill_counts would degrade skill_abundance_select to dataset order).
    error_count = getattr(identifier, "_error_count", 0)
    call_count = getattr(identifier, "_call_count", len(calls))
    if call_count > 0 and error_count > call_count // 2:
        raise RuntimeError(
            f"Skill tagging failed: {error_count}/{call_count} API calls errored. "
            "Check LLM_JUDGE_API_KEY and API connectivity before continuing."
        )

    n_cats = len(SKILL_CATEGORIES)
    per_row = [
        flat_results[i * n_cats : (i + 1) * n_cats] for i in range(len(questions))
    ]

    columns = {
        "index": list(range(len(questions))),
        "question": questions,
        "cot_type": cot_type,
        "source_type": source_type,
    }
    for ci, cat in enumerate(SKILL_CATEGORIES):
        columns[f"skills_{cat}"] = [row[ci] for row in per_row]
    columns["skill_count"] = [sum(len(lst) for lst in row) for row in per_row]

    pq.write_table(pa.table(columns), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag s1K problems with paper-native salient math skills."
    )
    parser.add_argument("--input", default="data/s1K/s1k.parquet")
    parser.add_argument("--output", default="data/s1K/s1k_skills.parquet")
    parser.add_argument(
        "--api-base",
        default=os.environ.get(
            "LLM_JUDGE_API_BASE",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
    )
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    identifier = SkillIdentifier(
        api_base=args.api_base,
        model=args.model,
        max_workers=args.workers,
        timeout=args.timeout,
    )
    tag_dataset(
        input_path=args.input,
        output_path=args.output,
        identifier=identifier,
    )
    print(f"Wrote tagged skills to {args.output}")


if __name__ == "__main__":
    main()
