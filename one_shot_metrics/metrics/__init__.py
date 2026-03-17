"""Modular metric registry for one-shot RLVR example evaluation.

Each metric is a callable: (ExampleRollouts) -> dict[str, float]
Metrics return a dict so a single metric file can produce multiple related values
(e.g., answer_entropy might also return unique_answer_count).

To add a new metric:
1. Create a file in this directory (e.g., my_metric.py)
2. Define a function decorated with @register_metric("my_metric")
3. The function takes an ExampleRollouts and returns dict[str, float]
"""

from dataclasses import dataclass, field
from typing import Callable

_REGISTRY: dict[str, Callable] = {}


@dataclass
class ExampleRollouts:
    """Standardized rollout data for one example."""
    example_id: str
    prompt: str
    ground_truth: str
    completions: list[str]  # raw model outputs
    extracted_answers: list[str | None]  # parsed \boxed{} answers
    is_correct: list[bool]  # per-rollout correctness
    metadata: dict = field(default_factory=dict)  # extra info


def register_metric(name: str):
    """Decorator to register a metric function."""
    def decorator(fn: Callable[[ExampleRollouts], dict[str, float]]):
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_all_metrics() -> dict[str, Callable]:
    """Return all registered metrics."""
    return dict(_REGISTRY)


def compute_all_metrics(rollouts: ExampleRollouts) -> dict[str, float]:
    """Run all registered metrics on a single example's rollouts."""
    results = {}
    for name, fn in _REGISTRY.items():
        results.update(fn(rollouts))
    return results


# Auto-import all metric modules so they self-register
import importlib
import pkgutil
import pathlib

_pkg_dir = pathlib.Path(__file__).parent
for _info in pkgutil.iter_modules([str(_pkg_dir)]):
    importlib.import_module(f".{_info.name}", __package__)
