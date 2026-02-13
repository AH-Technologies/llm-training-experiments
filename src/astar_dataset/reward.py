"""Reward and evaluation functions for A* search grokking experiments.

Provides path extraction, validation, metrics, and a verl-compatible
compute_score function.
"""

import json
import re

from .grid import Grid, WALL, manhattan_distance, DIRECTIONS


def extract_path(model_output: str) -> list[tuple[int, int]] | None:
    """Extract path from <answer>...</answer> tags in model output.

    Returns list of (row, col) tuples, or None if extraction fails.
    """
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", model_output, re.DOTALL)
    if not match:
        return None

    path_str = match.group(1).strip()
    coords = re.findall(r"\((\d+),\s*(\d+)\)", path_str)
    if not coords:
        return None

    return [(int(r), int(c)) for r, c in coords]


def parse_grid_string(grid_string: str) -> list[list[str]]:
    """Parse a grid string back into a 2D cell array."""
    rows = []
    for line in grid_string.strip().split("\n"):
        cells = line.split()
        if cells:
            rows.append(cells)
    return rows


def validate_path(
    path: list[tuple[int, int]],
    grid_string: str,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> dict:
    """Validate a path against a grid.

    Returns dict with:
        - valid: bool
        - errors: list of error strings
    """
    errors = []
    cells = parse_grid_string(grid_string)
    rows = len(cells)
    cols = len(cells[0]) if cells else 0

    if not path:
        return {"valid": False, "errors": ["Empty path"]}

    # Check start
    if path[0] != start:
        errors.append(
            f"Path starts at {path[0]}, expected {start}"
        )

    # Check goal
    if path[-1] != goal:
        errors.append(
            f"Path ends at {path[-1]}, expected {goal}"
        )

    # Check each step
    for i, (r, c) in enumerate(path):
        # Bounds check
        if not (0 <= r < rows and 0 <= c < cols):
            errors.append(f"Step {i}: ({r},{c}) out of bounds ({rows}x{cols})")
            continue

        # Wall check (start/goal cells won't be walls in valid grids)
        if cells[r][c] == WALL:
            errors.append(f"Step {i}: ({r},{c}) is a wall")

    # Check connectivity (each step is exactly 1 move apart)
    for i in range(1, len(path)):
        r1, c1 = path[i - 1]
        r2, c2 = path[i]
        dist = abs(r1 - r2) + abs(c1 - c2)
        if dist != 1:
            errors.append(
                f"Step {i}: ({r1},{c1})→({r2},{c2}) distance={dist}, expected 1"
            )

    return {"valid": len(errors) == 0, "errors": errors}


def compute_reward(
    model_output: str,
    grid_string: str,
    start: tuple[int, int],
    goal: tuple[int, int],
    optimal_length: int,
) -> float:
    """Compute scalar reward for a model output.

    Reward tiers:
        1.0 — valid path AND optimal length
        0.5 — valid path but suboptimal
        0.1 — format valid (answer tags parseable) but path invalid
        0.0 — cannot parse output at all
    """
    path = extract_path(model_output)
    if path is None:
        return 0.0

    result = validate_path(path, grid_string, start, goal)
    if not result["valid"]:
        return 0.1

    path_length = len(path) - 1  # steps = nodes - 1
    if path_length <= optimal_length:
        return 1.0

    return 0.5


def compute_metrics(
    model_output: str,
    grid_string: str,
    start: tuple[int, int],
    goal: tuple[int, int],
    optimal_path: str,
    optimal_path_length: int,
) -> dict:
    """Compute all evaluation metrics for a single instance.

    Returns dict with:
        - format_valid: bool
        - path_valid: bool
        - path_optimal: bool
        - path_length_ratio: float (1.0 = optimal, <1.0 = suboptimal)
        - exact_match: bool
        - reward: float
    """
    path = extract_path(model_output)

    if path is None:
        return {
            "format_valid": False,
            "path_valid": False,
            "path_optimal": False,
            "path_length_ratio": 0.0,
            "exact_match": False,
            "reward": 0.0,
        }

    validation = validate_path(path, grid_string, start, goal)
    path_valid = validation["valid"]

    if not path_valid:
        return {
            "format_valid": True,
            "path_valid": False,
            "path_optimal": False,
            "path_length_ratio": 0.0,
            "exact_match": False,
            "reward": 0.1,
        }

    path_length = len(path) - 1
    is_optimal = path_length <= optimal_path_length
    length_ratio = optimal_path_length / path_length if path_length > 0 else 0.0

    # Check exact match against ground truth path
    gt_coords = re.findall(r"\((\d+),\s*(\d+)\)", optimal_path)
    gt_path = [(int(r), int(c)) for r, c in gt_coords]
    exact = path == gt_path

    return {
        "format_valid": True,
        "path_valid": True,
        "path_optimal": is_optimal,
        "path_length_ratio": min(length_ratio, 1.0),
        "exact_match": exact,
        "reward": 1.0 if is_optimal else 0.5,
    }


def evaluate_dataset(instances: list[dict]) -> dict:
    """Evaluate a batch of model outputs against ground truth.

    Each instance dict should have: model_output, grid_string, start, goal,
    optimal_path, optimal_path_length.

    Returns aggregate metrics.
    """
    results = []
    for inst in instances:
        start = inst["start"]
        goal = inst["goal"]
        if isinstance(start, str):
            m = re.match(r"\((\d+),\s*(\d+)\)", start)
            start = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        if isinstance(goal, str):
            m = re.match(r"\((\d+),\s*(\d+)\)", goal)
            goal = (int(m.group(1)), int(m.group(2))) if m else (0, 0)

        metrics = compute_metrics(
            model_output=inst["model_output"],
            grid_string=inst["grid_string"],
            start=start,
            goal=goal,
            optimal_path=inst["optimal_path"],
            optimal_path_length=inst["optimal_path_length"],
        )
        results.append(metrics)

    n = len(results)
    if n == 0:
        return {}

    return {
        "format_valid_rate": sum(r["format_valid"] for r in results) / n,
        "path_valid_rate": sum(r["path_valid"] for r in results) / n,
        "path_optimal_rate": sum(r["path_optimal"] for r in results) / n,
        "exact_match_rate": sum(r["exact_match"] for r in results) / n,
        "mean_path_length_ratio": sum(r["path_length_ratio"] for r in results) / n,
        "mean_reward": sum(r["reward"] for r in results) / n,
        "num_instances": n,
        "results_per_instance": results,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict = None,
) -> float:
    """verl-compatible reward function.

    Args:
        data_source: Should be "astar" or "astar_grid".
        solution_str: Model's full response text.
        ground_truth: The optimal path length (int or str).
        extra_info: Dict with "grid_string", "start", "goal" keys.

    Returns:
        Float reward: 0.0, 0.1, 0.5, or 1.0
    """
    if extra_info is None:
        return 0.0

    grid_string = extra_info.get("grid_string", "")
    start = extra_info.get("start", (0, 0))
    goal = extra_info.get("goal", (0, 0))

    # Parse string coordinates if needed
    if isinstance(start, str):
        m = re.match(r"\((\d+),\s*(\d+)\)", start)
        start = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    if isinstance(goal, str):
        m = re.match(r"\((\d+),\s*(\d+)\)", goal)
        goal = (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    optimal_length = int(ground_truth) if ground_truth else 0

    return compute_reward(solution_str, grid_string, start, goal, optimal_length)
