"""Convert structured A* execution traces into model-facing text.

All algorithm-specific terminology is mapped to generic terms:
    g-value  → cost
    h-value  → est (estimated remaining)
    f-value  → total
    open list → candidates
    closed/visited → visited
    expansion → Expand

The algorithm name "A*" is NEVER mentioned in any model-facing text.
"""

from .config import DatasetConfig
from .grid import Grid

# Default config for standalone usage
_DEFAULT_CONFIG = DatasetConfig()


def format_grid_section(grid: Grid) -> str:
    """Format the grid display section of the response."""
    lines = [f"Grid ({grid.rows}x{grid.cols}):"]
    lines.append(grid.to_string())
    lines.append(f"Start: ({grid.start[0]},{grid.start[1]}) "
                 f"Goal: ({grid.goal[0]},{grid.goal[1]})")
    return "\n".join(lines)


def format_neighbor(neighbor: dict) -> str:
    """Format a single neighbor evaluation line."""
    r, c = neighbor["pos"]
    status = neighbor["status"]

    if status == "wall":
        return f"    ({r},{c}) — wall. Skip."
    elif status == "visited":
        return f"    ({r},{c}) — already visited. Skip."
    elif status == "added":
        g = neighbor["g"]
        h = neighbor["h"]
        f = neighbor["f"]
        return f"    ({r},{c}) — open, cost={g}, est={h}, total={f}. Added."
    elif status == "not_improved":
        return f"    ({r},{c}) — no improvement. Skip."
    else:
        return f"    ({r},{c}) — {status}."


def format_step(step: dict, top_k: int = 3) -> str:
    """Format a single search expansion step."""
    r, c = step["expanded"]
    g = step["g"]
    h = step["h"]
    f = step["f"]

    lines = [f"Step {step['step']}: Expand ({r},{c}) [cost={g}, est={h}, total={f}]"]

    if step.get("goal_reached"):
        lines.append("  Goal reached!")
        return "\n".join(lines)

    # Neighbors
    if step["neighbors"]:
        lines.append("  Neighbors:")
        for neighbor in step["neighbors"]:
            lines.append(format_neighbor(neighbor))

    # Top candidates
    candidates = step.get("top_candidates", [])
    if candidates:
        parts = []
        for cand in candidates[:top_k]:
            cr, cc = cand["pos"]
            parts.append(f"({cr},{cc}) total={cand['f']}")
        lines.append(f"  Top candidates: [{', '.join(parts)}]")

    return "\n".join(lines)


def format_trace(grid: Grid, astar_result: dict, config: DatasetConfig | None = None) -> str:
    """Format the full assistant response with grid, search trace, and answer.

    Args:
        grid: The grid world instance.
        astar_result: Result from solve_astar() with "path" and "trace" keys.
        config: Dataset configuration (uses defaults if None).

    Returns:
        Formatted assistant response string.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    sections = []

    # Grid section
    if config.show_grid_in_response:
        sections.append(format_grid_section(grid))

    # Search section
    sections.append("Searching for shortest path.")
    sections.append("")

    for step in astar_result["trace"]:
        sections.append(format_step(step, top_k=config.show_top_k_candidates))

    # Answer section
    path = astar_result["path"]
    path_str = " ".join(f"({r},{c})" for r, c in path)
    sections.append("")
    sections.append(f"<answer>\n{path_str}\n</answer>")

    return "\n".join(sections)


def format_user_prompt(grid: Grid) -> str:
    """Format the user message with the grid and task."""
    lines = [
        "Find the shortest path in this grid:",
        "",
        grid.to_string(),
        "",
        f"Start: ({grid.start[0]},{grid.start[1]})",
        f"Goal: ({grid.goal[0]},{grid.goal[1]})",
    ]
    return "\n".join(lines)


def format_messages(
    grid: Grid,
    astar_result: dict,
    config: DatasetConfig | None = None,
) -> list[dict[str, str]]:
    """Create the full ChatML messages list for a training example.

    Returns:
        List of dicts with "role" and "content" keys.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    return [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": format_user_prompt(grid)},
        {"role": "assistant", "content": format_trace(grid, astar_result, config)},
    ]
