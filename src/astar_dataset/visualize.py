"""Optional grid visualization for debugging."""

from .grid import Grid


def render_grid(grid: Grid, path: list[tuple[int, int]] | None = None) -> str:
    """Render a grid as ASCII art, optionally with a path overlay.

    Path cells are marked with '*' (except start 'S' and goal 'G').
    """
    # Copy cells
    display = [row[:] for row in grid.cells]

    if path:
        path_set = set(path)
        for r, c in path_set:
            if (r, c) != grid.start and (r, c) != grid.goal:
                display[r][c] = "*"

    lines = []
    for r, row in enumerate(display):
        lines.append(" ".join(row))

    return "\n".join(lines)


def print_instance(grid: Grid, astar_result: dict) -> None:
    """Print a complete instance for debugging."""
    path = astar_result["path"]

    print(f"Grid ({grid.rows}x{grid.cols}), seed={grid.seed}, "
          f"obstacles={grid.obstacle_ratio:.0%}")
    print(f"Start: {grid.start}, Goal: {grid.goal}")
    print(f"Path length: {len(path) - 1} steps, "
          f"Search steps: {astar_result['num_steps']}")
    print()
    print(render_grid(grid, path))
    print()
    print("Path:", " → ".join(f"({r},{c})" for r, c in path))
