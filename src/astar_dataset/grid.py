"""Grid world generation and A* solver with full execution trace logging."""

import heapq
import random
from dataclasses import dataclass, field


WALL = "#"
OPEN = "."
START = "S"
GOAL = "G"

# 4-directional movement: (delta_row, delta_col)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class Grid:
    """A rectangular grid world with obstacles, start, and goal."""

    cells: list[list[str]]
    start: tuple[int, int]
    goal: tuple[int, int]
    rows: int
    cols: int
    seed: int
    obstacle_ratio: float

    def is_valid(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_walkable(self, r: int, c: int) -> bool:
        return self.is_valid(r, c) and self.cells[r][c] != WALL

    def to_string(self) -> str:
        return "\n".join(" ".join(row) for row in self.cells)

    def get_neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        neighbors = []
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if self.is_valid(nr, nc):
                neighbors.append((nr, nc))
        return neighbors


@dataclass(order=True)
class AStarNode:
    """Priority queue node for A*."""

    f: int
    h: int  # tie-break: prefer lower h
    r: int  # tie-break: top-to-bottom
    c: int  # tie-break: left-to-right
    g: int = field(compare=False)


def manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    return abs(r1 - r2) + abs(c1 - c2)


def generate_grid(
    rows: int, cols: int, obstacle_ratio: float, seed: int
) -> Grid | None:
    """Generate a random grid world with a guaranteed valid path.

    Uses a local Random instance for reproducibility.
    Returns None if no valid path found after placement (caller should retry
    with different seed).
    """
    rng = random.Random(seed)

    cells = [[OPEN for _ in range(cols)] for _ in range(rows)]

    # Collect all positions, shuffle, then place obstacles
    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(all_positions)

    num_obstacles = int(rows * cols * obstacle_ratio)

    # Reserve two positions for start and goal
    start_pos = all_positions[0]
    goal_pos = all_positions[1]

    # Ensure start != goal (guaranteed since they're different indices)
    obstacle_candidates = all_positions[2:]
    for i in range(min(num_obstacles, len(obstacle_candidates))):
        r, c = obstacle_candidates[i]
        cells[r][c] = WALL

    cells[start_pos[0]][start_pos[1]] = START
    cells[goal_pos[0]][goal_pos[1]] = GOAL

    actual_walls = sum(
        1 for r in range(rows) for c in range(cols) if cells[r][c] == WALL
    )
    actual_ratio = actual_walls / (rows * cols)

    grid = Grid(
        cells=cells,
        start=start_pos,
        goal=goal_pos,
        rows=rows,
        cols=cols,
        seed=seed,
        obstacle_ratio=actual_ratio,
    )

    # Validate: must have a path
    result = solve_astar(grid)
    if result is None:
        return None

    return grid


def solve_astar(grid: Grid) -> dict | None:
    """Run A* search on the grid.

    Returns dict with:
        - "path": list of (row, col) tuples
        - "trace": list of step dicts (structured trace for logging)
        - "num_steps": number of expansion steps
    Returns None if no path exists.
    """
    sr, sc = grid.start
    gr, gc = grid.goal

    h_start = manhattan_distance(sr, sc, gr, gc)
    start_node = AStarNode(f=h_start, h=h_start, r=sr, c=sc, g=0)

    open_list: list[AStarNode] = [start_node]
    g_scores: dict[tuple[int, int], int] = {(sr, sc): 0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    trace: list[dict] = []
    step_num = 0

    while open_list:
        current = heapq.heappop(open_list)
        pos = (current.r, current.c)

        if pos in closed:
            continue

        closed.add(pos)
        step_num += 1

        # Record this expansion step
        step_record = {
            "step": step_num,
            "expanded": pos,
            "g": current.g,
            "h": current.h,
            "f": current.f,
            "neighbors": [],
            "top_candidates": [],
        }

        # Check if goal reached
        if pos == (gr, gc):
            step_record["goal_reached"] = True
            trace.append(step_record)

            # Reconstruct path
            path = []
            node = pos
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            path.reverse()

            return {
                "path": path,
                "trace": trace,
                "num_steps": step_num,
            }

        step_record["goal_reached"] = False

        # Expand neighbors
        for nr, nc in grid.get_neighbors(current.r, current.c):
            neighbor_info = {"pos": (nr, nc)}

            if grid.cells[nr][nc] == WALL:
                neighbor_info["status"] = "wall"
                step_record["neighbors"].append(neighbor_info)
                continue

            if (nr, nc) in closed:
                neighbor_info["status"] = "visited"
                step_record["neighbors"].append(neighbor_info)
                continue

            tentative_g = current.g + 1
            if tentative_g < g_scores.get((nr, nc), float("inf")):
                g_scores[(nr, nc)] = tentative_g
                came_from[(nr, nc)] = pos
                h = manhattan_distance(nr, nc, gr, gc)
                f = tentative_g + h
                heapq.heappush(
                    open_list, AStarNode(f=f, h=h, r=nr, c=nc, g=tentative_g)
                )
                neighbor_info["status"] = "added"
                neighbor_info["g"] = tentative_g
                neighbor_info["h"] = h
                neighbor_info["f"] = f
            else:
                neighbor_info["status"] = "not_improved"

            step_record["neighbors"].append(neighbor_info)

        # Record top-k candidates from open list (by f-value)
        # Peek without popping: sort a copy
        visible = []
        for node in open_list:
            if (node.r, node.c) not in closed:
                visible.append(node)
        visible.sort()
        for node in visible[:3]:
            step_record["top_candidates"].append(
                {"pos": (node.r, node.c), "f": node.f}
            )

        trace.append(step_record)

    return None  # No path found
