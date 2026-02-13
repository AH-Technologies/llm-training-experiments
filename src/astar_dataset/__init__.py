"""A* Search Grokking Dataset — grid generation, trace formatting, and evaluation."""

__all__ = [
    "DatasetConfig",
    "Grid",
    "generate_grid",
    "solve_astar",
    "format_trace",
    "format_messages",
    "extract_path",
    "validate_path",
    "compute_reward",
    "compute_score",
]

from .config import DatasetConfig
from .grid import Grid, generate_grid, solve_astar
from .trace_formatter import format_messages, format_trace
from .reward import compute_reward, compute_score, extract_path, validate_path
