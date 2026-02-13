"""Central configuration for A* grokking dataset generation."""

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    # Grid parameters
    grid_sizes: list[tuple[int, int]] = field(
        default_factory=lambda: [(7, 7)]
    )
    obstacle_ratios: list[float] = field(default_factory=lambda: [0.20, 0.25, 0.30])
    movement: str = "4dir"

    # Dataset size
    num_instances_per_config: int = 1000
    train_fraction: float = 0.40
    val_fraction: float = 0.30
    test_fraction: float = 0.30

    # Trace formatting
    show_top_k_candidates: int = 3
    show_grid_in_response: bool = True

    # Generation
    seed: int = 42
    num_workers: int = 8
    max_retries_per_instance: int = 50

    # Token length
    max_token_length: int = 5120

    # Output
    output_dir: str = "data/astar_grokking_dataset"

    # System prompt — no algorithm names
    system_prompt: str = (
        "You are a pathfinding assistant. Given a grid world with obstacles, "
        "find the shortest path from Start to Goal. Search efficiently by "
        "always expanding the most promising position first. Show your search "
        "process step by step, then provide the final path."
    )
