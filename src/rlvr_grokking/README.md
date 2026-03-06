# rlvr_grokking

Core package for GRPO training experiments on math reasoning tasks.

## rewards/

Reward functions for verl's `RewardManager`. Both implement the `compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float` interface.

- **`deepscaler_reward.py`** — Port of the [One-Shot-RLVR](https://github.com/ypwang61/One-Shot-RLVR) grading logic. Uses Dan Hendrycks' MATH normalization + SymPy symbolic comparison. This is the primary reward function.
- **`verl_reward.py`** — Simpler math reward with `\boxed{}` extraction, LaTeX normalization, and numeric comparison. Includes variants with format bonuses.

## ber/

Bidirectional Experience Replay — addresses the homogeneous rollout problem in 1-shot RLVR where all `n` responses to a prompt are either all-correct or all-incorrect, producing zero GRPO advantage.

BER classifies each rollout group into three phases:
1. **All-incorrect** — injects a cached correct response as a positive signal
2. **Mixed** — standard GRPO; caches the latest incorrect rollout
3. **All-correct** — injects a cached incorrect response as a negative signal

Files:
- `ber_module.py` — Cache (ring buffer) and inject logic
- `ber_trainer.py` — `BERRayPPOTrainer` subclass of verl's `RayPPOTrainer`
- `main_ber.py` — Entry point (`python -m src.rlvr_grokking.ber.main_ber`)

Run via: `scripts/run_grpo_ber.sh`
