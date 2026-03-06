# LLM Training Experiments

LLM training experiments on NRIS HPC clusters using [verl](https://github.com/volcengine/verl) for GRPO training.

## Structure

```
├── benchmarks/              # HPC benchmarking (model sizes, throughput)
│
├── configs/                 # Hydra/YAML configs
│   └── self_teach_interaction.yaml
│
├── data/                    # Training and evaluation datasets
│
├── scripts/                     # All runnable scripts
│   ├── setup_env*.sh               # Env setup per cluster (Idun, Olivia)
│   ├── submit_training.slurm       # Main SLURM job submission
│   ├── run_grpo_*.sh               # GRPO launch configs (hyperparams, model, data)
│   ├── prepare_*.py                # Dataset preparation and formatting
│   └── eval_s1k.py                 # s1K evaluation script
│
├── src/
│   ├── rlvr_grokking/              # Core GRPO training package
│   │   ├── rewards/                # Reward functions (deepscaler, verl)
│   │   └── ber/                    # Bidirectional Experience Replay
│   │
│   ├── self_teach/                 # Student-teacher self-play GRPO
│   │   ├── main.py                 # Entry point
│   │   ├── trainer.py              # 2-phase trainer and task runner
│   │   ├── interaction.py          # 3-turn interaction for verl multi-turn
│   │   ├── prompts.py              # Prompt templates
│   │   └── rewards.py              # Self-teach reward logic
│   │
│   └── astar_dataset/             # A* pathfinding dataset and SFT training
│
└── docs/                          # Design docs
```

## Setup

Must be run on a GPU node (ARM on Olivia, x86 on Idun):

```bash
# Olivia
srun --account=nn12068k --time=00:30:00 --partition=accel \
     --gpus=1 --cpus-per-task=4 --mem-per-cpu=16G ./scripts/setup_env_olivia.sh

# Idun
srun --account=... --time=00:30:00 --partition=... \
     --gpus=1 --cpus-per-task=4 --mem-per-cpu=16G ./scripts/setup_env.sh
```

## Training

### Standard GRPO

```bash
sbatch scripts/submit_training.slurm <experiment_name>
```

### Self-Teach (student-teacher self-play)

3-turn self-play where a student answers, a teacher gives feedback, and the student revises. Both roles are trained via GRPO with separate reward signals.

```bash
sbatch scripts/submit_training.slurm self_teach_s1k_qwen3
```

### Benchmarks

```bash
sbatch benchmarks/scripts/submit_benchmark.slurm        # single-node
sbatch benchmarks/scripts/submit_multinode.slurm         # multi-node
```

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_env.sh` / `setup_env_olivia*.sh` | Build the Python venv on the target cluster's GPU nodes |
| `submit_training.slurm` | Main SLURM entrypoint — takes an experiment name that maps to a `run_grpo_*.sh` config |
| `run_grpo_*.sh` | Per-experiment launch scripts with model path, dataset, and hyperparameters |
| `run_grpo_self_teach.sh` | Launches self-teach 2-phase training via `src.self_teach.main` |
| `run_grpo_ber.sh` | Launches BER-enhanced GRPO via `src.rlvr_grokking.ber.main_ber` |
| `prepare_*.py` | Format raw datasets into verl-compatible parquet files |
| `eval_s1k.py` | Evaluate a checkpoint on the s1K test set |

## Dependencies

- verl (GRPO training framework)
- vLLM (rollout generation)
- PyTorch with CUDA
- FlashAttention2
