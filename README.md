# LLM Training Experiments

LLM training experiments on the Olivia HPC cluster using verl for GRPO and SFT.

## Structure

```
├── benchmarks/          # HPC benchmarking for model sizes and throughput
│   ├── grpo_benchmark.py        # GRPO training benchmarks
│   ├── sft_benchmark.py         # SFT training benchmarks
│   ├── benchmark_runner.py      # Main runner with result formatting
│   └── scripts/                 # SLURM submission scripts
│
├── scripts/             # Training and setup scripts
│   ├── setup_env.sh             # Environment setup (run on GPU node)
│   ├── submit_training.slurm    # SLURM job submission
│   ├── prepare_pi13_data.py     # Data preparation for Pi13 dataset
│   ├── prepare_multi_prompt_val.py # Multi-prompt validation data prep
│   └── run_grpo_*.sh            # Various GRPO training configurations
│
└── src/
    ├── rlvr_grokking/           # Main package
    │   └── rewards/             # Reward functions for verl
    │       ├── verl_reward.py   # Math problem reward (compute_score)
    │       └── deepscaler_reward.py
    │
    └── astar_dataset/           # A* pathfinding dataset and training
        ├── generate_dataset.py  # Generate A* training data
        ├── custom_sft_trainer.py # SFT trainer using verl
        ├── reward.py            # A* specific rewards
        └── eval.py              # Evaluation utilities
```

## Setup

Must be run on a GPU node (ARM architecture):

```bash
srun --account=nn12068k --time=00:30:00 --partition=accel \
     --gpus=1 --cpus-per-task=4 --mem-per-cpu=16G ./scripts/setup_env.sh
```

## Running Benchmarks

Single-node benchmark (tests 1B to 14B models):
```bash
sbatch benchmarks/scripts/submit_benchmark.slurm
```

Multi-node benchmark (tests 32B, 72B, 120B, and 235B models on 2 nodes):
```bash
sbatch benchmarks/scripts/submit_multinode.slurm
```

## Running Training

```bash
sbatch scripts/submit_training.slurm pi13_math500
```

### Multi-prompt validation

Evaluates each validation question with 3 prompting strategies and logs separate metrics in wandb:

- `math_no_cot` — raw question, no CoT instruction
- `math_train_cot` — training-style suffix appended to user message
- `math_qwen_cot` — Qwen system message CoT (matches `qwen25-math-cot` from One-Shot-RLVR eval)

```bash
MULTI_PROMPT_VAL=1 sbatch scripts/submit_training.slurm pi1_math500_v2
```

The multi-prompt parquet is auto-generated on first run. Wandb metrics are logged as `val-core/{math_no_cot,math_train_cot,math_qwen_cot}/reward/...`.

## Dependencies

- verl (GRPO/SFT training)
- vLLM 0.12.0 (rollout generation)
- PyTorch 2.9.0 with CUDA 12.8
- FlashAttention2
