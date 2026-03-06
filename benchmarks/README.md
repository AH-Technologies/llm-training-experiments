# Olivia HPC Benchmarking Suite

Benchmark suite for measuring model training capacity and throughput on the Olivia HPC cluster.

## Goals

1. **Maximum model size** that fits in GPU memory (SFT vs GRPO)
2. **Optimal batch size** for best throughput
3. **Realistic throughput** (trained tokens/sec, not just processed)

## Hardware

Olivia HPC `accel` partition:
- **76 nodes** × 4 GH200 GPUs = 304 total GPUs
- **Per GPU**: 96GB usable HBM3 memory (GH200 superchip)
- **Per node**: 288 CPUs, ~808GB RAM

## Quick Start

### 1. Download Benchmark Data (Run Once)

```bash
sbatch benchmarks/scripts/download_benchmark_data.slurm
```

This downloads [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) for SFT benchmarking and analyzes token lengths for accurate throughput measurement.

### 2. Run Benchmarks

The benchmark suite is split by strategy and scale:

**FSDP (small models, 1 node):**
```bash
sbatch benchmarks/scripts/sft_fsdp_singlenode.slurm    # SFT: 0.5B, 1.5B, 3B, 7B
sbatch benchmarks/scripts/grpo_fsdp_singlenode.slurm   # GRPO: 0.5B, 1.5B, 3B
```

**Megatron (mid/large models):**
```bash
# Single node (4 GPUs)
sbatch benchmarks/scripts/sft_megatron_singlenode.slurm    # SFT: 14B, 32B
sbatch benchmarks/scripts/grpo_megatron_singlenode.slurm   # GRPO: 7B, 14B, 32B

# 2 nodes (8 GPUs)
sbatch benchmarks/scripts/sft_megatron_2node.slurm         # SFT: 32B
sbatch benchmarks/scripts/grpo_megatron_2node.slurm        # GRPO: 32B

# 4 nodes (16 GPUs)
sbatch benchmarks/scripts/sft_megatron_72b_4node.slurm     # SFT: 72B
sbatch benchmarks/scripts/grpo_megatron_4node.slurm        # GRPO: 72B
```

## Strategies

| Strategy | Backend | Used For |
|----------|---------|----------|
| `auto` (default) | torchtune FSDP2+TP (SFT), FSDP (GRPO) | Small models (≤7B SFT, ≤3B GRPO) |
| `megatron` | Megatron-LM TP+PP via verl | Mid/large models (14B+ SFT, 7B+ GRPO) |
| `deepspeed` | TRL ZeRO-3 + CPU offload | Legacy fallback |

## Parallelism Configs

### Megatron SFT (TP+PP)
| Model | 4 GPUs (1 node) | 8 GPUs (2 nodes) | 16 GPUs (4 nodes) |
|-------|-----------------|-------------------|---------------------|
| 14B   | TP=2, DP=2      | TP=4, DP=2        | —                   |
| 32B   | TP=4            | TP=4, DP=2        | —                   |
| 72B   | —               | TP=4, PP=2        | TP=4, PP=4          |

### Megatron GRPO (TP+PP)
| Model | 4 GPUs (1 node) | 8 GPUs (2 nodes) | 16 GPUs (4 nodes) |
|-------|-----------------|-------------------|---------------------|
| 7B    | TP=2, DP=2      | TP=4, DP=2        | —                   |
| 14B   | TP=4            | TP=4, DP=2        | —                   |
| 32B   | TP=4            | TP=4, DP=2        | —                   |
| 72B   | —               | TP=4, PP=2        | TP=4, PP=2, DP=2    |

## Datasets

| Method | Dataset | Description |
|--------|---------|-------------|
| **SFT** | OpenMathInstruct-2 | 10k math problems with solutions |
| **GRPO** | MATH500 / pi1 | Math problems with reward functions |

Both are math-focused for fair comparison of training methods.

## Throughput Measurement

### Why "Trained Tokens" Matters

**SFT**: Only **response tokens** contribute to the loss:
- Prompt tokens: forward pass only (not trained)
- Response tokens: forward + backward (actually trained)

```
trained_tokens = batch_size × avg_response_length

Example (OpenMathInstruct-2):
- Avg prompt: ~64 tokens
- Avg response: ~308 tokens
- Response ratio: ~83%

If processed_tokens/sec = 10,000
Then trained_tokens/sec = 8,300 (the realistic number)
```

**GRPO**: Generates `n` responses per prompt, all used for policy gradient:
- Each prompt generates `rollout_n` samples (default: 8)
- All generated tokens are trained (used in reward + policy update)

```
trained_tokens = batch_size × rollout_n × avg_response_length

Example:
- Batch size: 4
- Rollout n: 8
- Avg response: ~300 tokens
- Tokens per step: 4 × 8 × 300 = 9,600 tokens
```

## Output Metrics

| Metric | Description |
|--------|-------------|
| `tokens_per_second` | **Trained** tokens/sec (response tokens only) |
| `processed_tokens_per_second` | Total tokens/sec (prompt + response) |
| `samples_per_second` | Batch processing rate |
| `time_per_step_ms` | Average step duration |
| `response_ratio` | Fraction of tokens that are trained |

## Command Line Options

```
python -m benchmarks.benchmark_runner [OPTIONS]

Options:
  --models              Model sizes: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B, or "all"
  --training-types      Training methods: sft, grpo, or "all"
  --num-gpus            GPUs per node (default: 4)
  --num-nodes           Number of nodes (default: 1)
  --sequence-length     Token sequence length (default: 2048)
  --num-steps           Training steps per benchmark (default: 50)
  --strategy            auto, torchtune, megatron, or deepspeed (default: auto)
  --find-max-batch      Find max batch size via binary search
  --output-dir          Results directory (default: benchmarks/results)

GRPO-specific options:
  --grpo-test-both-offload  Test both with and without CPU offloading
  --no-offload              Disable CPU offloading for actor model
  --no-ref-offload          Disable CPU offloading for reference model
```

## CPU Offloading (GRPO)

GRPO requires both an actor model (trainable) and a reference model (frozen for KL). CPU offloading trades speed for memory:

| Config | Description | Memory | Speed |
|--------|-------------|--------|-------|
| `ref` (default) | Ref model on CPU, actor on GPU | Balanced | Good |
| `actor+ref` | Both offloaded to CPU | Minimal | Slower |
| `none` | Both on GPU | Maximum | Fastest |

**Recommendation**: Start with default (`ref` offload). Use `--grpo-test-both-offload` to compare.

## Files

```
benchmarks/
├── benchmark_runner.py          # Main orchestrator
├── sft_benchmark.py             # SFT (torchtune FSDP2 + Megatron backends)
├── sft_torchtune_worker.py      # torchtune FSDP2+TP worker
├── grpo_benchmark.py            # GRPO via verl main_ppo
├── gpu_monitor.py               # Peak GPU memory tracking
├── simple_reward.py             # Dummy reward function for GRPO benchmarks
├── estimate_memory_v2.py        # Memory estimation formulas & plots
├── estimate_nodes_v2.py         # Minimum node estimation & plots
├── prepare_benchmark_data.py    # Download & analyze datasets
├── MEMORY_ESTIMATION_METHOD.md  # Detailed methodology documentation
├── configs/
│   └── ds_zero3_offload.json    # DeepSpeed ZeRO-3 config
├── scripts/
│   ├── download_benchmark_data.slurm
│   ├── sft_fsdp_singlenode.slurm        # FSDP SFT: 0.5B–7B, 1 node
│   ├── grpo_fsdp_singlenode.slurm       # FSDP GRPO: 0.5B–3B, 1 node
│   ├── sft_megatron_singlenode.slurm    # Megatron SFT: 14B, 32B, 1 node
│   ├── sft_megatron_2node.slurm         # Megatron SFT: 32B, 2 nodes
│   ├── sft_megatron_72b_4node.slurm     # Megatron SFT: 72B, 4 nodes
│   ├── grpo_megatron_singlenode.slurm   # Megatron GRPO: 7B–32B, 1 node
│   ├── grpo_megatron_2node.slurm        # Megatron GRPO: 32B, 2 nodes
│   └── grpo_megatron_4node.slurm        # Megatron GRPO: 72B, 4 nodes
└── results/                     # CSV/JSON output
```

## Memory & Node Estimation

Analytical memory estimators for planning training runs without GPU access:

```bash
# Estimate memory for N GPUs (generates SFT + GRPO plots)
srun ... python -m benchmarks.estimate_memory_v2 --num-gpus 16

# Find minimum nodes needed for each model size
srun ... python -m benchmarks.estimate_nodes_v2
```

Key features:
- **Megatron TP+PP**: TP capped at gpus_per_node (4), PP searched automatically across nodes
- **Three SFT offload modes**: No offload, optimizer offload, full offload
- **GRPO phase-based**: Estimates rollout, ref, actor, and weight sync phases
- **CPU memory tracking**: Per-node CPU with OS baseline, loading peak, process overhead

See [MEMORY_ESTIMATION_METHOD.md](MEMORY_ESTIMATION_METHOD.md) for detailed formulas.
