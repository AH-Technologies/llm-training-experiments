# Olivia HPC Benchmarking Suite

Benchmark suite for measuring model training capacity and throughput on the Olivia HPC cluster.

## Goals

1. **Maximum model size** that fits in GPU memory (SFT vs GRPO)
2. **Optimal batch size** for best throughput
3. **Realistic throughput** (trained tokens/sec, not just processed)

## Hardware

Olivia HPC `accel` partition:
- **76 nodes** × 4 GH200 GPUs = 304 total GPUs
- **Per GPU**: 120GB HBM3 memory
- **Per node**: 288 CPUs, ~808GB RAM

## Quick Start

### 1. Download Benchmark Data (Run Once)

```bash
sbatch benchmarks/scripts/download_benchmark_data.slurm
```

This downloads [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) for SFT benchmarking and analyzes token lengths for accurate throughput measurement.

### 2. Run Benchmarks

**Test single model:**
```bash
sbatch benchmarks/scripts/submit_benchmark.slurm --models 1.5B --training-types sft
```

**Find optimal batch size:**
```bash
sbatch benchmarks/scripts/submit_benchmark.slurm --models 7B --training-types sft grpo --find-optimal
```

**Full benchmark suite:**
```bash
sbatch benchmarks/scripts/submit_full_benchmark.slurm
```

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
  --sequence-length     Token sequence length (default: 2048)
  --num-steps           Training steps per benchmark (default: 50)
  --find-optimal        Find max batch size, then test range for optimal throughput
  --find-max-batch      Only find max batch size (no throughput sweep)
  --output-dir          Results directory (default: benchmarks/results)

GRPO-specific options:
  --grpo-test-both-offload  Test both with and without CPU offloading
  --actor-offload           Enable CPU offloading for actor model
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

## Expected Results

Example output for supervisor's report:

```
Hardware: 4x GH200 120GB (1 node)
Dataset: OpenMathInstruct-2 (SFT), MATH500 (GRPO)
Sequence length: 2048

=== Maximum Model Size ===
| Method | Max Model | Notes |
|--------|-----------|-------|
| SFT    | 32B       | Single node |
| GRPO   | 14B       | Actor + Ref + vLLM overhead |

=== Optimal Throughput (7B model) ===
| Method | Optimal Batch | Trained Tok/s |
|--------|---------------|---------------|
| SFT    | 32            | 12,500        |
| GRPO   | 8             | 3,200         |

=== Scaling ===
| Model | SFT Tok/s | GRPO Tok/s | GRPO/SFT |
|-------|-----------|------------|----------|
| 1.5B  | 35,000    | 8,500      | 0.24x    |
| 7B    | 12,500    | 3,200      | 0.26x    |
| 14B   | 6,800     | 1,500      | 0.22x    |
```

## Files

```
benchmarks/
├── benchmark_runner.py          # Main orchestrator
├── sft_benchmark.py             # SFT via verl fsdp_sft_trainer
├── grpo_benchmark.py            # GRPO via verl main_ppo
├── prepare_benchmark_data.py    # Download & analyze datasets
├── scripts/
│   ├── download_benchmark_data.slurm
│   ├── submit_benchmark.slurm
│   ├── submit_full_benchmark.slurm
│   └── submit_multinode.slurm
└── results/                     # CSV/JSON output
```
