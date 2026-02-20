#!/usr/bin/env python3
"""
SFT benchmark module using TRL's SFTTrainer with FSDP sharding.

Runs SFT training via torchrun + TRL to measure throughput and memory usage.
Uses OpenMathInstruct-2 dataset for realistic throughput measurements.

Throughput is calculated as TRAINED tokens/sec (response tokens only),
not total processed tokens, for accurate comparison.

Metrics extraction:
1. Worker JSON file (metrics written by sft_trl_worker.py)
2. Wall-clock estimation (fallback)
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from benchmarks.gpu_monitor import GPUMemoryMonitor


# Default dataset paths (relative to project root)
DEFAULT_TRAIN_FILE = "data/benchmark/openmath_sft_train.parquet"
DEFAULT_STATS_FILE = "data/benchmark/openmath_sft_stats.json"


def load_dataset_stats(project_dir: Path) -> dict:
    """Load dataset statistics for accurate throughput calculation."""
    stats_path = project_dir / DEFAULT_STATS_FILE
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return None


def run_sft_benchmark(
    model_name: str,
    batch_size: int,
    num_gpus: int,
    sequence_length: int,
    num_steps: int,
    gradient_accumulation: int = 1,
    train_file: str = None,
    val_file: str = None,
    num_nodes: int = 1,
    offload: bool = False,
) -> dict:
    """
    Run SFT benchmark via TRL's SFTTrainer with FSDP and return metrics.

    Uses real math datasets for realistic throughput measurements.

    Returns dict with:
        - peak_memory_gb: Peak GPU memory usage
        - tokens_per_second: Training throughput (actual trained tokens)
        - samples_per_second: Samples processed per second
        - time_per_step_ms: Average time per training step
        - total_time_s: Total benchmark time
        - metrics_source: "trl_trainer" or "estimated"
    """

    project_dir = Path(__file__).parent.parent
    worker_script = Path(__file__).parent / "sft_trl_worker.py"

    # Use real datasets
    train_file = train_file or str(project_dir / DEFAULT_TRAIN_FILE)

    # Calculate micro batch size (account for all GPUs across nodes)
    # Cap at 8 to avoid activation memory OOM — excess is handled via gradient accumulation
    total_gpus = num_gpus * num_nodes
    micro_batch_size = min(8, max(1, batch_size // (total_gpus * gradient_accumulation)))

    # Get multi-node settings from environment (set by SLURM)
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29502")

    # Build torchrun command
    if num_nodes > 1:
        # Multi-node: use srun to launch torchrun on each allocated node
        cmd = [
            "srun",
            f"--nodes={num_nodes}",
            "--ntasks-per-node=1",
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            f"--nnodes={num_nodes}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:{master_port}",
            str(worker_script),
        ]
    else:
        # Single-node: direct torchrun
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            f"--nnodes=1",
            "--node_rank=0",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
            str(worker_script),
        ]

    # Use shared filesystem for temp dir so multi-node workers can access it
    tmpdir_base = str(project_dir / "benchmarks" / "results") if num_nodes > 1 else None
    with tempfile.TemporaryDirectory(dir=tmpdir_base) as benchmark_tmpdir:
        benchmark_tmpdir = Path(benchmark_tmpdir)
        metrics_file = benchmark_tmpdir / "metrics.json"

        # Worker script args
        cmd += [
            "--model", model_name,
            "--train-file", train_file,
            "--batch-size", str(batch_size),
            "--micro-batch-size", str(micro_batch_size),
            "--sequence-length", str(sequence_length),
            "--num-steps", str(num_steps),
            "--num-gpus", str(total_gpus),
            "--metrics-file", str(metrics_file),
            "--output-dir", str(benchmark_tmpdir / "output"),
        ]
        if offload:
            cmd.append("--offload")

        env = os.environ.copy()
        # Disable wandb/reporting in TRL
        env["WANDB_MODE"] = "disabled"

        # Run benchmark with GPU memory monitoring
        start_time = time.perf_counter()

        try:
            with GPUMemoryMonitor(poll_interval=1.0) as gpu_monitor:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=3600,  # 1 hour timeout
                    env=env,
                    cwd=project_dir,
                )

            total_time = time.perf_counter() - start_time

            if result.returncode != 0:
                combined = result.stdout + result.stderr

                # Check for OOM
                if "OutOfMemoryError" in combined or "CUDA out of memory" in combined:
                    oom_detail = ""
                    for line in combined.splitlines():
                        if "out of memory" in line.lower() or "tried to allocate" in line.lower():
                            oom_detail = line.strip()
                            break
                    raise torch.cuda.OutOfMemoryError(f"CUDA OOM: {oom_detail or combined[-1000:]}")
                raise RuntimeError(f"SFT benchmark failed: {combined[-8000:]}")

            # Extract metrics
            metrics = {
                "peak_memory_gb": gpu_monitor.peak_memory_gb,
                "tokens_per_second": None,
                "samples_per_second": None,
                "time_per_step_ms": None,
                "total_time_s": total_time,
            }

            # Read worker metrics file
            worker_metrics = {}
            if metrics_file.exists():
                with open(metrics_file) as f:
                    worker_metrics = json.load(f)

            if "avg_step_time_s" in worker_metrics:
                time_per_step_s = worker_metrics["avg_step_time_s"]
                metrics["time_per_step_ms"] = time_per_step_s * 1000
                metrics["metrics_source"] = worker_metrics.get("metrics_source", "trl_trainer")
            else:
                # Fallback: estimate from total time (subtract ~10s for model loading)
                effective_time = max(total_time - 10, total_time * 0.8)
                time_per_step_s = effective_time / num_steps
                metrics["time_per_step_ms"] = time_per_step_s * 1000
                metrics["metrics_source"] = "estimated"

            # Load dataset stats for accurate throughput calculation
            stats = load_dataset_stats(project_dir)

            if stats:
                response_ratio = stats.get("response_ratio", 0.7)
                avg_response_len = stats["response_length"]["mean"]
                avg_total_len = stats["total_length"]["mean"]

                trained_tokens_per_step = batch_size * avg_response_len
                processed_tokens_per_step = batch_size * min(sequence_length, avg_total_len)

                metrics["trained_tokens_per_second"] = trained_tokens_per_step / time_per_step_s
                metrics["processed_tokens_per_second"] = processed_tokens_per_step / time_per_step_s
                metrics["response_ratio"] = response_ratio
                metrics["avg_response_length"] = avg_response_len
                metrics["tokens_per_second"] = metrics["trained_tokens_per_second"]
            else:
                response_ratio = 0.7
                tokens_per_step = batch_size * sequence_length * response_ratio
                metrics["tokens_per_second"] = tokens_per_step / time_per_step_s
                metrics["response_ratio"] = response_ratio

            metrics["samples_per_second"] = batch_size / time_per_step_s

            return metrics

        except subprocess.TimeoutExpired:
            raise RuntimeError("SFT benchmark timed out (1 hour)")


# Allow running standalone for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()

    result = run_sft_benchmark(
        model_name=args.model,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        sequence_length=args.sequence_length,
        num_steps=args.num_steps,
        offload=args.offload,
    )

    print("\nResults:")
    for k, v in result.items():
        print(f"  {k}: {v}")
