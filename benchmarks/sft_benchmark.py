#!/usr/bin/env python3
"""
SFT benchmark module using verl's fsdp_sft_trainer.

Runs SFT training via verl to measure throughput and memory usage.
Uses OpenMathInstruct-2 dataset for realistic throughput measurements.

Throughput is calculated as TRAINED tokens/sec (response tokens only),
not total processed tokens, for accurate comparison.

Extracts actual timing from verl's file logger (most reliable),
console output, or falls back to wall-clock estimation.

Metrics extraction priority:
1. File logger JSONL (VERL_FILE_LOGGER_PATH) - most reliable
2. Console output parsing - immediate, flushed
3. Wall-clock estimation - fallback
"""

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from benchmarks.gpu_monitor import GPUMemoryMonitor


# Default dataset paths (relative to project root)
DEFAULT_TRAIN_FILE = "data/benchmark/openmath_sft_train.parquet"
DEFAULT_VAL_FILE = "data/benchmark/openmath_sft_val.parquet"
DEFAULT_STATS_FILE = "data/benchmark/openmath_sft_stats.json"


def extract_file_logger_metrics(file_logger_path: Path) -> dict:
    """
    Extract metrics from verl's file logger JSONL output.

    verl's FileLogger writes lines like:
        {"step": N, "data": {"train/time(s)": 5.678, "train/loss": 0.123, ...}}

    Returns dict with averaged metrics from the training run.
    """
    metrics = {}
    if not file_logger_path.exists():
        return metrics

    try:
        steps_data = []
        with open(file_logger_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    steps_data.append(json.loads(line))

        if steps_data:
            # Skip warmup steps
            skip_steps = min(3, len(steps_data) // 4)
            valid_steps = steps_data[skip_steps:]

            for key in ["train/time(s)", "train/loss"]:
                values = [
                    s["data"].get(key)
                    for s in valid_steps
                    if isinstance(s.get("data"), dict) and s["data"].get(key) is not None
                ]
                if values:
                    metrics[key] = sum(values) / len(values)
    except Exception:
        pass

    return metrics


def parse_console_metrics(stdout: str, stderr: str) -> dict:
    """
    Parse verl's console output for SFT metrics.

    verl's console logger format (from aggregate_logger.py):
        step:N - train/time(s):5.678 - train/loss:0.123 - ...

    Returns dict with averaged metrics.
    """
    metrics = {}
    combined = stdout + stderr

    # Match verl console format: train/time(s):<value>
    step_times = []
    for match in re.finditer(r"train/time\(s\):(\d+\.?\d*)", combined):
        try:
            step_times.append(float(match.group(1)))
        except ValueError:
            pass

    if step_times:
        # Skip warmup steps
        if len(step_times) > 5:
            step_times = step_times[3:]
        metrics["train/time(s)"] = sum(step_times) / len(step_times)

    return metrics


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
    Run SFT benchmark via verl's fsdp_sft_trainer and return metrics.

    Uses real math datasets for realistic throughput measurements.

    Returns dict with:
        - peak_memory_gb: Peak GPU memory usage
        - tokens_per_second: Training throughput (actual trained tokens)
        - samples_per_second: Samples processed per second
        - time_per_step_ms: Average time per training step
        - total_time_s: Total benchmark time
        - metrics_source: "file_logger", "console", or "estimated"
    """

    project_dir = Path(__file__).parent.parent

    # Use real datasets
    train_file = train_file or str(project_dir / DEFAULT_TRAIN_FILE)
    val_file = val_file or str(project_dir / DEFAULT_VAL_FILE)

    # Calculate micro batch size (account for all GPUs across nodes)
    total_gpus = num_gpus * num_nodes
    micro_batch_size = max(1, batch_size // (total_gpus * gradient_accumulation))

    # Get multi-node settings from environment (set by SLURM)
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29502")

    # Build verl fsdp_sft_trainer command
    if num_nodes > 1:
        # Multi-node: use srun to launch torchrun on each allocated node.
        # c10d rendezvous handles node rank assignment automatically.
        cmd = [
            "srun",
            f"--nodes={num_nodes}",
            "--ntasks-per-node=1",
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            f"--nnodes={num_nodes}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:{master_port}",
            "-m", "verl.trainer.fsdp_sft_trainer",
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
            "-m", "verl.trainer.fsdp_sft_trainer",
        ]

    cmd += [
        f"model.partial_pretrain={model_name}",
        "model.trust_remote_code=True",
        "model.enable_gradient_checkpointing=True",
        "model.fsdp_config.model_dtype=bf16",
        "model.strategy=fsdp2",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={batch_size}",
        f"data.micro_batch_size_per_gpu={micro_batch_size}",
        f"data.max_length={sequence_length}",
        "data.prompt_key=prompt",
        "data.response_key=response",
        "data.truncation=left",
        "optim.lr=1e-5",
        "optim.weight_decay=0.01",
        "optim.clip_grad=1.0",
        "optim.lr_scheduler=cosine",
        "optim.lr_warmup_steps_ratio=0.0",
        "trainer.project_name=benchmark",
        "trainer.experiment_name=sft_benchmark",
        f"trainer.total_training_steps={num_steps}",
        "trainer.total_epochs=9999",  # Use steps, not epochs
        'trainer.logger=["console","file"]',
        f"trainer.n_gpus_per_node={num_gpus}",
        f"trainer.nnodes={num_nodes}",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.resume_mode=disable",
    ]

    # CPU offloading for large models that don't fit without it
    if offload:
        cmd.extend([
            f"model.fsdp_config.cpu_offload=True",
            f"model.fsdp_config.offload_params=True",
        ])

    # Create temporary directories for this benchmark run (no persistent checkpoints)
    with tempfile.TemporaryDirectory() as benchmark_tmpdir:
        benchmark_tmpdir = Path(benchmark_tmpdir)

        # Point checkpoint dir to temp so last-step checkpoint is discarded
        cmd.append(f"trainer.default_local_dir={benchmark_tmpdir}/checkpoints")

        # Set up file logger path
        file_logger_path = benchmark_tmpdir / "sft_metrics.jsonl"

        env = os.environ.copy()
        env["VERL_FILE_LOGGER_PATH"] = str(file_logger_path)

        # Run benchmark with GPU memory monitoring
        start_time = time.perf_counter()

        try:
            with GPUMemoryMonitor(poll_interval=1.0) as gpu_monitor:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                    env=env,
                    cwd=project_dir,
                )

            total_time = time.perf_counter() - start_time

            if result.returncode != 0:
                stderr = result.stderr
                stdout = result.stdout
                combined = stdout + stderr

                # Check for OOM
                if "OutOfMemoryError" in combined or "CUDA out of memory" in combined:
                    raise torch.cuda.OutOfMemoryError(f"CUDA OOM: {combined[-1000:]}")
                raise RuntimeError(f"SFT benchmark failed: {combined[-8000:]}")

            # Three-tier metric extraction:
            # 1. File logger JSONL (most reliable)
            # 2. Console output parsing
            # 3. Wall-clock estimation (fallback)

            metrics = {
                "peak_memory_gb": gpu_monitor.peak_memory_gb,
                "tokens_per_second": None,
                "samples_per_second": None,
                "time_per_step_ms": None,
            }
            metrics["total_time_s"] = total_time

            # Tier 1: File logger
            file_metrics = extract_file_logger_metrics(file_logger_path)
            if "train/time(s)" in file_metrics:
                metrics["time_per_step_ms"] = file_metrics["train/time(s)"] * 1000
                metrics["metrics_source"] = "file_logger"
            else:
                # Tier 2: Console output
                console_metrics = parse_console_metrics(result.stdout, result.stderr)
                if "train/time(s)" in console_metrics:
                    metrics["time_per_step_ms"] = console_metrics["train/time(s)"] * 1000
                    metrics["metrics_source"] = "console"
                else:
                    # Tier 3: Wall-clock estimation
                    metrics["metrics_source"] = "estimated"

            # Calculate time per step
            if metrics.get("time_per_step_ms") is not None:
                time_per_step_s = metrics["time_per_step_ms"] / 1000
            else:
                # Fallback: estimate from total time (subtract ~10s for model loading)
                effective_time = max(total_time - 10, total_time * 0.8)
                time_per_step_s = effective_time / num_steps
                metrics["time_per_step_ms"] = time_per_step_s * 1000

            # Load dataset stats for accurate throughput calculation
            stats = load_dataset_stats(project_dir)

            if stats:
                # Use actual response ratio from dataset analysis
                response_ratio = stats.get("response_ratio", 0.7)
                avg_response_len = stats["response_length"]["mean"]
                avg_total_len = stats["total_length"]["mean"]

                # Trained tokens = only response tokens (what actually contributes to loss)
                trained_tokens_per_step = batch_size * avg_response_len
                processed_tokens_per_step = batch_size * min(sequence_length, avg_total_len)

                metrics["trained_tokens_per_second"] = trained_tokens_per_step / time_per_step_s
                metrics["processed_tokens_per_second"] = processed_tokens_per_step / time_per_step_s
                metrics["response_ratio"] = response_ratio
                metrics["avg_response_length"] = avg_response_len

                # Primary metric is trained tokens
                metrics["tokens_per_second"] = metrics["trained_tokens_per_second"]
            else:
                # Fallback: estimate with 70% response ratio (typical for SFT)
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
    args = parser.parse_args()

    result = run_sft_benchmark(
        model_name=args.model,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        sequence_length=args.sequence_length,
        num_steps=args.num_steps,
    )

    print("\nResults:")
    for k, v in result.items():
        print(f"  {k}: {v}")
