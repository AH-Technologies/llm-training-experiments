#!/usr/bin/env python3
"""
SFT benchmark module using verl's fsdp_sft_trainer.

Runs SFT training via verl to measure throughput and memory usage.
Uses OpenMathInstruct-2 dataset for realistic throughput measurements.

Throughput is calculated as TRAINED tokens/sec (response tokens only),
not total processed tokens, for accurate comparison.

Extracts actual timing from wandb logs:
- train/time(s): actual time per training step from verl
"""

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch


# Default dataset paths (relative to project root)
DEFAULT_TRAIN_FILE = "data/benchmark/openmath_sft_train.parquet"
DEFAULT_VAL_FILE = "data/benchmark/openmath_sft_val.parquet"
DEFAULT_STATS_FILE = "data/benchmark/openmath_sft_stats.json"


def extract_wandb_metrics(wandb_dir: Path) -> dict:
    """
    Extract metrics from wandb offline logs.

    verl's SFT trainer logs:
    - train/time(s): time per training step
    - train/loss: training loss

    Returns dict with averaged metrics from the training run.
    """
    metrics = {}

    # Find the most recent run directory
    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        return metrics

    run_dir = run_dirs[0]

    # Try to read wandb history file
    history_file = run_dir / "files" / "wandb-history.jsonl"
    if history_file.exists():
        try:
            with open(history_file) as f:
                steps_data = [json.loads(line) for line in f]

            if steps_data:
                # Aggregate metrics (skip warmup steps)
                skip_steps = min(3, len(steps_data) // 4)
                valid_steps = steps_data[skip_steps:]

                for key in ["train/time(s)", "train/loss"]:
                    values = [s.get(key) for s in valid_steps if s.get(key) is not None]
                    if values:
                        metrics[key] = sum(values) / len(values)
        except Exception:
            pass

    # Also try summary file
    summary_file = run_dir / "files" / "wandb-summary.json"
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                summary = json.load(f)
                for key in ["train/time(s)", "train/loss"]:
                    if key in summary and key not in metrics:
                        metrics[key] = summary[key]
        except Exception:
            pass

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
    node_rank = int(os.environ.get("SLURM_NODEID", "0"))

    # Build verl fsdp_sft_trainer command
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "-m", "verl.trainer.fsdp_sft_trainer",
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
        'trainer.logger=["console","wandb"]',
        f"trainer.n_gpus_per_node={num_gpus}",
        f"trainer.nnodes={num_nodes}",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.resume_mode=disable",
    ]

    # Create temporary directories for this benchmark run (no persistent checkpoints)
    with tempfile.TemporaryDirectory() as benchmark_tmpdir, \
         tempfile.TemporaryDirectory() as wandb_tmpdir:
        # Point checkpoint dir to temp so last-step checkpoint is discarded
        cmd.append(f"trainer.default_local_dir={benchmark_tmpdir}/checkpoints")

        wandb_dir = Path(wandb_tmpdir) / "wandb"
        wandb_dir.mkdir()

        env = os.environ.copy()
        env["WANDB_MODE"] = "offline"  # Save logs locally for extraction
        env["WANDB_DIR"] = str(wandb_dir)

        # Run benchmark
        start_time = time.perf_counter()

        try:
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

            # Parse metrics from console output as fallback
            metrics = _parse_verl_sft_output(result.stdout, result.stderr)
            metrics["total_time_s"] = total_time

            # Try to extract actual metrics from wandb logs
            wandb_metrics = extract_wandb_metrics(wandb_dir)

            # Use actual time per step from wandb if available
            if "train/time(s)" in wandb_metrics:
                metrics["time_per_step_ms"] = wandb_metrics["train/time(s)"] * 1000
                metrics["metrics_source"] = "wandb"
            else:
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


def _parse_verl_sft_output(stdout: str, stderr: str) -> dict:
    """Parse verl SFT trainer output to extract metrics."""

    metrics = {
        "peak_memory_gb": None,
        "tokens_per_second": None,
        "samples_per_second": None,
        "time_per_step_ms": None,
    }

    combined = stdout + stderr

    # Look for step timing info
    # verl logs: "step X/Y, loss: Z, time: Ws"
    step_times = []
    for match in re.finditer(r"time[:\s]+(\d+\.?\d*)\s*s", combined, re.IGNORECASE):
        try:
            step_times.append(float(match.group(1)) * 1000)  # Convert to ms
        except ValueError:
            pass

    # Also look for ms format
    for match in re.finditer(r"time[:\s]+(\d+\.?\d*)\s*ms", combined, re.IGNORECASE):
        try:
            step_times.append(float(match.group(1)))
        except ValueError:
            pass

    if step_times:
        # Skip first few warmup steps
        if len(step_times) > 5:
            step_times = step_times[3:]
        metrics["time_per_step_ms"] = sum(step_times) / len(step_times)

    # Look for memory info
    for match in re.finditer(r"memory.*?(\d+\.?\d*)\s*GB", combined, re.IGNORECASE):
        try:
            metrics["peak_memory_gb"] = float(match.group(1))
        except ValueError:
            pass

    # Also check nvidia-smi style output
    for match in re.finditer(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", combined):
        try:
            used_mb = float(match.group(1))
            metrics["peak_memory_gb"] = max(
                metrics["peak_memory_gb"] or 0,
                used_mb / 1024
            )
        except ValueError:
            pass

    return metrics


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
