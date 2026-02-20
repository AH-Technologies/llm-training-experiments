#!/usr/bin/env python3
"""
GRPO benchmark module.

Runs GRPO training via verl to measure throughput and memory usage.
Uses synthetic prompts to avoid dataset dependencies.

Extracts actual metrics from verl's file logger (most reliable),
console output, or falls back to wall-clock estimation.

Metrics extraction priority:
1. File logger JSONL (VERL_FILE_LOGGER_PATH) - most reliable
2. Console output parsing - immediate, flushed
3. Wall-clock estimation - fallback

Key GRPO metrics:
- perf/throughput: tokens/sec/GPU from verl
- perf/time_per_step: actual step timing
- response_length/mean: actual average response length
"""

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch

from benchmarks.gpu_monitor import GPUMemoryMonitor


# Fallback average response length if metrics unavailable
DEFAULT_AVG_RESPONSE_LENGTH = 300


def extract_file_logger_metrics(file_logger_path: Path) -> dict:
    """
    Extract metrics from verl's file logger JSONL output.

    verl's FileLogger writes lines like:
        {"step": N, "data": {"perf/throughput": 1234.5, "perf/time_per_step": 5.678, ...}}

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

            for key in ["perf/throughput", "perf/time_per_step",
                        "response_length/mean", "prompt_length/mean"]:
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
    Parse verl's console output for GRPO metrics.

    verl's console logger format (from aggregate_logger.py):
        step:N - perf/throughput:1234.5 - perf/time_per_step:5.678 - ...

    Returns dict with averaged metrics.
    """
    metrics = {}
    combined = stdout + stderr

    # Match verl console format for each metric key
    metric_patterns = {
        "perf/throughput": r"perf/throughput:(\d+\.?\d*)",
        "perf/time_per_step": r"perf/time_per_step:(\d+\.?\d*)",
        "response_length/mean": r"response_length/mean:(\d+\.?\d*)",
        "prompt_length/mean": r"prompt_length/mean:(\d+\.?\d*)",
    }

    for key, pattern in metric_patterns.items():
        values = []
        for match in re.finditer(pattern, combined):
            try:
                values.append(float(match.group(1)))
            except ValueError:
                pass
        if values:
            # Skip warmup steps
            if len(values) > 5:
                values = values[3:]
            metrics[key] = sum(values) / len(values)

    return metrics


def create_synthetic_dataset(num_samples: int, output_path: Path) -> Path:
    """Create a synthetic parquet dataset for GRPO benchmarking.

    Matches the format verl expects: prompt as chat messages (list of dicts),
    data_source as string, reward_model as dict with ground_truth.
    """
    import numpy as np

    samples = []
    for i in range(num_samples):
        a, b = i % 100, (i * 7) % 100
        answer = str(a + b)
        samples.append({
            "prompt": np.array(
                [{"role": "user", "content": f"What is {a} + {b}? Think step by step and provide the answer."}],
                dtype=object,
            ),
            "data_source": "synthetic_math",
            "reward_model": {"ground_truth": answer},
        })

    df = pd.DataFrame(samples)
    df.to_parquet(output_path)
    return output_path


def run_grpo_benchmark(
    model_name: str,
    batch_size: int,
    num_gpus: int,
    sequence_length: int,
    num_steps: int,
    rollout_n: int = 8,  # Match production settings
    temperature: float = 0.6,
    actor_offload: bool = True,  # CPU offload for actor params/optimizer
    ref_offload: bool = True,  # CPU offload for reference model (always True by default)
    num_nodes: int = 1,
    gpu_memory_utilization: float = 0.7,  # vLLM GPU memory fraction for KV cache
) -> dict:
    """
    Run GRPO benchmark via verl and return metrics.

    Note: GRPO is more memory-intensive than SFT due to:
    - Actor model (trainable)
    - Reference model (frozen, for KL)
    - Rollout buffers
    - vLLM inference engine

    CPU Offloading:
    - actor_offload: Offload actor params/optimizer to CPU (slower but saves GPU memory)
    - ref_offload: Offload reference model to CPU (almost always True for large models)

    Token Counting:
    - GRPO generates `rollout_n` responses per prompt
    - Trained tokens = batch_size x rollout_n x avg_response_length

    Returns dict with:
        - peak_memory_gb: Peak GPU memory usage
        - tokens_per_second: Training throughput (trained tokens)
        - generated_tokens_per_second: Total generated tokens
        - samples_per_second: Samples processed per second
        - time_per_step_ms: Average time per training step
        - total_time_s: Total benchmark time
        - offload_config: Dict describing offload settings
        - metrics_source: "file_logger", "console", or "estimated"
    """

    project_dir = Path(__file__).parent.parent

    # Create temporary synthetic dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        train_file = create_synthetic_dataset(
            num_samples=batch_size * num_steps * 2,
            output_path=tmpdir / "train.parquet",
        )
        val_file = create_synthetic_dataset(
            num_samples=100,
            output_path=tmpdir / "val.parquet",
        )

        # Prepare verl command
        max_prompt_length = min(512, sequence_length // 2)
        max_response_length = sequence_length - max_prompt_length

        # Scale token budget so it doesn't bottleneck large batches
        total_gpus = num_gpus * num_nodes
        ppo_max_token_len_per_gpu = max(16000, batch_size * sequence_length // total_gpus)

        cmd = [
            "python3", "-m", "verl.trainer.main_ppo",
            "algorithm.adv_estimator=grpo",
            f"data.train_files={train_file}",
            f"data.val_files={val_file}",
            f"data.train_batch_size={batch_size}",
            "data.val_batch_size=50",
            f"data.max_prompt_length={max_prompt_length}",
            f"data.max_response_length={max_response_length}",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            f"actor_rollout_ref.model.path={model_name}",
            "actor_rollout_ref.model.use_remove_padding=True",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            "actor_rollout_ref.actor.optim.lr=1e-6",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={batch_size}",
            "actor_rollout_ref.actor.use_dynamic_bsz=True",
            f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
            "actor_rollout_ref.actor.use_kl_loss=True",
            "actor_rollout_ref.actor.kl_loss_coef=0.001",
            f"actor_rollout_ref.actor.fsdp_config.param_offload={str(actor_offload)}",
            f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(actor_offload)}",
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.n={rollout_n}",
            f"actor_rollout_ref.rollout.temperature={temperature}",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}",
            f"actor_rollout_ref.rollout.max_model_len={sequence_length}",
            "actor_rollout_ref.rollout.enable_chunked_prefill=True",
            "actor_rollout_ref.rollout.enforce_eager=True",
            f"actor_rollout_ref.ref.fsdp_config.param_offload={str(ref_offload)}",
            "algorithm.kl_ctrl.kl_coef=0.001",
            "algorithm.use_kl_in_reward=False",
            # Use simple reward for benchmarking
            f"custom_reward_function.path={project_dir}/benchmarks/simple_reward.py",
            "custom_reward_function.name=compute_score",
            "trainer.critic_warmup=0",
            'trainer.logger=["console"]',
            "trainer.project_name=benchmark",
            "trainer.experiment_name=grpo_benchmark",
            f"trainer.n_gpus_per_node={num_gpus}",
            f"trainer.nnodes={num_nodes}",
            "trainer.save_freq=-1",  # Don't save checkpoints
            f"trainer.default_local_dir={tmpdir}/checkpoints",
            "trainer.val_before_train=False",
            "trainer.test_freq=9999",
            f"trainer.total_epochs={num_steps}",
            f"trainer.total_training_steps={num_steps}",
        ]

        # Set up file logger path
        file_logger_path = tmpdir / "grpo_metrics.jsonl"

        # Run benchmark
        start_time = time.perf_counter()

        env = os.environ.copy()
        env["VERL_FILE_LOGGER_PATH"] = str(file_logger_path)
        # Remove ROCR_VISIBLE_DEVICES to prevent conflict with CUDA_VISIBLE_DEVICES
        # that Ray sets for workers. verl raises ValueError if both are set.
        env.pop("ROCR_VISIBLE_DEVICES", None)

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
                # Check for OOM (CUDA OOM or vLLM KV cache memory errors)
                combined_output = result.stderr + result.stdout
                oom_patterns = [
                    "OutOfMemoryError",
                    "CUDA out of memory",
                    "No available memory for the cache blocks",
                    "KV cache is needed, which is larger than the available",
                ]
                if any(pat in combined_output for pat in oom_patterns):
                    # Find the line with memory details
                    oom_detail = ""
                    for line in combined_output.splitlines():
                        if "out of memory" in line.lower() or "tried to allocate" in line.lower() or "cache blocks" in line.lower():
                            oom_detail = line.strip()
                            break
                    raise torch.cuda.OutOfMemoryError(f"CUDA OOM: {oom_detail or combined_output[-1000:]}")
                raise RuntimeError(f"GRPO failed: {result.stderr[-1000:]}")

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

            # Record offload configuration
            metrics["offload_config"] = {
                "actor_offload": actor_offload,
                "ref_offload": ref_offload,
            }

            # Tier 1: File logger
            file_metrics = extract_file_logger_metrics(file_logger_path)

            # Tier 2: Console output (used as fallback for each metric)
            console_metrics = parse_console_metrics(result.stdout, result.stderr)

            # Determine metrics source and extract timing
            if "perf/time_per_step" in file_metrics:
                metrics["time_per_step_ms"] = file_metrics["perf/time_per_step"] * 1000
                metrics["metrics_source"] = "file_logger"
            elif "perf/time_per_step" in console_metrics:
                metrics["time_per_step_ms"] = console_metrics["perf/time_per_step"] * 1000
                metrics["metrics_source"] = "console"
            else:
                metrics["metrics_source"] = "estimated"

            # Extract throughput (prefer file logger, then console)
            verl_metrics = file_metrics if file_metrics else console_metrics
            if "perf/throughput" in verl_metrics:
                # verl logs throughput as tokens/sec/GPU, multiply by num_gpus
                metrics["tokens_per_second"] = verl_metrics["perf/throughput"] * num_gpus
                metrics["wandb_throughput_per_gpu"] = verl_metrics["perf/throughput"]

            if "response_length/mean" in verl_metrics:
                metrics["avg_response_length"] = verl_metrics["response_length/mean"]

            if "prompt_length/mean" in verl_metrics:
                metrics["avg_prompt_length"] = verl_metrics["prompt_length/mean"]

            # Calculate time per step (prefer extracted, fallback to estimate)
            if metrics.get("time_per_step_ms") is not None:
                time_per_step_s = metrics["time_per_step_ms"] / 1000
            else:
                # Fallback: estimate from total time (subtract ~30s for model loading)
                effective_time = max(total_time - 30, total_time * 0.7)
                time_per_step_s = effective_time / num_steps
                metrics["time_per_step_ms"] = time_per_step_s * 1000

            # Calculate token throughput if not from verl
            avg_response_length = metrics.get("avg_response_length", DEFAULT_AVG_RESPONSE_LENGTH)

            if "tokens_per_second" not in metrics or metrics["tokens_per_second"] is None:
                # Fallback calculation: batch_size x rollout_n x avg_response_length
                generated_tokens_per_step = batch_size * rollout_n * avg_response_length
                metrics["tokens_per_second"] = generated_tokens_per_step / time_per_step_s

            # Additional metrics
            metrics["generated_tokens_per_second"] = metrics["tokens_per_second"]
            metrics["samples_per_second"] = (batch_size * rollout_n) / time_per_step_s
            metrics["prompts_per_second"] = batch_size / time_per_step_s

            # Store config for analysis
            metrics["rollout_n"] = rollout_n
            metrics["avg_response_length"] = avg_response_length

            return metrics

        except subprocess.TimeoutExpired:
            raise RuntimeError("GRPO benchmark timed out (1 hour)")


# Allow running standalone
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--rollout-n", type=int, default=8)
    parser.add_argument("--actor-offload", action="store_true",
                        help="Enable CPU offloading for actor model")
    parser.add_argument("--no-ref-offload", action="store_true",
                        help="Disable CPU offloading for reference model")
    args = parser.parse_args()

    result = run_grpo_benchmark(
        model_name=args.model,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        sequence_length=args.sequence_length,
        num_steps=args.num_steps,
        rollout_n=args.rollout_n,
        actor_offload=args.actor_offload,
        ref_offload=not args.no_ref_offload,
    )

    print("\nResults:")
    for k, v in result.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
