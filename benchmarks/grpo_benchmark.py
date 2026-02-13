#!/usr/bin/env python3
"""
GRPO benchmark module.

Runs GRPO training via verl to measure throughput and memory usage.
Uses synthetic prompts to avoid dataset dependencies.

Extracts actual metrics from wandb logs for accurate measurements:
- perf/throughput: tokens/sec/GPU from verl
- response_length/mean: actual average response length
- perf/time_per_step: actual step timing
"""

import gc
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch


# Fallback average response length if wandb metrics unavailable
DEFAULT_AVG_RESPONSE_LENGTH = 300


def extract_wandb_metrics(wandb_dir: Path) -> dict:
    """
    Extract metrics from wandb offline logs.

    verl logs these useful metrics to wandb:
    - perf/throughput: tokens/sec/GPU (actual measured)
    - perf/time_per_step: actual step time
    - response_length/mean: actual average response length
    - prompt_length/mean: actual prompt length

    Returns dict with averaged metrics from the training run.
    """
    metrics = {}

    # Find the most recent run directory
    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        return metrics

    run_dir = run_dirs[0]

    # Try to read the wandb binary file using wandb's API
    try:
        import wandb

        # Load the run from offline files
        history_file = run_dir / "files" / "wandb-history.jsonl"
        if history_file.exists():
            with open(history_file) as f:
                steps_data = [json.loads(line) for line in f]

            if steps_data:
                # Aggregate metrics (skip warmup steps)
                skip_steps = min(3, len(steps_data) // 4)
                valid_steps = steps_data[skip_steps:]

                for key in ["perf/throughput", "perf/time_per_step",
                            "response_length/mean", "prompt_length/mean"]:
                    values = [s.get(key) for s in valid_steps if s.get(key) is not None]
                    if values:
                        metrics[key] = sum(values) / len(values)

    except Exception as e:
        # Fallback: try to parse the .wandb binary file
        pass

    # Also try parsing the summary file
    summary_file = run_dir / "files" / "wandb-summary.json"
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                summary = json.load(f)
                for key in ["perf/throughput", "perf/time_per_step",
                            "response_length/mean", "prompt_length/mean"]:
                    if key in summary and key not in metrics:
                        metrics[key] = summary[key]
        except Exception:
            pass

    return metrics


def create_synthetic_dataset(num_samples: int, output_path: Path) -> Path:
    """Create a synthetic parquet dataset for GRPO benchmarking."""

    # Simple math-like prompts
    prompts = []
    for i in range(num_samples):
        a, b = i % 100, (i * 7) % 100
        prompt = f"What is {a} + {b}? Think step by step and provide the answer."
        prompts.append({
            "prompt": prompt,
            "answer": str(a + b),
        })

    df = pd.DataFrame(prompts)
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
    actor_offload: bool = False,  # CPU offload for actor params/optimizer
    ref_offload: bool = True,  # CPU offload for reference model (always True by default)
    num_nodes: int = 1,
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
    - Trained tokens = batch_size × rollout_n × avg_response_length

    Returns dict with:
        - peak_memory_gb: Peak GPU memory usage
        - tokens_per_second: Training throughput (trained tokens)
        - generated_tokens_per_second: Total generated tokens
        - samples_per_second: Samples processed per second
        - time_per_step_ms: Average time per training step
        - total_time_s: Total benchmark time
        - offload_config: Dict describing offload settings
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
        # We'll capture timing from the output
        max_prompt_length = min(512, sequence_length // 2)
        max_response_length = sequence_length - max_prompt_length

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
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000",
            "actor_rollout_ref.actor.use_kl_loss=True",
            "actor_rollout_ref.actor.kl_loss_coef=0.001",
            f"actor_rollout_ref.actor.fsdp_config.param_offload={str(actor_offload)}",
            f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(actor_offload)}",
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.n={rollout_n}",
            f"actor_rollout_ref.rollout.temperature={temperature}",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
            "actor_rollout_ref.rollout.enforce_eager=True",
            f"actor_rollout_ref.ref.fsdp_config.param_offload={str(ref_offload)}",
            "algorithm.kl_ctrl.kl_coef=0.001",
            "algorithm.use_kl_in_reward=False",
            # Use simple reward for benchmarking
            f"custom_reward_function.path={project_dir}/benchmarks/simple_reward.py",
            "custom_reward_function.name=compute_score",
            "trainer.critic_warmup=0",
            'trainer.logger=["console","wandb"]',
            "trainer.project_name=benchmark",
            "trainer.experiment_name=grpo_benchmark",
            f"trainer.n_gpus_per_node={num_gpus}",
            f"trainer.nnodes={num_nodes}",
            "trainer.save_freq=9999",  # Don't save checkpoints
            "trainer.val_before_train=False",
            "trainer.test_freq=9999",
            f"trainer.total_epochs={num_steps}",
            f"trainer.total_training_steps={num_steps}",
        ]

        # Create temporary wandb directory for this benchmark run
        wandb_dir = tmpdir / "wandb"
        wandb_dir.mkdir()

        # Run benchmark
        start_time = time.perf_counter()

        env = os.environ.copy()
        env["WANDB_MODE"] = "offline"  # Save logs locally for extraction
        env["WANDB_DIR"] = str(wandb_dir)

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
                # Check for OOM
                if "OutOfMemoryError" in result.stderr or "CUDA out of memory" in result.stderr:
                    raise torch.cuda.OutOfMemoryError("CUDA OOM in GRPO")
                raise RuntimeError(f"GRPO failed: {result.stderr[-1000:]}")

            # Parse metrics from console output as fallback
            metrics = _parse_verl_output(result.stdout, result.stderr)
            metrics["total_time_s"] = total_time

            # Try to extract actual metrics from wandb logs
            wandb_metrics = extract_wandb_metrics(wandb_dir)

            # Record offload configuration
            metrics["offload_config"] = {
                "actor_offload": actor_offload,
                "ref_offload": ref_offload,
            }

            # Use actual metrics from wandb if available
            if "perf/throughput" in wandb_metrics:
                # verl logs throughput as tokens/sec/GPU, multiply by num_gpus
                metrics["tokens_per_second"] = wandb_metrics["perf/throughput"] * num_gpus
                metrics["wandb_throughput_per_gpu"] = wandb_metrics["perf/throughput"]

            if "perf/time_per_step" in wandb_metrics:
                metrics["time_per_step_ms"] = wandb_metrics["perf/time_per_step"] * 1000

            if "response_length/mean" in wandb_metrics:
                metrics["avg_response_length"] = wandb_metrics["response_length/mean"]

            if "prompt_length/mean" in wandb_metrics:
                metrics["avg_prompt_length"] = wandb_metrics["prompt_length/mean"]

            # Calculate time per step (prefer wandb, fallback to estimate)
            if metrics.get("time_per_step_ms") is not None:
                time_per_step_s = metrics["time_per_step_ms"] / 1000
            else:
                # Fallback: estimate from total time (subtract ~30s for model loading)
                effective_time = max(total_time - 30, total_time * 0.7)
                time_per_step_s = effective_time / num_steps
                metrics["time_per_step_ms"] = time_per_step_s * 1000

            # Calculate token throughput if not from wandb
            avg_response_length = metrics.get("avg_response_length", DEFAULT_AVG_RESPONSE_LENGTH)

            if "tokens_per_second" not in metrics or metrics["tokens_per_second"] is None:
                # Fallback calculation: batch_size × rollout_n × avg_response_length
                generated_tokens_per_step = batch_size * rollout_n * avg_response_length
                metrics["tokens_per_second"] = generated_tokens_per_step / time_per_step_s

            # Additional metrics
            metrics["generated_tokens_per_second"] = metrics["tokens_per_second"]
            metrics["samples_per_second"] = (batch_size * rollout_n) / time_per_step_s
            metrics["prompts_per_second"] = batch_size / time_per_step_s

            # Store config for analysis
            metrics["rollout_n"] = rollout_n
            metrics["avg_response_length"] = avg_response_length
            metrics["metrics_source"] = "wandb" if "perf/throughput" in wandb_metrics else "estimated"

            return metrics

        except subprocess.TimeoutExpired:
            raise RuntimeError("GRPO benchmark timed out (1 hour)")


def _parse_verl_output(stdout: str, stderr: str) -> dict:
    """Parse verl output to extract metrics."""

    # Default values if parsing fails
    metrics = {
        "peak_memory_gb": None,
        "tokens_per_second": None,
        "samples_per_second": None,
        "time_per_step_ms": None,
    }

    combined = stdout + stderr

    # Look for timing info in verl output
    # verl typically logs: "step X, time: Y.Zs"
    import re

    step_times = []
    for match in re.finditer(r"step.*?time[:\s]+(\d+\.?\d*)\s*s", combined, re.IGNORECASE):
        try:
            step_times.append(float(match.group(1)) * 1000)  # Convert to ms
        except ValueError:
            pass

    if step_times:
        metrics["time_per_step_ms"] = sum(step_times) / len(step_times)

    # Look for memory info
    for match in re.finditer(r"memory.*?(\d+\.?\d*)\s*GB", combined, re.IGNORECASE):
        try:
            metrics["peak_memory_gb"] = float(match.group(1))
        except ValueError:
            pass

    # Look for throughput
    for match in re.finditer(r"throughput.*?(\d+\.?\d*)\s*tokens?/s", combined, re.IGNORECASE):
        try:
            metrics["tokens_per_second"] = float(match.group(1))
        except ValueError:
            pass

    return metrics


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
