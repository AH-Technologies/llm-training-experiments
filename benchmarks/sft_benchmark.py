#!/usr/bin/env python3
"""
SFT benchmark module using TRL's SFTTrainer.

Runs SFT training via torchrun + TRL to measure throughput and memory usage.
Uses OpenMathInstruct-2 dataset for realistic throughput measurements.

Supports two backends:
- FSDP1 (default): for 0.5B-14B models
- DeepSpeed ZeRO-3 (deepspeed_config): for 72B+ models with full CPU offload

Throughput is calculated as TRAINED tokens/sec (response tokens only),
not total processed tokens, for accurate comparison.

Metrics extraction:
1. Worker JSON file (metrics written by sft_trl_worker.py)
2. Wall-clock estimation (fallback)
"""

import json
import os
import subprocess
import sys
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
    tp_degree: int = 1,
    deepspeed_config: str = None,
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
    # DeepSpeed ZeRO-3: all GPUs are DP (no TP), same as FSDP with tp=1
    # Cap at 8 to avoid activation memory OOM — excess is handled via gradient accumulation
    total_gpus = num_gpus * num_nodes
    dp_size = total_gpus if deepspeed_config else total_gpus // tp_degree
    micro_batch_size = min(8, max(1, batch_size // (dp_size * gradient_accumulation)))

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
            "--export=ALL",
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
        if deepspeed_config:
            cmd += ["--deepspeed", deepspeed_config]
        elif offload:
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


def run_sft_torchtune_benchmark(
    model_name: str,
    batch_size: int,
    num_gpus: int,
    sequence_length: int,
    num_steps: int,
    gradient_accumulation: int = 1,
    train_file: str = None,
    num_nodes: int = 1,
    tp_degree: int = 1,
    cpu_offload: bool = False,
) -> dict:
    """
    Run SFT benchmark via torchtune FSDP2+TP and return metrics.

    Unified strategy for all model sizes: FSDP2 shards params+grads+optimizer
    across all N GPUs; TP (T ≤ gpus_per_node) shards activations within a node.
    No CPU offload — relies on FSDP2 sharding for memory efficiency.

    Returns same metrics dict as run_sft_benchmark.
    """
    project_dir = Path(__file__).parent.parent

    train_file = train_file or str(project_dir / DEFAULT_TRAIN_FILE)

    total_gpus = num_gpus * num_nodes
    dp_size = total_gpus // tp_degree
    micro_batch_size = min(8, max(1, batch_size // (dp_size * gradient_accumulation)))

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29503")  # Different port from TRL worker

    tmpdir_base = str(project_dir / "benchmarks" / "results") if num_nodes > 1 else None
    with tempfile.TemporaryDirectory(dir=tmpdir_base) as benchmark_tmpdir:
        benchmark_tmpdir = Path(benchmark_tmpdir)
        metrics_file = benchmark_tmpdir / "metrics.json"
        (benchmark_tmpdir / "output").mkdir(exist_ok=True)

        worker_args = [
            "--model",            model_name,
            "--train-file",       str(train_file),
            "--batch-size",       str(batch_size),
            "--micro-batch-size", str(micro_batch_size),
            "--sequence-length",  str(sequence_length),
            "--num-steps",        str(num_steps),
            "--num-gpus",         str(total_gpus),
            "--tp-degree",        str(tp_degree),
            "--metrics-file",     str(metrics_file),
            "--output-dir",       str(benchmark_tmpdir / "output"),
        ]
        if cpu_offload:
            worker_args.append("--cpu-offload")

        if num_nodes > 1:
            # Multi-node: srun launches one torchrun per node.
            # Must unset ROCR_VISIBLE_DEVICES inside srun context (SLURM re-injects it).
            torchrun_cmd = " ".join([
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                f"--nnodes={num_nodes}",
                "--rdzv_backend=c10d",
                f"--rdzv_endpoint={master_addr}:{master_port}",
                "-m", "benchmarks.sft_torchtune_worker",
            ] + worker_args)
            cmd = [
                "srun",
                f"--nodes={num_nodes}",
                "--ntasks-per-node=1",
                "--export=ALL",
                "bash", "-c", f"unset ROCR_VISIBLE_DEVICES; {torchrun_cmd}",
            ]
        else:
            cmd = [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                "--nnodes=1",
                "--node_rank=0",
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                "-m", "benchmarks.sft_torchtune_worker",
            ] + worker_args

        env = os.environ.copy()
        env.pop("ROCR_VISIBLE_DEVICES", None)
        start_time = time.perf_counter()

        try:
            # Stream output in real-time (visible in SLURM logs) while also
            # capturing it for OOM detection.
            log_path = benchmark_tmpdir / "sft_subprocess.log"
            with GPUMemoryMonitor(poll_interval=1.0) as gpu_monitor, \
                 open(log_path, "w") as log_fh:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    env=env,
                    cwd=project_dir,
                )
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_fh.write(line)
                proc.wait(timeout=3600)

            total_time = time.perf_counter() - start_time

            combined = log_path.read_text(errors="replace")
            if proc.returncode != 0:
                oom_patterns = [
                    "OutOfMemoryError", "CUDA out of memory",
                    "out of memory", "OOM",
                ]
                if any(pat in combined for pat in oom_patterns):
                    oom_detail = ""
                    for line in combined.splitlines():
                        if "out of memory" in line.lower() or "tried to allocate" in line.lower():
                            oom_detail = line.strip()
                            break
                    raise torch.cuda.OutOfMemoryError(f"CUDA OOM: {oom_detail or combined[-1000:]}")
                error_lines = []
                for line in combined.splitlines():
                    lower = line.lower()
                    if any(k in lower for k in ["error", "traceback", "assert", "exception", "raise "]):
                        error_lines.append(line.strip())
                error_summary = "\n".join(error_lines[-20:]) if error_lines else combined[-2000:]
                raise RuntimeError(f"SFT torchtune failed:\n{error_summary}")

            metrics = {
                "peak_memory_gb": gpu_monitor.peak_memory_gb,
                "tokens_per_second": None,
                "samples_per_second": None,
                "time_per_step_ms": None,
                "total_time_s": total_time,
            }

            if metrics_file.exists():
                with open(metrics_file) as f:
                    worker_metrics = json.load(f)
                # Worker already computes tok/s, samples/s, peak_mem
                metrics.update({
                    "time_per_step_ms":   worker_metrics.get("time_per_step_ms"),
                    "tokens_per_second":  worker_metrics.get("tokens_per_second"),
                    "samples_per_second": worker_metrics.get("samples_per_second"),
                    "peak_memory_gb":     worker_metrics.get("peak_memory_gb") or gpu_monitor.peak_memory_gb,
                    "metrics_source":     worker_metrics.get("metrics_source", "torchtune"),
                })
            else:
                effective_time = max(total_time - 10, total_time * 0.8)
                time_per_step_s = effective_time / num_steps
                metrics["time_per_step_ms"] = time_per_step_s * 1000
                metrics["metrics_source"] = "estimated"

            # Compute tok/s from dataset stats if worker didn't provide it
            if metrics["tokens_per_second"] is None and metrics["time_per_step_ms"]:
                time_per_step_s = metrics["time_per_step_ms"] / 1000
                stats = load_dataset_stats(project_dir)
                if stats:
                    metrics["tokens_per_second"] = (
                        batch_size * stats["response_length"]["mean"] / time_per_step_s
                    )
                else:
                    metrics["tokens_per_second"] = batch_size * sequence_length * 0.7 / time_per_step_s
                metrics["samples_per_second"] = batch_size / time_per_step_s

            return metrics

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError("SFT torchtune benchmark timed out (1 hour)")


def create_synthetic_sft_dataset(num_samples: int, output_path: Path) -> Path:
    """Create a synthetic parquet dataset for SFT benchmarking.

    Matches the format verl MultiTurnSFTDataset expects: 'messages' column
    with chat-format messages (list of dicts with role/content).
    """
    import numpy as np
    import pandas as pd

    samples = []
    for i in range(num_samples):
        a, b = i % 100, (i * 7) % 100
        answer = a + b
        messages = [
            {"role": "user", "content": f"What is {a} + {b}? Think step by step and provide the answer."},
            {"role": "assistant", "content": (
                f"Let me solve {a} + {b} step by step.\n"
                f"First, I identify the two numbers: {a} and {b}.\n"
                f"Adding them together: {a} + {b} = {answer}.\n"
                f"The answer is {answer}."
            )},
        ]
        samples.append({
            "messages": np.array(messages, dtype=object),
        })

    df = pd.DataFrame(samples)
    df.to_parquet(output_path)
    return output_path


def run_sft_megatron_benchmark(
    model_name: str,
    batch_size: int,
    num_gpus: int,
    sequence_length: int,
    num_steps: int,
    gradient_accumulation: int = 1,
    num_nodes: int = 1,
    tp_degree: int = 1,
    pp_degree: int = 1,
    param_offload: bool = True,
    optimizer_offload: bool = True,
) -> dict:
    """
    Run SFT benchmark via verl's Megatron engine and return metrics.

    Uses verl's sft_trainer_ray with engine=megatron, which provides:
    - Tensor Parallelism (TP) for activation+parameter splitting
    - Pipeline Parallelism (PP) for layer splitting
    - CPU offloading for params/optimizer/gradients
    - Sequence packing via flash_attn (use_remove_padding=True)
    - HF<->Megatron weight conversion via mbridge

    Returns same metrics dict as other SFT benchmark functions.
    """
    project_dir = Path(__file__).parent.parent

    total_gpus = num_gpus * num_nodes
    micro_batch_per_gpu = max(1, batch_size // total_gpus)
    max_token_len_per_gpu = max(8192, batch_size * sequence_length // total_gpus)

    # Use shared filesystem for tmpdir so all nodes can access the data file.
    # /tmp is node-local and won't work for multi-node srun+torchrun.
    shared_tmp = project_dir / "benchmarks" / "results" / ".tmp_megatron"
    shared_tmp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=shared_tmp) as tmpdir:
        tmpdir = Path(tmpdir)
        train_file = create_synthetic_sft_dataset(
            num_samples=batch_size * num_steps,
            output_path=tmpdir / "train.parquet",
        )

        # Hydra overrides for verl SFT trainer (works with both Ray and torchrun)
        hydra_overrides = [
            # Switch engine and optimizer to megatron
            "engine=megatron",
            "optim=megatron",
            # Model
            f"model.path={model_name}",
            "model.use_remove_padding=True",
            "model.enable_gradient_checkpointing=True",
            # Data
            f"data.train_files={train_file}",
            "data.val_files=null",
            f"data.train_batch_size={batch_size}",
            f"data.micro_batch_size_per_gpu={micro_batch_per_gpu}",
            f"data.max_token_len_per_gpu={max_token_len_per_gpu}",
            "data.use_dynamic_bsz=True",
            f"data.max_length={sequence_length}",
            "data.pad_mode=no_padding",
            "data.truncation=error",
            # Megatron engine config
            f"engine.tensor_model_parallel_size={tp_degree}",
            f"engine.pipeline_model_parallel_size={pp_degree}",
            # IMPORTANT: verl's param_offload/optimizer_offload/grad_offload are
            # designed for PPO (swap model between rollout/train phases). For SFT,
            # they cause verl's context manager to load ALL optimizer states to GPU
            # before megatron starts, defeating megatron's native cpu offload.
            # On GH200, pinned memory also consumes HBM. So disable verl's manual
            # offloading and use megatron-core's native optimizer_cpu_offload instead.
            "engine.param_offload=False",
            "engine.optimizer_offload=False",
            "engine.grad_offload=False",
            "engine.sequence_parallel=False",
            "engine.use_remove_padding=True",
            "engine.use_mbridge=True",
            "engine.vanilla_mbridge=True",
            "engine.dtype=bfloat16",
            # Optimizer
            "optim.lr=1e-5",
            "optim.weight_decay=0.01",
            "optim.clip_grad=1.0",
            "optim.lr_decay_style=cosine",
            "optim.lr_warmup_steps_ratio=0.1",
            # Use megatron-core's native optimizer CPU offload to keep Adam states
            # on CPU during the optimizer step. This is the correct way to reduce
            # GPU memory for large models (vs verl's manual offload which loads
            # everything to GPU before training).
            *(
                [
                    "++optim.override_optimizer_config.optimizer_cpu_offload=True",
                    "++optim.override_optimizer_config.optimizer_offload_fraction=1.0",
                    # On GH200, pinned (page-locked) CPU memory is allocated from
                    # the same HBM as GPU memory (unified memory architecture).
                    # Disabling pinning frees GPU HBM at the cost of slower D2H/H2D.
                    "++optim.override_optimizer_config.pin_cpu_grads=False",
                    "++optim.override_optimizer_config.pin_cpu_params=False",
                ]
                if optimizer_offload
                else []
            ),
            # Trainer
            f"trainer.n_gpus_per_node={num_gpus}",
            f"trainer.nnodes={num_nodes}",
            # Avoid checkpoint saving: the verl trainer saves at is_last_step
            # (global_step >= total_training_steps), which causes CPU OOM on
            # 72B from serializing optimizer states. Fix: set total_training_steps
            # very high so is_last_step is never true, and use total_epochs=1
            # with exactly num_steps batches of data so the trainer exits by
            # exhausting the dataloader instead.
            "trainer.total_training_steps=999999",
            "trainer.total_epochs=1",
            "trainer.save_freq=99999",
            "trainer.test_freq=-1",
            "trainer.resume_mode=disable",
            'trainer.logger=["console"]',
            "trainer.project_name=benchmark",
            "trainer.experiment_name=sft_megatron_bench",
            f"trainer.default_local_dir={tmpdir}/checkpoints",
        ]

        # File logger for metrics
        file_logger_path = tmpdir / "sft_metrics.jsonl"

        env = os.environ.copy()
        env["VERL_FILE_LOGGER_PATH"] = str(file_logger_path)
        env["HYDRA_FULL_ERROR"] = "1"
        # Remove ROCR_VISIBLE_DEVICES — verl errors if both ROCR and CUDA visible devices are set.
        # For srun, also unset inside the bash -c command since SLURM re-injects it.
        env.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)

        # Use torchrun (non-Ray) launcher — inherits SLURM's Slingshot VNI
        # assignments, which is required for NCCL P2P (pipeline parallelism)
        # over CXI/Slingshot. Ray workers don't inherit these VNIs properly.
        master_addr = os.environ.get("MASTER_ADDR")
        if not master_addr:
            import subprocess as _sp
            result = _sp.run(
                ["scontrol", "show", "hostnames", os.environ.get("SLURM_JOB_NODELIST", "")],
                capture_output=True, text=True,
            )
            master_addr = result.stdout.strip().split("\n")[0] if result.stdout.strip() else "localhost"
        master_port = os.environ.get("MASTER_PORT", "29500")

        # torchrun args with c10d rendezvous (handles node ranks automatically)
        torchrun_args = [
            f"--nproc_per_node={num_gpus}",
            f"--nnodes={num_nodes}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:{master_port}",
        ]
        trainer_cmd = ["-m", "verl.trainer.sft_trainer"] + hydra_overrides

        if num_nodes > 1:
            # Multi-node: srun launches one torchrun per node.
            # srun propagates SLURM fabric env (VNIs, CXI devices).
            # Each torchrun instance uses $SLURM_NODEID as its node rank.
            # unset ROCR_VISIBLE_DEVICES inside srun — SLURM re-injects it
            inner = "unset ROCR_VISIBLE_DEVICES; " + " ".join(
                ["torchrun"]
                + torchrun_args
                + ["--node_rank=$SLURM_NODEID"]
                + trainer_cmd
            )
            cmd = [
                "srun",
                f"--nodes={num_nodes}",
                "--ntasks-per-node=1",
                "--export=ALL",
                "bash", "-c", inner,
            ]
        else:
            cmd = ["torchrun"] + torchrun_args + ["--node_rank=0"] + trainer_cmd

        print(f"[SFT Megatron] num_nodes={num_nodes}, tp={tp_degree}, pp={pp_degree}, launcher=torchrun")

        start_time = time.perf_counter()

        try:
            # Stream output in real-time (visible in SLURM logs) while also
            # capturing it for OOM detection.  Previous capture_output=True
            # hid everything, making it impossible to diagnose hangs/timeouts.
            log_path = tmpdir / "sft_subprocess.log"
            with GPUMemoryMonitor(poll_interval=1.0) as gpu_monitor, \
                 open(log_path, "w") as log_fh:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    env=env,
                    cwd=project_dir,
                )
                # Stream line-by-line to both stdout and log file
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_fh.write(line)
                proc.wait(timeout=3600)

            total_time = time.perf_counter() - start_time

            combined = log_path.read_text(errors="replace")
            if proc.returncode != 0:
                oom_patterns = [
                    "OutOfMemoryError", "CUDA out of memory",
                    "out of memory", "OOM",
                ]
                if any(pat in combined for pat in oom_patterns):
                    oom_detail = ""
                    for line in combined.splitlines():
                        if "out of memory" in line.lower() or "tried to allocate" in line.lower():
                            oom_detail = line.strip()
                            break
                    raise torch.cuda.OutOfMemoryError(
                        f"CUDA OOM: {oom_detail or combined[-1000:]}"
                    )
                error_lines = []
                for line in combined.splitlines():
                    lower = line.lower()
                    if any(k in lower for k in ["error", "traceback", "assert", "exception", "raise "]):
                        error_lines.append(line.strip())
                error_summary = "\n".join(error_lines[-20:]) if error_lines else combined[-2000:]
                raise RuntimeError(f"SFT Megatron failed:\n{error_summary}")

            metrics = {
                "peak_memory_gb": gpu_monitor.peak_memory_gb,
                "tokens_per_second": None,
                "samples_per_second": None,
                "time_per_step_ms": None,
                "total_time_s": total_time,
                "metrics_source": "estimated",
            }

            # Try to extract metrics from file logger
            if file_logger_path.exists():
                from benchmarks.grpo_benchmark import extract_file_logger_metrics
                file_metrics = extract_file_logger_metrics(file_logger_path)
                if "perf/time_per_step" in file_metrics:
                    metrics["time_per_step_ms"] = file_metrics["perf/time_per_step"] * 1000
                    metrics["metrics_source"] = "file_logger"
                if "perf/throughput" in file_metrics:
                    metrics["tokens_per_second"] = file_metrics["perf/throughput"] * total_gpus

            # Fallback: estimate from wall clock
            if metrics["time_per_step_ms"] is None:
                effective_time = max(total_time - 10, total_time * 0.8)
                time_per_step_s = effective_time / num_steps
                metrics["time_per_step_ms"] = time_per_step_s * 1000

            if metrics["tokens_per_second"] is None and metrics["time_per_step_ms"]:
                time_per_step_s = metrics["time_per_step_ms"] / 1000
                metrics["tokens_per_second"] = batch_size * sequence_length * 0.7 / time_per_step_s

            if metrics["time_per_step_ms"]:
                metrics["samples_per_second"] = batch_size / (metrics["time_per_step_ms"] / 1000)

            return metrics

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError("SFT Megatron benchmark timed out (1 hour)")


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
