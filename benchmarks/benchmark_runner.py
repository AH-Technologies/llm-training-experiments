#!/usr/bin/env python3
"""
Benchmark runner for measuring model training capacity and speed on Olivia HPC.

Tests both SFT and GRPO training across different model sizes to determine:
1. Maximum model size that fits in memory
2. Maximum batch size per model
3. Training throughput (tokens/sec, samples/sec)
4. GPU memory utilization
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    timestamp: str
    model_name: str
    model_size_b: float  # billions of parameters
    training_type: str  # "sft" or "grpo"
    num_gpus: int
    num_nodes: int
    batch_size: int
    gradient_accumulation: int
    sequence_length: int
    precision: str

    # Training config
    fsdp_strategy: str = "fsdp2"  # FSDP sharding strategy
    gradient_checkpointing: bool = True  # Memory optimization

    # GRPO-specific config
    actor_offload: bool = True   # CPU offload for actor
    ref_offload: bool = True     # CPU offload for ref model
    rollout_n: int = 8           # Responses per prompt (GRPO)
    gpu_memory_utilization: float = 0.7  # vLLM GPU memory fraction for KV cache

    # Results
    success: bool = False
    error_message: Optional[str] = None
    peak_memory_gb: Optional[float] = None
    tokens_per_second: Optional[float] = None
    samples_per_second: Optional[float] = None
    time_per_step_ms: Optional[float] = None
    total_steps: int = 0
    total_time_s: Optional[float] = None
    metrics_source: Optional[str] = None  # "file_logger", "console", or "estimated"


# Model configurations to benchmark
MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B",
    "1.5B": "Qwen/Qwen2.5-1.5B",
    "3B": "Qwen/Qwen2.5-3B",
    "7B": "Qwen/Qwen2.5-7B",
    "14B": "Qwen/Qwen2.5-14B",
    "32B": "Qwen/Qwen2.5-32B",
    "72B": "Qwen/Qwen2.5-72B",
    "120B": "openai/gpt-oss-120b",
    "235B": "Qwen/Qwen3-235B-A22B",
}

# Starting batch sizes for binary search (will be halved on OOM)
INITIAL_BATCH_SIZES = {
    "0.5B": 64,
    "1.5B": 32,
    "3B": 16,
    "7B": 8,
    "14B": 4,
    "32B": 4,
    "72B": 4,
    "120B": 4,
    "235B": 4,
}

# Batch size limits
MIN_BATCH_SIZE = 4   # verl FSDP2 has issues with batch sizes < 4
MAX_BATCH_SIZE_CAP = 256


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"available": False}

    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        info[f"gpu_{i}"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
        }
    return info


def _is_memory_error(error_message: Optional[str]) -> bool:
    """Check if an error message indicates a memory-related failure (CUDA OOM or vLLM KV cache)."""
    if not error_message:
        return False
    memory_keywords = ["CUDA OOM", "out of memory", "OutOfMemoryError", "cache blocks", "KV cache"]
    return any(kw.lower() in error_message.lower() for kw in memory_keywords)


def find_max_batch_size(
    model_size: str,
    training_type: str,
    num_gpus: int,
    sequence_length: int,
    num_steps: int = 10,
    actor_offload: bool = True,
    ref_offload: bool = True,
    num_nodes: int = 1,
) -> tuple[int, BenchmarkResult]:
    """
    Binary search to find maximum batch size that fits in memory.

    Returns (max_batch_size, benchmark_result)
    """
    # Batch size must be divisible by total number of GPUs (dp_size)
    total_gpus = num_gpus * num_nodes
    min_batch_size = max(MIN_BATCH_SIZE, total_gpus)

    batch_size = max(INITIAL_BATCH_SIZES.get(model_size, 8), min_batch_size)
    last_successful_result = None

    while batch_size >= min_batch_size:
        # Cap at maximum
        if batch_size > MAX_BATCH_SIZE_CAP:
            print(f"Reached max batch size cap ({MAX_BATCH_SIZE_CAP})")
            break

        print(f"\n{'='*60}")
        print(f"Testing {model_size} with batch_size={batch_size}")
        print(f"{'='*60}")

        result = run_single_benchmark(
            model_size=model_size,
            training_type=training_type,
            batch_size=batch_size,
            num_gpus=num_gpus,
            sequence_length=sequence_length,
            num_steps=num_steps,
            num_nodes=num_nodes,
            actor_offload=actor_offload,
            ref_offload=ref_offload,
        )

        if result.success:
            last_successful_result = result
            print(f"SUCCESS: batch_size={batch_size} fits!")
            # Try larger batch size
            batch_size = batch_size * 2
        else:
            is_oom = _is_memory_error(result.error_message)
            if is_oom:
                print(f"OOM: batch_size={batch_size} too large - {result.error_message[:500]}")
            else:
                print(f"FAILED: batch_size={batch_size} - {result.error_message}")
            if last_successful_result:
                break
            if not is_oom:
                # Non-OOM errors (e.g. env issues) won't be fixed by reducing batch size
                break
            # Try smaller batch size
            batch_size = batch_size // 2

    if last_successful_result:
        return last_successful_result.batch_size, last_successful_result

    # Phase 2: OOM fallback — retry with higher gpu_memory_utilization
    is_oom = _is_memory_error(result.error_message)
    if is_oom:
        retry_configs = [
            {"actor_offload": True, "gpu_memory_utilization": 0.85,
             "label": "gpu_mem=0.85"},
        ]

        for retry_cfg in retry_configs:
            print(f"\n{'='*60}")
            print(f"OOM at minimum batch size, retrying with {retry_cfg['label']}...")
            print(f"{'='*60}")

            batch_size = max(INITIAL_BATCH_SIZES.get(model_size, 8), min_batch_size)
            last_successful_result = None

            while batch_size >= min_batch_size:
                if batch_size > MAX_BATCH_SIZE_CAP:
                    print(f"Reached max batch size cap ({MAX_BATCH_SIZE_CAP})")
                    break

                print(f"\n{'='*60}")
                print(f"Testing {model_size} with batch_size={batch_size} ({retry_cfg['label']})")
                print(f"{'='*60}")

                result = run_single_benchmark(
                    model_size=model_size,
                    training_type=training_type,
                    batch_size=batch_size,
                    num_gpus=num_gpus,
                    sequence_length=sequence_length,
                    num_steps=num_steps,
                    num_nodes=num_nodes,
                    actor_offload=retry_cfg["actor_offload"],
                    ref_offload=ref_offload,
                    gpu_memory_utilization=retry_cfg["gpu_memory_utilization"],
                )

                if result.success:
                    last_successful_result = result
                    print(f"SUCCESS: batch_size={batch_size} fits ({retry_cfg['label']})!")
                    batch_size = batch_size * 2
                else:
                    is_oom = _is_memory_error(result.error_message)
                    if is_oom:
                        print(f"OOM: batch_size={batch_size} too large ({retry_cfg['label']}) - {result.error_message[:500]}")
                    else:
                        print(f"FAILED: batch_size={batch_size} - {result.error_message}")
                    if last_successful_result:
                        break
                    if not is_oom:
                        break
                    batch_size = batch_size // 2

            if last_successful_result:
                return last_successful_result.batch_size, last_successful_result

    # Even minimum batch size failed (with and without offloading)
    return 0, result



def run_single_benchmark(
    model_size: str,
    training_type: str,
    batch_size: int,
    num_gpus: int,
    sequence_length: int,
    num_steps: int,
    num_nodes: int = 1,
    gradient_accumulation: int = 1,
    actor_offload: bool = True,
    ref_offload: bool = True,
    gpu_memory_utilization: float = 0.7,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""

    model_name = MODELS[model_size]
    timestamp = datetime.now().isoformat()

    base_result = BenchmarkResult(
        timestamp=timestamp,
        model_name=model_name,
        model_size_b=float(model_size.replace("B", "")),
        training_type=training_type,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        sequence_length=sequence_length,
        precision="bf16",
        actor_offload=actor_offload,
        ref_offload=ref_offload,
        gpu_memory_utilization=gpu_memory_utilization,
        success=False,
        error_message=None,
        peak_memory_gb=None,
        tokens_per_second=None,
        samples_per_second=None,
        time_per_step_ms=None,
        total_steps=num_steps,
        total_time_s=None,
    )

    try:
        if training_type == "sft":
            result = _run_sft_benchmark(
                model_name=model_name,
                batch_size=batch_size,
                num_gpus=num_gpus,
                sequence_length=sequence_length,
                num_steps=num_steps,
                gradient_accumulation=gradient_accumulation,
                num_nodes=num_nodes,
                offload=actor_offload,
            )
        elif training_type == "grpo":
            result = _run_grpo_benchmark(
                model_name=model_name,
                batch_size=batch_size,
                num_gpus=num_gpus,
                sequence_length=sequence_length,
                num_steps=num_steps,
                actor_offload=actor_offload,
                ref_offload=ref_offload,
                num_nodes=num_nodes,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        else:
            raise ValueError(f"Unknown training type: {training_type}")

        # Merge results
        base_result.success = True
        base_result.peak_memory_gb = result.get("peak_memory_gb")
        base_result.tokens_per_second = result.get("tokens_per_second")
        base_result.samples_per_second = result.get("samples_per_second")
        base_result.time_per_step_ms = result.get("time_per_step_ms")
        base_result.total_time_s = result.get("total_time_s")
        base_result.metrics_source = result.get("metrics_source")

        # Update GRPO-specific fields from actual results
        if training_type == "grpo":
            if "rollout_n" in result:
                base_result.rollout_n = result["rollout_n"]

    except torch.cuda.OutOfMemoryError as e:
        base_result.error_message = f"CUDA OOM: {str(e)}"
    except Exception as e:
        base_result.error_message = f"Error: {str(e)}"

    return base_result


def _run_sft_benchmark(
    model_name: str,
    batch_size: int,
    num_gpus: int,
    sequence_length: int,
    num_steps: int,
    gradient_accumulation: int = 1,
    num_nodes: int = 1,
    offload: bool = False,
) -> dict:
    """Run SFT benchmark using the sft_benchmark module."""
    from benchmarks.sft_benchmark import run_sft_benchmark

    return run_sft_benchmark(
        model_name=model_name,
        batch_size=batch_size,
        num_gpus=num_gpus,
        sequence_length=sequence_length,
        num_steps=num_steps,
        gradient_accumulation=gradient_accumulation,
        num_nodes=num_nodes,
        offload=offload,
    )


def _run_grpo_benchmark(
    model_name: str,
    batch_size: int,
    num_gpus: int,
    sequence_length: int,
    num_steps: int,
    actor_offload: bool = True,
    ref_offload: bool = True,
    num_nodes: int = 1,
    gpu_memory_utilization: float = 0.7,
) -> dict:
    """Run GRPO benchmark using the grpo_benchmark module."""
    from benchmarks.grpo_benchmark import run_grpo_benchmark

    return run_grpo_benchmark(
        model_name=model_name,
        batch_size=batch_size,
        num_gpus=num_gpus,
        sequence_length=sequence_length,
        num_steps=num_steps,
        actor_offload=actor_offload,
        ref_offload=ref_offload,
        num_nodes=num_nodes,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def save_results(results: list[BenchmarkResult], output_dir: Path):
    """Save benchmark results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as CSV
    csv_path = output_dir / f"benchmark_results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    print(f"Results saved to {csv_path}")

    # Save as JSON
    json_path = output_dir / f"benchmark_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved to {json_path}")

    return csv_path, json_path


def print_summary(results: list[BenchmarkResult]):
    """Print a summary table of results."""
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY")
    print("="*90)

    # Print common configuration
    if results:
        r = results[0]
        print(f"\nConfiguration:")
        print(f"  Hardware: {r.num_gpus} GPUs × {r.num_nodes} node(s)")
        print(f"  Precision: {r.precision}")
        print(f"  FSDP Strategy: {r.fsdp_strategy}")
        print(f"  Gradient Checkpointing: {'enabled' if r.gradient_checkpointing else 'disabled'}")
        print(f"  Sequence Length: {r.sequence_length}")

        # Check metrics source
        sources = set(r.metrics_source for r in results if r.metrics_source)
        if sources:
            print(f"  Metrics Source: {', '.join(sources)}")

    # Group by training type
    for training_type in ["sft", "grpo"]:
        type_results = [r for r in results if r.training_type == training_type]
        if not type_results:
            continue

        print(f"\n{training_type.upper()} Results:")

        if training_type == "grpo":
            # Show GRPO-specific config
            grpo_results = [r for r in type_results if r.training_type == "grpo"]
            if grpo_results:
                print(f"  Rollout N: {grpo_results[0].rollout_n} responses per prompt")
        print("-"*90)

        if training_type == "grpo":
            print(f"{'Model':<8} {'GPUs':<5} {'Batch':<6} {'Offload':<12} {'Mem(GB)':<9} {'Tok/s':<10} {'ms/step':<9} {'Status'}")
            print("-"*90)

            for r in sorted(type_results, key=lambda x: (x.model_size_b, x.actor_offload)):
                status = "OK" if r.success else "FAIL"
                mem = f"{r.peak_memory_gb:.1f}" if r.peak_memory_gb else "-"
                tps = f"{r.tokens_per_second:.0f}" if r.tokens_per_second else "-"
                ms = f"{r.time_per_step_ms:.1f}" if r.time_per_step_ms else "-"
                offload = "actor+ref" if r.actor_offload else ("ref" if r.ref_offload else "none")
                print(f"{r.model_size_b}B{'':<5} {r.num_gpus:<5} {r.batch_size:<6} {offload:<12} {mem:<9} {tps:<10} {ms:<9} {status}")
        else:
            print(f"{'Model':<8} {'GPUs':<5} {'Batch':<6} {'Offload':<12} {'Mem(GB)':<9} {'Tok/s':<10} {'ms/step':<9} {'Status'}")
            print("-"*90)

            for r in sorted(type_results, key=lambda x: (x.model_size_b, x.actor_offload)):
                status = "OK" if r.success else "FAIL"
                mem = f"{r.peak_memory_gb:.1f}" if r.peak_memory_gb else "-"
                tps = f"{r.tokens_per_second:.0f}" if r.tokens_per_second else "-"
                ms = f"{r.time_per_step_ms:.1f}" if r.time_per_step_ms else "-"
                offload = "cpu" if r.actor_offload else "none"
                print(f"{r.model_size_b}B{'':<5} {r.num_gpus:<5} {r.batch_size:<6} {offload:<12} {mem:<9} {tps:<10} {ms:<9} {status}")



def main():
    parser = argparse.ArgumentParser(description="Benchmark model training on Olivia HPC")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["1.5B"],
        help="Model sizes to benchmark",
    )
    parser.add_argument(
        "--training-types",
        nargs="+",
        choices=["sft", "grpo", "all"],
        default=["sft"],
        help="Training types to benchmark",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes to use",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of training steps per benchmark",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Fixed batch size (if not set, will search for max)",
    )
    parser.add_argument(
        "--find-max-batch",
        action="store_true",
        help="Find maximum batch size via binary search",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--grpo-test-both-offload",
        action="store_true",
        help="For GRPO, test both with and without CPU offloading",
    )
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Disable CPU offloading for actor model (GRPO) or model (SFT). Offloading is enabled by default.",
    )
    parser.add_argument(
        "--no-ref-offload",
        action="store_true",
        help="Disable CPU offloading for reference model (GRPO only)",
    )

    args = parser.parse_args()

    # Expand "all" options
    if "all" in args.models:
        model_sizes = list(MODELS.keys())
    else:
        model_sizes = args.models

    if "all" in args.training_types:
        training_types = ["sft", "grpo"]
    else:
        training_types = args.training_types

    # Determine offload configurations to test
    # Offloading is enabled by default; --no-offload disables it
    actor_offload = not args.no_offload
    ref_offload = not args.no_ref_offload

    # For GRPO, determine which offload configs to test
    if args.grpo_test_both_offload:
        # Test both with and without actor offloading
        grpo_offload_configs = [
            {"actor_offload": False, "ref_offload": True, "label": "no_actor_offload"},
            {"actor_offload": True, "ref_offload": True, "label": "with_actor_offload"},
        ]
    else:
        grpo_offload_configs = [
            {"actor_offload": actor_offload, "ref_offload": ref_offload, "label": "default"}
        ]

    print("="*80)
    print("OLIVIA HPC BENCHMARK")
    print("="*80)
    print(f"Models: {model_sizes}")
    print(f"Training types: {training_types}")
    print(f"GPUs: {args.num_gpus} x {args.num_nodes} nodes")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Steps per benchmark: {args.num_steps}")
    if "grpo" in training_types:
        print(f"GRPO offload configs: {[c['label'] for c in grpo_offload_configs]}")
    print("="*80)

    results = []

    for model_size in model_sizes:
        for training_type in training_types:
            # For GRPO, test each offload config
            if training_type == "grpo":
                offload_configs = grpo_offload_configs
            else:
                # SFT: offload off by default, OOM fallback enables it with FSDP1
                offload_configs = [{"actor_offload": False, "ref_offload": True, "label": "sft"}]

            for offload_config in offload_configs:
                config_label = offload_config["label"]
                print(f"\n>>> Benchmarking {model_size} with {training_type.upper()} ({config_label})")

                if args.find_max_batch:
                    # Step 1: Find maximum batch size
                    max_batch, max_result = find_max_batch_size(
                        model_size=model_size,
                        training_type=training_type,
                        num_gpus=args.num_gpus,
                        sequence_length=args.sequence_length,
                        num_steps=min(args.num_steps, 10),  # Fewer steps for max search
                        actor_offload=offload_config["actor_offload"],
                        ref_offload=offload_config["ref_offload"],
                        num_nodes=args.num_nodes,
                    )
                    print(f"Max batch size for {model_size} ({training_type}, {config_label}): {max_batch}")
                    results.append(max_result)
                else:
                    batch_size = args.batch_size or INITIAL_BATCH_SIZES.get(model_size, 8)
                    result = run_single_benchmark(
                        model_size=model_size,
                        training_type=training_type,
                        batch_size=batch_size,
                        num_gpus=args.num_gpus,
                        sequence_length=args.sequence_length,
                        num_steps=args.num_steps,
                        num_nodes=args.num_nodes,
                        actor_offload=offload_config["actor_offload"],
                        ref_offload=offload_config["ref_offload"],
                    )
                    results.append(result)

    # Save and print results
    if results:
        save_results(results, args.output_dir)
        print_summary(results)


if __name__ == "__main__":
    main()
