#!/usr/bin/env python3
"""Parallel evaluation wrapper — distributes model dirs across GPUs.

Splits model_dirs evenly across available GPUs and runs eval_checkpoints.py
on each GPU in parallel via subprocesses.

Usage:
  python scripts/gem/eval_parallel.py \
      --model_dirs models/sft_1shot_ce_bon_pi1 models/sft_1shot_gem_bon_pi1 ... \
      --num_gpus 4 \
      --benchmarks math500 amc_2023
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Parallel eval across GPUs")
    parser.add_argument("--model_dirs", nargs="+", required=True)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--benchmarks", nargs="+", default=["math500", "amc_2023"])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="results/sft_1shot_eval")
    parser.add_argument("--also_eval_base", action="store_true")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    args = parser.parse_args()

    script = str(Path(__file__).parent / "eval_checkpoints.py")
    os.makedirs(args.output_dir, exist_ok=True)

    # Distribute model dirs across GPUs round-robin
    gpu_assignments: dict[int, list[str]] = {i: [] for i in range(args.num_gpus)}
    for i, model_dir in enumerate(args.model_dirs):
        gpu_assignments[i % args.num_gpus].append(model_dir)

    # Base model goes on GPU 0
    if args.also_eval_base:
        print(f"Base model ({args.base_model}) will be evaluated on GPU 0")

    print(f"GPU assignments:")
    for gpu, dirs in gpu_assignments.items():
        print(f"  GPU {gpu}: {[Path(d).name for d in dirs]}")
    print()

    # Launch subprocesses
    processes = []
    for gpu, dirs in gpu_assignments.items():
        if not dirs and not (gpu == 0 and args.also_eval_base):
            continue

        cmd = [
            sys.executable, script,
            "--model_dirs", *dirs,
            "--benchmarks", *args.benchmarks,
            "--max_new_tokens", str(args.max_new_tokens),
            "--output_dir", args.output_dir,
        ]
        if args.checkpoints:
            cmd += ["--checkpoints", *[str(c) for c in args.checkpoints]]
        if gpu == 0 and args.also_eval_base:
            cmd += ["--also_eval_base", "--base_model", args.base_model]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        log_file = Path(args.output_dir) / f"eval_gpu{gpu}.log"
        print(f"GPU {gpu}: launching with {len(dirs)} model dirs -> {log_file}")
        f = open(log_file, "w")
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        processes.append((gpu, p, f))

    # Wait for all
    failed = []
    for gpu, p, f in processes:
        p.wait()
        f.close()
        if p.returncode != 0:
            failed.append(gpu)
            print(f"GPU {gpu}: FAILED (exit code {p.returncode})")
        else:
            print(f"GPU {gpu}: done")

    if failed:
        print(f"\nFailed GPUs: {failed}")
        print("Check logs in", args.output_dir)
        sys.exit(1)

    print(f"\nAll done! Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
