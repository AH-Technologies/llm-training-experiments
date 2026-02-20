#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --output=benchmarks/results/slurm-%j.out
#SBATCH --error=benchmarks/results/slurm-%j.err

# Olivia HPC Benchmark Submission Script
# Usage: sbatch benchmarks/scripts/submit_benchmark.slurm [options]
#
# Examples:
#   sbatch benchmarks/scripts/submit_benchmark.slurm --models 1.5B 7B --training-types sft
#   sbatch benchmarks/scripts/submit_benchmark.slurm --models all --training-types all --find-max-batch

set -x

# Project setup
PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR

# Load modules
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

# Activate environment
source venv/bin/activate

# Set environment variables
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_RS_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
# Note: WANDB_MODE is set to "offline" inside the benchmark scripts for metric extraction

# GPU info
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run benchmark with passed arguments (or defaults)
BENCHMARK_ARGS="${@:---models 1.5B --training-types sft --num-steps 50}"

echo "=== Running Benchmark ==="
echo "Arguments: $BENCHMARK_ARGS"
echo ""

python -m benchmarks.benchmark_runner $BENCHMARK_ARGS

echo ""
echo "=== Benchmark Complete ==="
