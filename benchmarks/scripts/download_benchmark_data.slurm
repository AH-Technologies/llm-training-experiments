#!/bin/bash
#SBATCH --job-name=download-data
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=benchmarks/results/download-%j.out
#SBATCH --error=benchmarks/results/download-%j.err

# Download OpenMathInstruct-2 for SFT benchmarking
# Run this ONCE before running benchmarks

set -x

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/rlvr-for-grokking
cd $PROJECT_DIR

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
source venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface

echo "=== Downloading OpenMathInstruct-2 ==="
echo "This will download ~10k samples for SFT benchmarking"
echo ""

python -m benchmarks.prepare_benchmark_data download \
    --output-dir data/benchmark \
    --num-samples 10000 \
    --tokenizer Qwen/Qwen2.5-1.5B \
    --max-length 4096

echo ""
echo "=== Download Complete ==="
echo "Files saved to data/benchmark/"
