#!/bin/bash
#SBATCH --job-name=full-benchmark
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --output=benchmarks/results/full-benchmark-%j.out
#SBATCH --error=benchmarks/results/full-benchmark-%j.err

# Full benchmark suite - tests all model sizes with both SFT and GRPO
# This will take several hours to complete

set -x

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
source venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export WANDB_MODE=disabled

echo "=== Full Benchmark Suite ==="
echo "Start time: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Run full benchmark with optimal batch size search
# Step 1: Find max batch size for each model
# Step 2: Test batch sizes 1,2,4,8,16,... up to max to find optimal throughput
python -m benchmarks.benchmark_runner \
    --models 0.5B 1.5B 3B 7B 14B 32B \
    --training-types sft grpo \
    --num-gpus 4 \
    --sequence-length 2048 \
    --num-steps 20 \
    --find-optimal \
    --output-dir benchmarks/results

echo ""
echo "End time: $(date)"
echo "=== Full Benchmark Complete ==="
