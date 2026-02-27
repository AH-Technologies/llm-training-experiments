#!/bin/bash
#SBATCH --job-name=sft-singlenode
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --output=benchmarks/results/sft-singlenode-%j.out
#SBATCH --error=benchmarks/results/sft-singlenode-%j.err

# SFT benchmarks for 0.5B-32B models on single node (4 GPUs)
# 0.5B-14B: FSDP with offload
# 32B: DeepSpeed ZeRO-3 with offload (auto-resolved by benchmark_runner)
# Finds max batch size for each model via binary search

set -x

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
source venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_RS_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
export LIBRARY_PATH=${MATH_LIBS}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${MATH_LIBS}:${LD_LIBRARY_PATH}

echo "=== SFT Single-Node Benchmark (0.5B-32B) ==="
echo "Start time: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

python -m benchmarks.benchmark_runner \
    --models 0.5B 1.5B 3B 7B 14B 32B \
    --training-types sft \
    --num-gpus 4 \
    --sequence-length 2048 \
    --num-steps 10 \
    --find-max-batch \
    --output-dir benchmarks/results

echo ""
echo "End time: $(date)"
echo "=== SFT Single-Node Benchmark Complete ==="
