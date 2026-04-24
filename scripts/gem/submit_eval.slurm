#!/bin/bash
#SBATCH --job-name=sft1shot-eval
#SBATCH --account=nn12068k
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --output=logs/sft1shot-eval-%j.out
#SBATCH --error=logs/sft1shot-eval-%j.err

# Evaluate SFT 1-shot checkpoints on MATH-500, AIME 2025, AMC 2023
# Uses vLLM for fast batched generation, then plots accuracy curves.
#
# Usage:
#   sbatch scripts/gem/submit_eval.slurm
#
#   # Custom model dirs:
#   MODEL_DIRS="models/sft_1shot_ce models/sft_1shot_gem" sbatch scripts/gem/submit_eval.slurm
#
#   # Only specific checkpoints:
#   CHECKPOINTS="100 300 500 700" sbatch scripts/gem/submit_eval.slurm
#
#   # Include base model comparison:
#   EVAL_BASE=1 sbatch scripts/gem/submit_eval.slurm

set -e

# Configurable
MODEL_DIRS=${MODEL_DIRS:-"models/sft_1shot_ce models/sft_1shot_gem"}
BENCHMARKS=${BENCHMARKS:-"math500 aime_2025 amc_2023"}
CHECKPOINTS=${CHECKPOINTS:-""}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
OUTPUT_DIR=${OUTPUT_DIR:-"results/sft_1shot_eval"}
EVAL_BASE=${EVAL_BASE:-0}
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
NUM_GPUS=${NUM_GPUS:-4}

echo "=========================================="
echo "1-Shot SFT Checkpoint Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Model dirs: $MODEL_DIRS"
echo "Benchmarks: $BENCHMARKS"
echo "Checkpoints: ${CHECKPOINTS:-all}"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Output: $OUTPUT_DIR"
echo "Eval base model: $EVAL_BASE"
echo "=========================================="

mkdir -p logs "$OUTPUT_DIR"

# Load modules
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

# Project directory
PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments

# Activate venv
source $PROJECT_DIR/venv/bin/activate

# Load .env (HF_TOKEN etc.)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Environment
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1

export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8

# vLLM compile cache — use project dir (home dir has issues on compute nodes)
export VLLM_CACHE_ROOT=$PROJECT_DIR/.cache/vllm
mkdir -p "$VLLM_CACHE_ROOT"

# Verify environment
echo ""
echo "Environment check:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
import vllm
print(f'  vLLM: {vllm.__version__}')
"
echo ""

cd $PROJECT_DIR

# Build command — use parallel wrapper with multiple GPUs
CMD="python scripts/gem/eval_parallel.py"
CMD="$CMD --model_dirs $MODEL_DIRS"
CMD="$CMD --num_gpus $NUM_GPUS"
CMD="$CMD --benchmarks $BENCHMARKS"
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --output_dir $OUTPUT_DIR"

if [ -n "$CHECKPOINTS" ]; then
    CMD="$CMD --checkpoints $CHECKPOINTS"
fi

if [ "$EVAL_BASE" = "1" ]; then
    CMD="$CMD --also_eval_base --base_model $BASE_MODEL"
fi

echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Plots saved to: $OUTPUT_DIR/*.png"
