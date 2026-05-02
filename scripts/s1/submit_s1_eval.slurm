#!/bin/bash
#SBATCH --job-name=s1-eval
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/s1-eval-%j.out
#SBATCH --error=logs/s1-eval-%j.err

# Evaluate s1-32B on MATH500, AIME24, GPQA Diamond
#
# Usage:
#   sbatch scripts/s1/submit_s1_eval.slurm                                    # eval latest checkpoint
#   MODEL=checkpoints/s1_sft_qwen32b/checkpoint-epoch5-step315 sbatch scripts/s1/submit_s1_eval.slurm
#   MODEL=Qwen/Qwen2.5-32B-Instruct sbatch scripts/s1/submit_s1_eval.slurm   # baseline

set -e

MODEL=${MODEL:-"checkpoints/s1_sft_qwen32b/checkpoint-epoch5-step315"}
BENCHMARKS=${BENCHMARKS:-"math500 aime24 gpqa"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/s1_eval"}

echo "=========================================="
echo "S1 Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL"
echo "Benchmarks: $BENCHMARKS"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs ${OUTPUT_DIR}

# Load modules
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/alexaau/llm-training-experiments
cd $PROJECT_DIR

if [ -d "$PROJECT_DIR/venv" ]; then
    source $PROJECT_DIR/venv/bin/activate
fi

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# HF cache
MODEL_CACHE_KEY=$(echo "${MODEL}" | tr '/' '_')
export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface_${MODEL_CACHE_KEY}
mkdir -p $HF_HOME

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}
unset ROCR_VISIBLE_DEVICES

nvidia-smi --query-gpu=name,memory.total --format=csv

python -m s1.eval \
    --model ${MODEL} \
    --benchmarks ${BENCHMARKS} \
    --tp 4 \
    --max-tokens 32768 \
    --output-dir ${OUTPUT_DIR}

echo "End: $(date)"
echo "=== Done ==="
