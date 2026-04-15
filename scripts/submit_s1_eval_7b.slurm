#!/bin/bash
#SBATCH --job-name=s1-eval-7b
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --constraint=v100&gpu32g
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --output=logs/s1-eval-7b-%j.out
#SBATCH --error=logs/s1-eval-7b-%j.err

# Compare Qwen2.5-7B-Instruct vs s1.1-7B on MATH500, AIME24, GPQA Diamond
#
# Usage:
#   sbatch scripts/submit_s1_eval_7b.slurm

set -e

echo "=========================================="
echo "S1 7B Model Comparison"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs

# --- Environment setup ---
source /cluster/home/alexaau/llm-training-experiments/scripts/cluster_config.sh
load_cluster_modules

PROJECT_DIR=/cluster/home/alexaau/llm-training-experiments
cd $PROJECT_DIR

if [ -d "$PROJECT_DIR/venv" ]; then
    source $PROJECT_DIR/venv/bin/activate
fi

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

export HF_HOME=$PROJECT_DIR/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}

nvidia-smi --query-gpu=name,memory.total --format=csv

BENCHMARKS=${BENCHMARKS:-"math500 aime24 gpqa"}
MAX_TOKENS=30000

# --- Model 1: Qwen2.5-7B-Instruct (baseline) ---
echo ""
echo "=========================================="
echo "Evaluating: Qwen/Qwen2.5-7B-Instruct"
echo "=========================================="

python -m s1.eval \
    --model Qwen/Qwen2.5-7B-Instruct \
    --benchmarks ${BENCHMARKS} \
    --tp 2 \
    --max-tokens ${MAX_TOKENS} \
    --output-dir eval_results/qwen25_7b_instruct

# --- Model 2: s1.1-7B ---
echo ""
echo "=========================================="
echo "Evaluating: simplescaling/s1.1-7B"
echo "=========================================="

python -m s1.eval \
    --model simplescaling/s1.1-7B \
    --benchmarks ${BENCHMARKS} \
    --tp 2 \
    --max-tokens ${MAX_TOKENS} \
    --output-dir eval_results/s1.1_7b

# --- Summary ---
echo ""
echo "=========================================="
echo "Both evaluations complete."
echo "Results in: eval_results/"
echo "End time: $(date)"
echo "=========================================="
