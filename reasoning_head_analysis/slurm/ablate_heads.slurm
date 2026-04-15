#!/bin/bash
#SBATCH --job-name=ablate-heads
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=reasoning_head_analysis/logs/ablate_%j.log
#SBATCH --error=reasoning_head_analysis/logs/ablate_%j.err

# Reasoning Head Ablation Study
#
# Usage:
#   sbatch reasoning_head_analysis/slurm/ablate_heads.slurm
#   sbatch --export=ALL,MODEL="Qwen/Qwen3-4B",NUM_PROBLEMS=100 reasoning_head_analysis/slurm/ablate_heads.slurm

set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-1.5B-Instruct"}
IMPORTANCE_PATH=${IMPORTANCE_PATH:-"reasoning_head_analysis/results/Qwen_Qwen2.5-1.5B-Instruct/head_importance.pt"}
DATA_PATH=${DATA_PATH:-"data/math500.parquet"}
NUM_PROBLEMS=${NUM_PROBLEMS:-50}

echo "==========================================="
echo "Reasoning Head Ablation Study"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "Model: $MODEL"
echo "Importance: $IMPORTANCE_PATH"
echo "Problems: $NUM_PROBLEMS"
echo "==========================================="

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/alexaau/llm-training-experiments
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

MODEL_CACHE_KEY=$(echo "$MODEL" | tr '/' '_')
export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface_${MODEL_CACHE_KEY}
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_DATASETS_CACHE

export HF_TOKEN=${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export VLLM_USE_V1=1

if [ -d "$PROJECT_DIR/tiktoken_encodings" ]; then
    export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
fi

mkdir -p reasoning_head_analysis/logs

echo ""
echo "Environment check:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

python -m reasoning_head_analysis.ablate_heads \
    --model "$MODEL" \
    --importance_path "$IMPORTANCE_PATH" \
    --data_path "$DATA_PATH" \
    --num_problems "$NUM_PROBLEMS"

echo "Done. Job ID: $SLURM_JOB_ID"
