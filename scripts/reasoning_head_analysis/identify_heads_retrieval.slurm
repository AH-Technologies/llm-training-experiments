#!/bin/bash
#SBATCH --job-name=retrieval-heads
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/retrieval_heads_%j.log
#SBATCH --error=logs/retrieval_heads_%j.err

# Retrieval Head Detection: Identify attention heads that perform copy-paste
# operations during needle-in-a-haystack retrieval (Wu et al., 2024).
#
# Uses 1 GPU with eager attention (not flash) to access attention weights.
#
# Usage:
#   sbatch scripts/reasoning_head_analysis/identify_heads_retrieval.slurm
#
#   sbatch --export=ALL,\
#     MODEL="Qwen/Qwen2.5-Math-1.5B",\
#     OUTPUT_DIR="results/reasoning_head_analysis/identification/retrieval/base_model" \
#     scripts/reasoning_head_analysis/identify_heads_retrieval.slurm

set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
ROUGE_THRESHOLD=${ROUGE_THRESHOLD:-0.5}
OUTPUT_DIR=${OUTPUT_DIR:-""}

echo "==========================================="
echo "Retrieval Head Detection"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "Model: $MODEL"
echo "ROUGE threshold: $ROUGE_THRESHOLD"
echo "==========================================="

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_DATASETS_CACHE

export HF_TOKEN=${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH

mkdir -p logs

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

# Build command
CMD="python -m reasoning_head_analysis.identify_heads_retrieval \
    --model \"$MODEL\" \
    --rouge_threshold $ROUGE_THRESHOLD"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
fi

eval $CMD

echo "Done. Job ID: $SLURM_JOB_ID"
