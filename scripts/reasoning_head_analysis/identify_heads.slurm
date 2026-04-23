#!/bin/bash
#SBATCH --job-name=eap-ig
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/identify_%j.log
#SBATCH --error=logs/identify_%j.err

# EAP-IG Reasoning Head Identification
#
# Usage:
#   sbatch reasoning_head_analysis/slurm/identify_heads.slurm
#   sbatch --export=ALL,MODEL="Qwen/Qwen3-4B" reasoning_head_analysis/slurm/identify_heads.slurm

set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-1.5B-Instruct"}
N_PAIRS=${N_PAIRS:-300}
IG_STEPS=${IG_STEPS:-100}

echo "==========================================="
echo "EAP-IG Reasoning Head Identification"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "Model: $MODEL"
echo "Pairs: $N_PAIRS, IG steps: $IG_STEPS"
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
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH
export NCCL_DEBUG=WARN

if [ -d "$PROJECT_DIR/tiktoken_encodings" ]; then
    export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
fi

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

python -m reasoning_head_analysis.identify_heads \
    --model "$MODEL" \
    --n_pairs "$N_PAIRS" \
    --ig_steps "$IG_STEPS"

echo "Done. Job ID: $SLURM_JOB_ID"
