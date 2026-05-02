#!/bin/bash
#SBATCH --job-name=mathneuro
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mathneuro_%j.log
#SBATCH --error=logs/mathneuro_%j.err

# MathNeuro: Identify math-specific attention heads via forward-pass importance scoring.
#
# Uses 4 GPUs (one GH200 node) with data parallelism — each GPU processes
# a shard of math and non-math samples, then results are aggregated.
#
# Usage:
#   sbatch scripts/reasoning_head_analysis/identify_heads_mathneuro.slurm
#
#   sbatch --export=ALL,\
#     MODEL="Qwen/Qwen2.5-Math-1.5B",\
#     OUTPUT_DIR="results/reasoning_head_analysis/identification/mathneuro/base_model" \
#     scripts/reasoning_head_analysis/identify_heads_mathneuro.slurm
#
#   # For a training checkpoint:
#   sbatch --export=ALL,\
#     MODEL="/cluster/projects/nn12068k/.../checkpoint-700",\
#     OUTPUT_DIR="results/reasoning_head_analysis/identification/mathneuro/pi13_step700" \
#     scripts/reasoning_head_analysis/identify_heads_mathneuro.slurm

set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
N_SAMPLES=${N_SAMPLES:-500}
KEEP_RATIO=${KEEP_RATIO:-0.05}
MAX_LENGTH=${MAX_LENGTH:-512}
CONTRAST=${CONTRAST:-"math_vs_reading"}
OUTPUT_DIR=${OUTPUT_DIR:-""}

echo "==========================================="
echo "Neurosurgery Head Identification"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "Model: $MODEL"
echo "Contrast: $CONTRAST"
echo "Samples/domain: $N_SAMPLES"
echo "Keep ratio: $KEEP_RATIO"
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
export NCCL_DEBUG=WARN

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
CMD="python -m reasoning_head_analysis.identify_heads_mathneuro \
    --model \"$MODEL\" \
    --contrast $CONTRAST \
    --n_samples $N_SAMPLES \
    --keep_ratio $KEEP_RATIO \
    --max_length $MAX_LENGTH"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
fi

eval $CMD

echo "Done. Job ID: $SLURM_JOB_ID"
