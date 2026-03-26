#!/bin/bash
#SBATCH --job-name=eapig-heads
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=attention_sparks_thinking/logs/eapig_%j.log
#SBATCH --error=attention_sparks_thinking/logs/eapig_%j.err

# EAP-IG-style head discovery + attention heatmap visualization.
#
# Phase 1: Activation patching to find top-20 important heads
# Phase 2: Visualize attention patterns of those heads
#
# Usage:
#   sbatch attention_sparks_thinking/slurm/analyze_heads.slurm
#   sbatch --export=ALL,NUM_PROBLEMS=50 attention_sparks_thinking/slurm/analyze_heads.slurm
#   sbatch --export=ALL,METHOD=patching attention_sparks_thinking/slurm/analyze_heads.slurm

set -e

MODEL=${MODEL:-"Qwen/Qwen3-4B-Base"}
NUM_PROBLEMS=${NUM_PROBLEMS:-100}
METHOD=${METHOD:-"attribution"}
PREFIX_TOKENS=${PREFIX_TOKENS:-50}
VIZ_PROBLEMS=${VIZ_PROBLEMS:-3}
TOP_K=${TOP_K:-20}

echo "==========================================="
echo "EAP-IG Head Discovery + Visualization"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Method: $METHOD"
echo "Num problems (Phase 1): $NUM_PROBLEMS"
echo "Viz problems (Phase 2): $VIZ_PROBLEMS"
echo "==========================================="

mkdir -p attention_sparks_thinking/logs

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# CUDA
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
export LIBRARY_PATH=${MATH_LIBS}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${MATH_LIBS}:${LD_LIBRARY_PATH}
unset ROCR_VISIBLE_DEVICES

# Output directory
SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
OUTPUT_DIR="$PROJECT_DIR/attention_sparks_thinking/logs/eapig_${SAFE_MODEL}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

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

# ── Phase 1: Head discovery ──
echo "======================================="
echo "Phase 1: Activation patching head discovery"
echo "======================================="

python -m attention_sparks_thinking.scripts.analyze_heads_qwen3 \
    --model_name "$MODEL" \
    --num_problems $NUM_PROBLEMS \
    --method $METHOD \
    --prefix_tokens $PREFIX_TOKENS \
    --top_k $TOP_K \
    --output_dir "$OUTPUT_DIR"

IMPORTANCE_PATH="$OUTPUT_DIR/head_importance_qwen3.pt"

if [ ! -f "$IMPORTANCE_PATH" ]; then
    echo "ERROR: Phase 1 did not produce $IMPORTANCE_PATH"
    exit 1
fi

echo ""
echo "======================================="
echo "Phase 2: Top-head attention visualization"
echo "======================================="

python -m attention_sparks_thinking.scripts.visualize_top_heads \
    --model_name "$MODEL" \
    --importance_path "$IMPORTANCE_PATH" \
    --num_problems $VIZ_PROBLEMS \
    --max_new_tokens 512 \
    --num_class_prompts 50 \
    --head_quantile 0.3 \
    --top_k $TOP_K \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "==========================================="
echo "Done! Output saved to: $OUTPUT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "==========================================="
ls -la "$OUTPUT_DIR"
