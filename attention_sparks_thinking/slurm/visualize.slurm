#!/bin/bash
#SBATCH --job-name=viz-rhythm
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=attention_sparks_thinking/logs/viz_%j.log
#SBATCH --error=attention_sparks_thinking/logs/viz_%j.err

# Visualize attention rhythm heatmaps for a given model.
#
# Usage:
#   sbatch attention_sparks_thinking/slurm/visualize.slurm
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-4B-Base attention_sparks_thinking/slurm/visualize.slurm

set -e

MODEL=${MODEL:-"Qwen/Qwen3-4B-Base"}
NUM_PROBLEMS=${NUM_PROBLEMS:-5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

echo "==========================================="
echo "Attention Rhythm Visualization"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Num problems: $NUM_PROBLEMS"
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

# Create output directory with model name
SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
OUTPUT_DIR="$PROJECT_DIR/attention_sparks_thinking/logs/viz_${SAFE_MODEL}_${SLURM_JOB_ID}"
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

echo "Running visualization..."
python -m attention_sparks_thinking.scripts.visualize_rhythm \
    --model_name "$MODEL" \
    --num_problems $NUM_PROBLEMS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir "$OUTPUT_DIR" \
    --num_class_prompts 50 \
    --head_quantile 0.3

echo ""
echo "Done! Output saved to: $OUTPUT_DIR"
echo "Job ID: $SLURM_JOB_ID"
ls -la "$OUTPUT_DIR"
