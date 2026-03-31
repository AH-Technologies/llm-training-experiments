#!/bin/bash
#SBATCH --job-name=dl-model
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/dl-model-%j.out
#SBATCH --error=logs/dl-model-%j.err

# Download a HuggingFace model to the project cache.
# Run this before training to avoid proxy issues during multi-node jobs.
#
# Usage:
#   sbatch scripts/download_model.slurm                                    # default: Qwen2.5-32B-Instruct
#   MODEL=Qwen/Qwen2.5-7B sbatch scripts/download_model.slurm             # different model

set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-32B-Instruct"}

echo "=========================================="
echo "Model Download: $MODEL"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

mkdir -p logs

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

# Set HF cache to the same location the training scripts use
MODEL_CACHE_KEY=$(echo "${MODEL}" | tr '/' '_')
export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface_${MODEL_CACHE_KEY}
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

echo "Cache dir: $HF_HOME"

python -c "
from huggingface_hub import snapshot_download
import os

model = '${MODEL}'
print(f'Downloading {model}...')
path = snapshot_download(model)
print(f'Done. Cached at: {path}')

# Verify
size = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, fns in os.walk(path)
    for f in fns
) / 1e9
print(f'Total size: {size:.1f} GB')
"

echo "End: $(date)"
echo "=== Done ==="
