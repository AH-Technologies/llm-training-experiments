#!/bin/bash
#SBATCH --job-name=compare-heads
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=attention_sparks_thinking/logs/compare_heads_%j.log
#SBATCH --error=attention_sparks_thinking/logs/compare_heads_%j.err

set -e

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PROJECT_DIR/src:$PROJECT_DIR:$PYTHONPATH

# Use 1.5B model for fast analysis (same architecture, just smaller)
export MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-Math-1.5B"}

echo "Running head classification comparison: $MODEL_NAME"
python scripts/attention_sparks_thinking/compare_head_classification.py

echo "Done. Outputs in attention_sparks_thinking/analysis/head_classification_comparison/"
