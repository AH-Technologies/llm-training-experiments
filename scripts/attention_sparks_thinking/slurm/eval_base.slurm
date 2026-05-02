#!/bin/bash
#SBATCH --job-name=eval-qwen3-base
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=attention_sparks_thinking/logs/eval_base_%j.out
#SBATCH --error=attention_sparks_thinking/logs/eval_base_%j.err

set -euo pipefail

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

source /cluster/projects/nn12068k/haaklau/llm-training-experiments/venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /cluster/projects/nn12068k/haaklau/llm-training-experiments

python scripts/attention_sparks_thinking/regen_eval_data_and_eval_base.py
