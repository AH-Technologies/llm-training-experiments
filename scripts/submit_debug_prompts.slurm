#!/bin/bash
#SBATCH --job-name=debug-prompts
#SBATCH --account=nn12068k
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=accel
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/debug-prompts-%j.out
#SBATCH --error=logs/debug-prompts-%j.err

set -e

mkdir -p logs

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
source $PROJECT_DIR/venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

cd $PROJECT_DIR
python3 scripts/debug_prompts.py
