#!/bin/bash
#SBATCH --job-name=incr-ablation
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=attention_based_rewards/logs/step3b_incremental_%j.log

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
source /cluster/projects/nn12068k/haaklau/llm-training-experiments/venv/bin/activate
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export PYTHONUNBUFFERED=1

python attention_based_rewards/scripts/step3b_incremental_ablation.py
