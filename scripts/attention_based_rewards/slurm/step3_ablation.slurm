#!/bin/bash
#SBATCH --job-name=ablation-val
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=attention_based_rewards/logs/step3_ablation_%j.log

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
source /cluster/projects/nn12068k/haaklau/llm-training-experiments/venv/bin/activate
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export PYTHONPATH=/cluster/projects/nn12068k/haaklau/llm-training-experiments/src:$PYTHONPATH
export PYTHONUNBUFFERED=1

python scripts/attention_based_rewards/step3_ablation.py
