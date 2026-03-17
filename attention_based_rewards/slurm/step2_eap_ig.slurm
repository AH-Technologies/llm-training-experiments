#!/bin/bash
#SBATCH --job-name=eap-ig-circuit
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=attention_based_rewards/logs/step2_eap_ig_%j.log

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
source /cluster/projects/nn12068k/haaklau/llm-training-experiments/venv/bin/activate
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface

python attention_based_rewards/scripts/step2_eap_ig.py
