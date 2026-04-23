#!/bin/bash
#SBATCH --job-name=full-circuit
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=attention_based_rewards/logs/full_circuit_%j.log
#SBATCH --error=attention_based_rewards/logs/full_circuit_%j.err

set -ex

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
source /cluster/projects/nn12068k/haaklau/llm-training-experiments/venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8

cd /cluster/projects/nn12068k/haaklau/llm-training-experiments

python3 attention_based_rewards/scripts/analyze_full_circuit.py
