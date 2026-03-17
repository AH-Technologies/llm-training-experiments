#!/bin/bash
#SBATCH --job-name=rsr-large
#SBATCH --account=nn12068k
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=accel
#SBATCH --output=logs/rsr-large-%j.out
#SBATCH --error=logs/rsr-large-%j.err

set -e

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
source $PROJECT_DIR/venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export HF_HUB_CACHE=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH

cd $PROJECT_DIR/one_shot_metrics/entropy_profiling

echo "=== Large run (644 examples, 8 rollouts each) ==="
python compute_rsr_standalone.py \
    --results-dir results/large_run/entropy_profiles \
    --model Qwen/Qwen2.5-Math-1.5B \
    --batch-size 8
