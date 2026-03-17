#!/bin/bash
#SBATCH --job-name=pi13-repeat
#SBATCH --account=nn12068k
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=accel
#SBATCH --output=logs/pi13-repeat-%j.out
#SBATCH --error=logs/pi13-repeat-%j.err

set -e

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
source $PROJECT_DIR/venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH
export VLLM_USE_V1=1

cd $PROJECT_DIR/one_shot_metrics/entropy_profiling

echo "=== Phase 1+2: Generate rollouts and compute entropy for pi13_repeat ==="
python compute_pi13_repeat.py

echo ""
echo "=== Phase 3: Compute RSR ==="
python compute_rsr_standalone.py \
    --results-dir results/pi13_repeat/entropy_profiles \
    --model Qwen/Qwen2.5-Math-1.5B \
    --batch-size 4
