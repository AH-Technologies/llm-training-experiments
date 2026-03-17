#!/bin/bash
#SBATCH --job-name=hs-variance
#SBATCH --account=nn12068k
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=accel
#SBATCH --output=logs/hs-variance-%j.out
#SBATCH --error=logs/hs-variance-%j.err

set -e

echo "=========================================="
echo "Hidden State Variance Analysis"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

mkdir -p logs

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

echo ""
echo "=== Wang 14 (1.5B model, 256 rollouts) ==="
python compute_hidden_states_standalone.py \
    --results-dir results/entropy_profiles \
    --model Qwen/Qwen2.5-Math-1.5B \
    --batch-size 4

# Li 9 requires Qwen2.5-7B which is not cached locally.
# To run Li: download model first, then:
# python compute_hidden_states_standalone.py \
#     --results-dir results/li_run/entropy_profiles \
#     --model Qwen/Qwen2.5-7B \
#     --batch-size 2

echo ""
echo "Done!"
