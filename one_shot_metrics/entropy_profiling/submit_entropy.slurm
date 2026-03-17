#!/bin/bash
#SBATCH --job-name=entropy-profile
#SBATCH --account=nn12068k
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=accel
#SBATCH --output=logs/entropy-%j.out
#SBATCH --error=logs/entropy-%j.err

# Per-Token Entropy Profiling for One-Shot RLVR Examples
# Usage: sbatch one_shot_metrics/entropy_profiling/submit_entropy.slurm

set -e

echo "=========================================="
echo "Entropy Profiling for Wang et al. Examples"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "=========================================="

# Create log directory
mkdir -p logs

# Load modules
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

# Project directory
PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments

# Activate virtual environment
if [ -d "$PROJECT_DIR/venv" ]; then
    source $PROJECT_DIR/venv/bin/activate
elif [ -d "$HOME/venv" ]; then
    source $HOME/venv/bin/activate
fi

# Environment variables
# The Math-1.5B model is cached at .cache/huggingface/ (models--* dirs at top level),
# not under hub/. HF_HUB_CACHE tells huggingface_hub where to find models--* directly.
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export HF_HUB_CACHE=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH

# Verify environment
echo ""
echo "Environment check:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

# Run entropy profiling
cd $PROJECT_DIR/one_shot_metrics/entropy_profiling

OUTPUT_DIR=$PROJECT_DIR/one_shot_metrics/entropy_profiling/results

python run_entropy_profiling.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --num-rollouts 256 \
    --batch-size 8 \
    --temperature 0.6 \
    --max-new-tokens 3072 \
    --output-dir $OUTPUT_DIR \
    --seed 42

echo ""
echo "Entropy profiling complete!"
echo "Results: $OUTPUT_DIR"
