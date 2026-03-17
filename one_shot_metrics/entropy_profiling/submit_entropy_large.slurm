#!/bin/bash
#SBATCH --job-name=entropy-large
#SBATCH --account=nn12068k
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=accel
#SBATCH --output=logs/entropy-large-%j.out
#SBATCH --error=logs/entropy-large-%j.err

# Entropy Profiling: 400 stratified examples × 64 rollouts
# Usage: sbatch one_shot_metrics/entropy_profiling/submit_entropy_large.slurm

set -e

echo "=========================================="
echo "Large Entropy Profiling (400 examples)"
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

# Run entropy profiling in large mode
cd $PROJECT_DIR/one_shot_metrics/entropy_profiling

OUTPUT_DIR=$PROJECT_DIR/one_shot_metrics/entropy_profiling/results

python run_entropy_profiling.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --num-rollouts 64 \
    --large --num-examples 400 \
    --batch-size 8 \
    --temperature 0.6 \
    --max-new-tokens 3072 \
    --output-dir $OUTPUT_DIR \
    --seed 42

echo ""
echo "Large entropy profiling complete!"
echo "Results: $OUTPUT_DIR/large_run"
