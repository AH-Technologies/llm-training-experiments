#!/bin/bash
# Setup script for RLVR Grokking environment on IDUN HPC (H100)
#
# IMPORTANT: This must be run on a GPU node (via srun), NOT on login node!
#
# Usage:
#   srun --account=share-ie-idi --time=00:30:00 --partition=GPUQ --gpus=1 \
#        --cpus-per-task=4 --mem-per-cpu=16G ./scripts/setup_env.sh

set -e

echo "=========================================="
echo "RLVR Grokking Environment Setup (IDUN)"
echo "=========================================="
echo "Architecture: $(uname -m)"
echo "Hostname: $(hostname)"
echo ""

# Check we're on x86_64 (GPU node)
if [ "$(uname -m)" != "x86_64" ]; then
    echo "ERROR: This script must be run on a GPU node (x86_64 architecture)!"
    echo "Use: srun --account=share-ie-idi --time=00:30:00 --partition=GPUQ --gpus=1 --cpus-per-task=4 --mem-per-cpu=16G $0"
    exit 1
fi

# Load modules - Python only, NOT PyTorch (we'll install our own)
echo "Loading modules..."
module load Python/3.12.3-GCCcore-13.3.0

echo "Python: $(python3 --version)"

# Set project directory
PROJECT_DIR=/cluster/home/haaklau/code/llm-training-experiments
cd $PROJECT_DIR

# Remove old venv if exists
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Note: PyTorch will be installed automatically by vLLM with the correct version
# vLLM 0.12.0 requires torch 2.9.0+ with CUDA support

# Install verl first, then pin vLLM to compatible version
# verl 0.7.0 was built for vLLM 0.12.0 (requires v1 engine API)
echo "Installing verl..."
pip install verl

echo "Installing vLLM 0.12.0 (compatible with verl 0.7.0)..."
pip install vllm==0.12.0

# Install flash-attn (required by verl for padding utilities)
# Download pre-built wheel for torch 2.9.0 + CUDA 12
echo "Installing flash-attn..."
FLASH_ATTN_WHEEL="flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
curl -L -o /tmp/$FLASH_ATTN_WHEEL \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$FLASH_ATTN_WHEEL"
pip install /tmp/$FLASH_ATTN_WHEEL
rm /tmp/$FLASH_ATTN_WHEEL

# Set HuggingFace cache (reuse shared cache from other project if available)
export HF_HOME=/cluster/projects/itea_lille-idi-epic-studenter/ah-technologies/.cache/huggingface
mkdir -p $HF_HOME

# Download and patch model config to remove flash_attention_2 requirement
echo "Downloading and patching Qwen model config..."
python scripts/patch_model_config.py "Qwen/Qwen2.5-Math-1.5B"

# Download tiktoken encodings for offline use (compute nodes have no internet)
echo "Downloading tiktoken encodings for offline use..."
mkdir -p $PROJECT_DIR/tiktoken_encodings
curl -L -o $PROJECT_DIR/tiktoken_encodings/o200k_base.tiktoken \
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
curl -L -o $PROJECT_DIR/tiktoken_encodings/cl100k_base.tiktoken \
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# Install additional dependencies
echo "Installing additional dependencies..."
pip install pandas pyarrow wandb hydra-core omegaconf

# Install project in editable mode
echo "Installing rlvr-grokking package..."
pip install -e . --no-deps

# Create directories
mkdir -p logs checkpoints data

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Installed versions:"
python -c "
import torch
import vllm
import verl
import ray
import transformers
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()} (version {torch.version.cuda})')
print(f'  vLLM:         {vllm.__version__}')
print(f'  verl:         {verl.__version__}')
print(f'  Ray:          {ray.__version__}')
print(f'  Transformers: {transformers.__version__}')
"
echo ""
echo "To use this environment in future sessions:"
echo "  module load Python/3.12.3-GCCcore-13.3.0"
echo "  source $PROJECT_DIR/venv/bin/activate"
echo ""
echo "To configure wandb (required for logging):"
echo "  srun --account=share-ie-idi --time=00:05:00 --partition=GPUQ --gpus=1 \\"
echo "       --cpus-per-task=4 --mem-per-cpu=8G --pty bash -c '\\"
echo "       source $PROJECT_DIR/venv/bin/activate && wandb login'"
echo ""
echo "To run training:"
echo "  sbatch scripts/submit_training.slurm ber"
echo ""
