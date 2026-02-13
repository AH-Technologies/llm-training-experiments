#!/bin/bash
# Setup script for RLVR Grokking environment on NRIS HPC (GH200 ARM)
#
# IMPORTANT: This must be run on a GPU node (via srun), NOT on login node!
# The GPU nodes are ARM (aarch64) while login nodes are x86_64.
#
# Usage:
#   srun --account=nn12068k --time=00:30:00 --partition=accel --gpus=1 \
#        --cpus-per-task=4 --mem-per-cpu=16G ./scripts/setup_env.sh

set -e

echo "=========================================="
echo "RLVR Grokking Environment Setup"
echo "=========================================="
echo "Architecture: $(uname -m)"
echo "Hostname: $(hostname)"
echo ""

# Check we're on ARM (GPU node)
if [ "$(uname -m)" != "aarch64" ]; then
    echo "ERROR: This script must be run on a GPU node (ARM architecture)!"
    echo "Use: srun --account=nn12068k --time=00:30:00 --partition=accel --gpus=1 --cpus-per-task=4 --mem-per-cpu=16G $0"
    exit 1
fi

# Load modules - Python only, NOT PyTorch (we'll install our own)
echo "Loading modules..."
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

echo "Python: $(python3 --version)"

# Set project directory
PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
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

# Install vLLM 0.12.0 first (has ARM wheels AND is compatible with verl 0.7.0)
# Note: vLLM 0.14.x has API changes that break verl 0.7.0
echo "Installing vLLM 0.12.0..."
pip install vllm==0.12.0

# Reinstall PyTorch with CUDA support - vLLM installs CPU-only version
# Must use --force-reinstall because version numbers match but builds differ
echo "Reinstalling PyTorch 2.9.0 with CUDA 12.8 (ARM compatible)..."
pip install --force-reinstall torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Fix numpy and setuptools versions
echo "Fixing numpy and setuptools versions..."
pip install "numpy>=2.0,<2.3" "setuptools>=77.0.3,<81.0.0"

# Install verl
echo "Installing verl..."
pip install verl

# Install FlashAttention2 from pre-built ARM wheel
echo "Installing FlashAttention2 (pre-built ARM wheel)..."
FLASH_ATTN_WHEEL="flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_aarch64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FLASH_ATTN_WHEEL}"
curl -L -o "/tmp/${FLASH_ATTN_WHEEL}" "$FLASH_ATTN_URL"
pip install "/tmp/${FLASH_ATTN_WHEEL}"
rm -f "/tmp/${FLASH_ATTN_WHEEL}"

# Install additional dependencies
echo "Installing additional dependencies..."
pip install pandas pyarrow wandb hydra-core omegaconf

# Install project in editable mode
echo "Installing rlvr-grokking package..."
pip install -e . --no-deps

# Create directories
mkdir -p logs checkpoints data

# Set HuggingFace cache
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
mkdir -p $HF_HOME

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
echo "  module load NRIS/GPU"
echo "  module load Python/3.12.3-GCCcore-13.3.0"
echo "  source $PROJECT_DIR/venv/bin/activate"
echo ""
echo "To configure wandb (required for logging):"
echo "  srun --account=nn12068k --time=00:05:00 --partition=accel --gpus=1 \\"
echo "       --cpus-per-task=4 --mem-per-cpu=8G --pty bash -c '\\"
echo "       source $PROJECT_DIR/venv/bin/activate && wandb login'"
echo ""
echo "To run training:"
echo "  sbatch scripts/submit_training.slurm pi13_math500"
echo ""
