#!/bin/bash
# Unified environment setup for both Olivia (GH200 ARM) and IDUN (H100/A100 x86).
#
# IMPORTANT: Must be run on a GPU node (via srun), NOT on login node!
#
# Usage:
#   # Olivia:
#   srun --account=nn12068k --time=00:30:00 --partition=accel --gpus=1 \
#        --cpus-per-task=4 --mem-per-cpu=16G bash scripts/setup_env.sh
#
#   # IDUN:
#   srun --account=share-ie-idi --time=00:30:00 --partition=GPUQ --gpus=1 \
#        --cpus-per-task=4 --mem-per-cpu=16G bash scripts/setup_env.sh
#
# Options:
#   --s1-only    Skip vLLM/verl/ray (not needed for S1 SFT)
#   --full       Install the full RLVR stack (vLLM, verl, ray) [default]

set -e

# Parse flags
S1_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --s1-only) S1_ONLY=true ;;
        --full)    S1_ONLY=false ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster_config.sh"

echo "=========================================="
echo "Environment Setup ($CLUSTER_NAME)"
echo "=========================================="
echo "Architecture: $(uname -m)"
echo "Hostname: $(hostname)"
echo "S1-only mode: $S1_ONLY"
echo ""

# Check we're on a GPU node
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: This script must be run on a GPU node!"
    if [ "$CLUSTER_NAME" = "olivia" ]; then
        echo "Use: srun --account=nn12068k --time=00:30:00 --partition=accel --gpus=1 --cpus-per-task=4 --mem-per-cpu=16G $0"
    else
        echo "Use: srun --account=share-ie-idi --time=00:30:00 --partition=GPUQ --gpus=1 --cpus-per-task=4 --mem-per-cpu=16G $0"
    fi
    exit 1
fi

# Load modules
load_cluster_modules
echo "Python: $(python3 --version)"

cd "$PROJECT_DIR"

# Remove old venv if exists
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

# IDUN needs CUDA module for nvcc
if [ "$CLUSTER_NAME" = "idun" ]; then
    export CUDA_HOME=$EBROOTCUDA
fi

if [ "$S1_ONLY" = "false" ]; then
    # Full stack: install vLLM first, then force-reinstall PyTorch with CUDA
    echo "Installing vLLM 0.12.0..."
    pip install vllm==0.12.0

    echo "Reinstalling PyTorch 2.9.0 with CUDA ($CUDA_TAG)..."
    pip install --force-reinstall torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
        --index-url "$CUDA_INDEX_URL"

    echo "Fixing numpy and setuptools versions..."
    pip install "numpy>=2.0,<2.3" "setuptools>=77.0.3,<81.0.0"

    echo "Installing verl..."
    pip install verl
else
    # S1-only: just PyTorch + torchtune deps
    echo "Installing PyTorch 2.9.0 with CUDA ($CUDA_TAG)..."
    pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
        --index-url "$CUDA_INDEX_URL"

    echo "Installing torchtune + torchao..."
    pip install torchao==0.9.0 torchtune
fi

# Install FlashAttention2 from pre-built wheel (arch-aware)
echo "Installing FlashAttention2..."
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    FA_WHEEL="flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
else
    FA_WHEEL="flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_aarch64.whl"
fi
curl -L -o "/tmp/$FA_WHEEL" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$FA_WHEEL"
pip install "/tmp/$FA_WHEEL"
rm -f "/tmp/$FA_WHEEL"

# Download tiktoken encodings for offline use (compute nodes may have no internet)
echo "Downloading tiktoken encodings for offline use..."
mkdir -p "$PROJECT_DIR/tiktoken_encodings"
curl -L -o "$PROJECT_DIR/tiktoken_encodings/o200k_base.tiktoken" \
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
curl -L -o "$PROJECT_DIR/tiktoken_encodings/cl100k_base.tiktoken" \
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# Install additional dependencies
echo "Installing additional dependencies..."
pip install pandas pyarrow wandb hydra-core omegaconf

# Install project in editable mode
echo "Installing project package..."
pip install -e . --no-deps

# Create directories
mkdir -p logs checkpoints data

# Set HuggingFace cache
export HF_HOME=${HF_CACHE_BASE}/huggingface
mkdir -p "$HF_HOME"

echo ""
echo "=========================================="
echo "Setup complete! ($CLUSTER_NAME)"
echo "=========================================="
echo ""
echo "Installed versions:"
if [ "$S1_ONLY" = "false" ]; then
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
else
    python -c "
import torch
import transformers
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()} (version {torch.version.cuda})')
print(f'  Transformers: {transformers.__version__}')
"
fi
echo ""
echo "To use this environment in future sessions:"
echo "  source $PROJECT_DIR/venv/bin/activate"
echo ""
echo "To submit S1 SFT training:"
echo "  bash scripts/submit_s1_sft.sh"
echo "  bash scripts/submit_s1_sft_grokking.sh"
echo ""
