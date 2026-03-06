#!/bin/bash
# Setup Megatron-LM dependencies for verl benchmarking on NRIS HPC (GH200 ARM)
#
# Installs megatron-core and transformer-engine on top of the existing venv.
# The existing venv must already have verl, vLLM, and PyTorch installed
# (via setup_env.sh).
#
# IMPORTANT: This must be run on a GPU node (via srun), NOT on login node!
# The GPU nodes are ARM (aarch64) while login nodes are x86_64.
#
# Usage:
#   srun --account=nn12068k --time=01:00:00 --partition=accel --gpus=1 \
#        --cpus-per-task=16 --mem-per-cpu=8G ./scripts/setup_megatron_env.sh

set -e

echo "=========================================="
echo "Megatron-LM Environment Setup"
echo "=========================================="
echo "Architecture: $(uname -m)"
echo "Hostname: $(hostname)"
echo ""

# Check we're on ARM (GPU node)
if [ "$(uname -m)" != "aarch64" ]; then
    echo "ERROR: This script must be run on a GPU node (ARM architecture)!"
    echo "Use: srun --account=nn12068k --time=01:00:00 --partition=accel --gpus=1 --cpus-per-task=16 --mem-per-cpu=8G $0"
    exit 1
fi

# Load modules
echo "Loading modules..."
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

# Set project directory and activate venv
PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR

if [ ! -d "venv" ]; then
    echo "ERROR: venv not found. Run scripts/setup_env.sh first."
    exit 1
fi

source venv/bin/activate
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

# Set CUDA paths for building extensions
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
export LIBRARY_PATH=${MATH_LIBS}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${MATH_LIBS}:${LD_LIBRARY_PATH}

# ---- Step 1: Install megatron-core ----
echo "Installing megatron-core..."
pip install megatron-core

# ---- Step 2: Install transformer-engine ----
echo "Installing transformer-engine..."
# transformer-engine provides fused kernels used by megatron-core
pip install transformer-engine

# ---- Step 3: Install NVIDIA Apex (optional but recommended) ----
echo "Attempting to install NVIDIA Apex..."
APEX_DIR=$(mktemp -d)
cd $APEX_DIR

if git clone --depth 1 https://github.com/NVIDIA/apex.git 2>/dev/null; then
    cd apex
    # Build with CUDA extensions for ARM
    pip install -v --disable-pip-version-check --no-cache-dir \
        --no-build-isolation --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" . 2>&1 || {
        echo "WARNING: Apex build failed. This is optional — megatron-core works without it."
        echo "         Some fused optimizers won't be available."
    }
    cd $PROJECT_DIR
else
    echo "WARNING: Could not clone Apex (no internet?). Skipping."
    echo "         You can install Apex later from a compute node with internet."
fi
rm -rf $APEX_DIR

cd $PROJECT_DIR

# ---- Verification ----
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python3 -c "
import torch
print(f'PyTorch:           {torch.__version__}')
print(f'CUDA available:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:      {torch.version.cuda}')
    print(f'GPU:               {torch.cuda.get_device_name(0)}')

try:
    import megatron.core
    print(f'megatron-core:     {megatron.core.__version__}')
except ImportError as e:
    print(f'megatron-core:     FAILED ({e})')

try:
    import transformer_engine
    print(f'transformer-engine: {transformer_engine.__version__}')
except ImportError as e:
    print(f'transformer-engine: FAILED ({e})')

try:
    import apex
    print(f'apex:              installed')
except ImportError:
    print(f'apex:              not installed (optional)')

import verl
print(f'verl:              {verl.__version__}')

import vllm
print(f'vLLM:              {vllm.__version__}')
"

echo ""
echo "=========================================="
echo "Megatron setup complete!"
echo "=========================================="
echo ""
echo "To run Megatron benchmarks:"
echo "  sbatch benchmarks/scripts/submit_grpo_megatron.slurm"
