#!/bin/bash
#SBATCH --job-name=rhythm-dryrun
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --output=attention_sparks_thinking/logs/dryrun_%x_%j.log
#SBATCH --error=attention_sparks_thinking/logs/dryrun_%x_%j.err

# Dry-run: 5 steps for a SINGLE condition.
# Each condition runs in its own SLURM job to avoid GPU conflicts.
#
# Usage:
#   sbatch --export=ALL,RUN_TYPE=A scripts/attention_sparks_thinking/slurm/dry_run.slurm
#   sbatch --export=ALL,RUN_TYPE=B scripts/attention_sparks_thinking/slurm/dry_run.slurm
#   sbatch --export=ALL,RUN_TYPE=C scripts/attention_sparks_thinking/slurm/dry_run.slurm
#
# Or submit all 3 at once:
#   for R in A B C; do sbatch --export=ALL,RUN_TYPE=$R scripts/attention_sparks_thinking/slurm/dry_run.slurm; done

set -e

RUN_TYPE=${RUN_TYPE:?ERROR: RUN_TYPE not set. Use --export=ALL,RUN_TYPE=<A|B|C>}

echo "=========================================="
echo "Attention-Rhythm GRPO Dry Run: $RUN_TYPE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "=========================================="

mkdir -p attention_sparks_thinking/logs

# Load modules
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# Environment
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:$PROJECT_DIR:$PYTHONPATH

# NCCL
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_CROSS_NIC=1
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:/usr/lib64:${LD_LIBRARY_PATH}

# vLLM / Ray
export VLLM_USE_V1=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export WANDB_MODE=online

# Tiktoken
if [ -d "$PROJECT_DIR/tiktoken_encodings" ]; then
    export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
    export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
    export TIKTOKEN_RS_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
fi

# CUDA
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
export LIBRARY_PATH=${MATH_LIBS}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${MATH_LIBS}:${LD_LIBRARY_PATH}
unset ROCR_VISIBLE_DEVICES

echo ""
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

bash scripts/attention_sparks_thinking/dry_run_single.sh
