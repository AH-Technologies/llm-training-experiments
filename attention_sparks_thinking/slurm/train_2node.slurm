#!/bin/bash
#SBATCH --job-name=rhythm-grpo
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --output=attention_sparks_thinking/logs/rhythm_%x_%j.log
#SBATCH --error=attention_sparks_thinking/logs/rhythm_%x_%j.err

# Attention-rhythm GRPO training (2-node, 8 GPUs) for Qwen3-4B.
#
# Usage:
#   sbatch --export=ALL,RUN_TYPE=A attention_sparks_thinking/slurm/train_2node.slurm

set -e

RUN_TYPE=${RUN_TYPE:?ERROR: RUN_TYPE not set. Use --export=ALL,RUN_TYPE=<A|B|C>}

echo "=========================================="
echo "Attention-Rhythm GRPO Training (2-node)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Run Type: $RUN_TYPE"
echo "=========================================="

# Create log directory
mkdir -p attention_sparks_thinking/logs

# Load modules
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

# Project directory
PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR

# Activate venv
source $PROJECT_DIR/venv/bin/activate

# Load .env (WANDB_API_KEY, etc.)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Environment variables
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# NCCL settings
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

# vLLM settings
export VLLM_USE_V1=1

# Allow TaskRunner (num_gpus=0) to see GPUs for eager attention model
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

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

# ── Multi-node Ray cluster setup ──
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | head -1)
RAY_PORT=6379

echo "Ray cluster: head=${HEAD_NODE} (${HEAD_NODE_IP}:${RAY_PORT}), nodes=${NODES[*]}"

# Start Ray head on first node
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head --port=$RAY_PORT \
    --num-gpus=$SLURM_GPUS_ON_NODE \
    --block &
sleep 10

# Start Ray workers on remaining nodes
for ((i=1; i<${#NODES[@]}; i++)); do
    echo "Starting Ray worker on ${NODES[$i]}"
    srun --nodes=1 --ntasks=1 -w "${NODES[$i]}" \
        ray start --address="${HEAD_NODE_IP}:${RAY_PORT}" \
        --num-gpus=$SLURM_GPUS_ON_NODE \
        --block &
    sleep 5
done

# Wait for all workers to join
sleep 10
echo "Ray cluster initialized. Checking resources..."
python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"

export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

# Prepare data if needed
python attention_sparks_thinking/scripts/prepare_data.py

# GPU monitor (every 60s)
(while true; do
    echo "--- GPU monitor $(date) ---"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
    sleep 60
done) &
MONITOR_PID=$!

# Run training
echo "Starting rhythm GRPO training: run_type=${RUN_TYPE}"
bash attention_sparks_thinking/scripts/run_condition.sh ${RUN_TYPE}

# Cleanup
kill $MONITOR_PID 2>/dev/null || true
wait

echo ""
echo "Training complete for run type: ${RUN_TYPE}"
echo "Job ID: $SLURM_JOB_ID"
