#!/bin/bash
#SBATCH --job-name=rhythm-grpo
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
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
module load aws-ofi-nccl/1.14.1-GCCcore-13.3.0-CUDA-12.6.0

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# NCCL settings (critical for multi-node Slingshot/CXI communication)
export NCCL_DEBUG=WARN
export NCCL_NET=ofi
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_CROSS_NIC=1
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:/usr/lib64:${LD_LIBRARY_PATH}
export CPATH=/opt/cray/libfabric/1.22.0/include:${CPATH}

# vLLM settings
export VLLM_USE_V1=1
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_USE_RAY_WRAPPED_PP_COMM=0

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
# Use single srun step so all nodes share Slingshot VNI/SVC_ID credentials
HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -i | awk '{print $1}')
RAY_PORT=6379
RAY_TEMP_DIR="/tmp/ray_${SLURM_JOB_ID}"

echo "Ray cluster: head=${HEAD_NODE} (${HEAD_IP}:${RAY_PORT}), all nodes: $SLURM_JOB_NODELIST"

# Clean stale Ray state on all nodes
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES \
    ray stop 2>/dev/null || true
sleep 5

# Start Ray cluster in a SINGLE srun step (critical for Slingshot credentials)
echo "Starting Ray cluster (all nodes in single srun step)..."
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --export=ALL \
    bash -c "
        unset ROCR_VISIBLE_DEVICES
        if [ \"\$SLURM_NODEID\" -eq 0 ]; then
            echo \"[Ray] Starting head on \$(hostname) (\$SLURM_NODEID)\"
            ray start --head \
                --node-ip-address='$HEAD_IP' \
                --port=$RAY_PORT \
                --temp-dir='$RAY_TEMP_DIR' \
                --num-cpus=72 --num-gpus=4 --block
        else
            echo \"[Ray] Starting worker on \$(hostname) (\$SLURM_NODEID), waiting for head...\"
            sleep 20
            ray start --address='$HEAD_IP:$RAY_PORT' \
                --temp-dir='$RAY_TEMP_DIR' \
                --num-cpus=72 --num-gpus=4 --block
        fi
    " &
RAY_PIDS=$!
sleep 30

echo "Ray cluster status (expecting 2 nodes, 8 GPUs):"
ray status --address="$HEAD_IP:$RAY_PORT" || echo "WARNING: ray status failed"
python -c "
import ray
ray.init(address='$HEAD_IP:$RAY_PORT')
resources = ray.cluster_resources()
print(f'Cluster resources: {resources}')
num_gpus = resources.get('GPU', 0)
assert num_gpus >= 8, f'Expected 8 GPUs, got {num_gpus}'
print(f'Total GPUs: {num_gpus} -- OK')
ray.shutdown()
"

export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

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
kill $RAY_PIDS 2>/dev/null || true
ray stop 2>/dev/null || true
wait

echo ""
echo "Training complete for run type: ${RUN_TYPE}"
echo "Job ID: $SLURM_JOB_ID"
