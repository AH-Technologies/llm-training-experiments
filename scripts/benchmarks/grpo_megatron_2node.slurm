#!/bin/bash
#SBATCH --job-name=meg-grpo-2n
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --output=benchmarks/results/megatron-grpo-2node-%j.out
#SBATCH --error=benchmarks/results/megatron-grpo-2node-%j.err

# Megatron-LM GRPO benchmark: larger Qwen2.5 models on 2 nodes (8 GPUs)
# Models: 32B (TP=4, DP=2)
# Uses Ray cluster across SLURM nodes for verl multi-node coordination.

set -x

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
module load aws-ofi-nccl/1.14.1-GCCcore-13.3.0-CUDA-12.6.0
source venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export HF_HUB_OFFLINE=1

# --- Slingshot / NCCL / libfabric networking ---
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_NET=ofi
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
# NOTE: Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
# vLLM's memory pool is incompatible with expandable segments.
export NCCL_CROSS_NIC=1
# Disable Slingshot eager messages — prevents NCCL hangs (CSCS/Isambard docs)
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:/usr/lib64:${LD_LIBRARY_PATH}
export CPATH=/opt/cray/libfabric/1.22.0/include:${CPATH}
export NCCL_DEBUG=WARN

# Disable Ray compiled DAG — hangs with Slingshot (vLLM #26318)
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_USE_RAY_WRAPPED_PP_COMM=0

export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_RS_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
export LIBRARY_PATH=${MATH_LIBS}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${MATH_LIBS}:${LD_LIBRARY_PATH}
unset ROCR_VISIBLE_DEVICES

# Multi-node Ray cluster setup
# NOTE: Slingshot VNI/SVC_ID credentials are inherited automatically within
# a single srun step. Do NOT capture them separately — each srun step gets
# its own credentials, so cross-step credential sharing causes CXI errors.
HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -i | awk '{print $1}')
RAY_PORT=6379
RAY_TEMP_DIR="/tmp/ray_${SLURM_JOB_ID}"

echo "=== Megatron GRPO - 2 Nodes (8 GPUs) ==="
echo "Start: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Head node: $HEAD_NODE ($HEAD_IP)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -c "import megatron.core; print(f'megatron-core {megatron.core.__version__}')"

# ---- Clean stale Ray state on all nodes ----
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES \
    ray stop 2>/dev/null || true
sleep 5

# ---- Start Ray cluster in a SINGLE srun step ----
# Critical: all Ray processes must share the same srun step so they get
# identical Slingshot VNI/SVC_ID credentials. Separate srun steps get
# different CXI service allocations, causing NCCL "Invalid resource domain"
# errors when actors try cross-node communication.
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

# Verify Ray cluster sees all nodes and GPUs
echo "Ray cluster status (expecting 2 nodes, 8 GPUs):"
ray status --address="$HEAD_IP:$RAY_PORT" || echo "WARNING: ray status failed"
python -c "
import ray
ray.init(address='$HEAD_IP:$RAY_PORT')
resources = ray.cluster_resources()
print(f'Cluster resources: {resources}')
num_gpus = resources.get('GPU', 0)
num_cpus = resources.get('CPU', 0)
print(f'Total GPUs: {num_gpus}, Total CPUs: {num_cpus}')
if num_gpus < 8:
    print(f'WARNING: Expected 8 GPUs but only found {num_gpus}!')
ray.shutdown()
"

export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"

# ---- Background GPU/memory monitor (every 60s) ----
(while true; do
    echo "--- GPU monitor $(date) ---"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
    echo "CPU mem: $(free -g | awk '/^Mem:/{print $3"/"$2" GB"}')"
    sleep 60
done) &
MONITOR_PID=$!

python -m benchmarks.benchmark_runner \
    --models 32B \
    --training-types grpo \
    --num-gpus 4 \
    --num-nodes 2 \
    --sequence-length 2048 \
    --num-steps 5 \
    --find-max-batch \
    --strategy megatron \
    --output-dir benchmarks/results

# Cleanup
echo "Cleaning up..."
kill $MONITOR_PID 2>/dev/null || true
kill $RAY_PIDS 2>/dev/null || true
wait

echo "End: $(date)"
echo "=== Done ==="
