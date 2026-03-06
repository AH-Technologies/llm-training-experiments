#!/bin/bash
#SBATCH --job-name=meg-sft72-4n
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --output=benchmarks/results/megatron-sft-72b-4node-%j.out
#SBATCH --error=benchmarks/results/megatron-sft-72b-4node-%j.err

# Test: Megatron-LM SFT benchmark for Qwen2.5-72B on 4 nodes (16 GPUs)
# TP=4 (intra-node), PP=4 (inter-node) = 16 GPUs total
# 80 layers / 4 PP stages = 20 layers per stage
#
# Uses torchrun (not Ray) so SLURM's Slingshot VNI assignments are inherited
# by all worker processes. Ray workers don't inherit VNIs, causing NCCL P2P
# failures (VNI_NOT_FOUND / ENOTTY) over CXI.

set -x

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
# Use 1.14.1 — compiled against Cray's CXI-enabled libfabric
module load aws-ofi-nccl/1.14.1-GCCcore-13.3.0-CUDA-12.6.0
source venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export HF_HUB_OFFLINE=1

# --- Slingshot / NCCL / libfabric networking ---
# NCCL bootstrap uses raw TCP sockets — must use bond0 (management network).
# Slingshot hsn interfaces do NOT support TCP; data transport uses OFI/CXI plugin automatically.
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
# Disable cuMem host allocations — fails with ENOTTY on GH200/aarch64
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CROSS_NIC=1
# Use Olivia-provided libfabric and CXI libraries (as per Sigma2 docs)
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:/usr/lib64:${LD_LIBRARY_PATH}
export CPATH=/opt/cray/libfabric/1.22.0/include:${CPATH}
export NCCL_DEBUG=WARN
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

# Multi-node setup (torchrun rendezvous)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "=== Megatron SFT 72B - 4 Nodes (16 GPUs) - torchrun ==="
echo "Start: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -c "import megatron.core; print(f'megatron-core {megatron.core.__version__}')"

# ---- Background GPU/memory monitor (every 60s) ----
(while true; do
    echo "--- GPU monitor $(date) ---"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
    echo "CPU mem: $(free -g | awk '/^Mem:/{print $3"/"$2" GB"}')"
    sleep 60
done) &
MONITOR_PID=$!

python -m benchmarks.benchmark_runner \
    --models 72B \
    --training-types sft \
    --num-gpus 4 \
    --num-nodes 4 \
    --sequence-length 2048 \
    --num-steps 5 \
    --find-max-batch \
    --strategy megatron \
    --output-dir benchmarks/results

# Cleanup
echo "Cleaning up..."
kill $MONITOR_PID 2>/dev/null || true
wait

echo "End: $(date)"
echo "=== Done ==="
