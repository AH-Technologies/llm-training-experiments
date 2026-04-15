#!/bin/bash
# Cluster detection and configuration for multi-cluster support.
# Source this file from any script to get cluster-specific variables.
#
# Detects: Olivia (GH200 ARM, Slingshot) vs IDUN (H100/A100 x86, InfiniBand)
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/cluster_config.sh"

# Skip if already sourced (in this shell)
if [ -n "$_CLUSTER_CONFIG_SOURCED" ]; then
    return 0 2>/dev/null || true
fi
_CLUSTER_CONFIG_SOURCED=1

# --- Cluster detection ---
detect_cluster() {
    if [[ "$(hostname)" == idun-* ]]; then
        echo "idun"
    elif [ "$(uname -m)" = "aarch64" ]; then
        echo "olivia"
    elif module avail NRIS/GPU 2>&1 | grep -q "NRIS/GPU"; then
        echo "olivia"
    else
        echo "idun"
    fi
}

export CLUSTER_NAME=$(detect_cluster)

# --- Per-cluster variables ---
if [ "$CLUSTER_NAME" = "olivia" ]; then
    export SLURM_ACCOUNT_DEFAULT=${SLURM_ACCOUNT_DEFAULT:-"nn12068k"}
    export SLURM_PARTITION_DEFAULT=${SLURM_PARTITION_DEFAULT:-"accel"}
    export DEFAULT_GPUS_PER_NODE=${DEFAULT_GPUS_PER_NODE:-4}
    export DEFAULT_NODES_16GPU=${DEFAULT_NODES_16GPU:-4}
    export DEFAULT_CPUS_PER_TASK=${DEFAULT_CPUS_PER_TASK:-72}
    export DEFAULT_MEM=${DEFAULT_MEM:-"800G"}
    export PROJECT_DIR=${PROJECT_DIR:-"/cluster/projects/nn12068k/alexaau/llm-training-experiments"}
    export HF_CACHE_BASE="/cluster/projects/nn12068k/alexaau/.cache"
    export CLUSTER_MODULES="NRIS/GPU Python/3.12.3-GCCcore-13.3.0 aws-ofi-nccl/1.14.1-GCCcore-13.3.0-CUDA-12.6.0"
    export CUDA_INDEX_URL="https://download.pytorch.org/whl/cu128"
    export CUDA_TAG="cu128"
else
    # IDUN
    export SLURM_ACCOUNT_DEFAULT=${SLURM_ACCOUNT_DEFAULT:-"share-ie-idi"}
    export SLURM_PARTITION_DEFAULT=${SLURM_PARTITION_DEFAULT:-"GPUQ"}
    export DEFAULT_CONSTRAINT=${DEFAULT_CONSTRAINT:-"gpu80g"}
    export DEFAULT_GPUS_PER_NODE=${DEFAULT_GPUS_PER_NODE:-8}
    export DEFAULT_NODES_16GPU=${DEFAULT_NODES_16GPU:-2}
    export DEFAULT_CPUS_PER_TASK=${DEFAULT_CPUS_PER_TASK:-52}
    export DEFAULT_MEM=${DEFAULT_MEM:-"800G"}
    export PROJECT_DIR=${PROJECT_DIR:-"/cluster/home/alexaau/llm-training-experiments"}
    export HF_CACHE_BASE="$PROJECT_DIR/.cache"
    export CLUSTER_MODULES="Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0"
    export CUDA_INDEX_URL="https://download.pytorch.org/whl/cu126"
    export CUDA_TAG="cu126"
fi

# --- Helper: load cluster modules ---
load_cluster_modules() {
    echo "Loading modules for $CLUSTER_NAME..."
    for mod in $CLUSTER_MODULES; do
        module load "$mod"
    done
}

# --- Helper: configure NCCL networking ---
configure_nccl() {
    if [ "$CLUSTER_NAME" = "olivia" ]; then
        # Slingshot / CXI / libfabric networking
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
        export FI_CXI_RDZV_GET_MIN=0
        export FI_CXI_RDZV_THRESHOLD=0
        export FI_CXI_RDZV_EAGER_SIZE=0
        export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:/usr/lib64:${LD_LIBRARY_PATH}
        export CPATH=/opt/cray/libfabric/1.22.0/include:${CPATH}

        # CUDA paths (Olivia aarch64 NVHPC)
        export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
        MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
        export LIBRARY_PATH=${MATH_LIBS}:${LIBRARY_PATH}
        export LD_LIBRARY_PATH=${MATH_LIBS}:${LD_LIBRARY_PATH}
    else
        # IDUN: InfiniBand defaults — NCCL auto-detects IB, minimal config needed
        export CUDA_HOME=${EBROOTCUDA:-/usr/local/cuda}
    fi

    # Common to both clusters
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_DEBUG=WARN
}

echo "Cluster: $CLUSTER_NAME ($(uname -m))"
