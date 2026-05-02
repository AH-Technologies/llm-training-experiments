#!/bin/bash
#SBATCH --job-name=s1-base-nll
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --output=logs/s1-base-nll-%j.out
#SBATCH --error=logs/s1-base-nll-%j.err

# One-off precomputation of per-example base-model NLL on s1K.
# Produces data/s1K/s1k_base_nll.parquet, consumed by the screening
# strategies `base_loss` and `base_logprob_mean`.
#
# Output schema: {index, total_nll, mean_nll, response_tokens}
#   - total_nll: -sum(logP) across response tokens (informativeness)
#   - mean_nll: total_nll / response_tokens (per-token surprise; -ve = coherence)
#
# Single node, TP=4 on Qwen-32B-Instruct, ~10 min wall-clock.
#
# Usage:
#   sbatch scripts/s1/submit_compute_base_nll.slurm

set -e

echo "=========================================="
echo "S1K base-model NLL precomputation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

mkdir -p logs data/s1K

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
module load aws-ofi-nccl/1.14.1-GCCcore-13.3.0-CUDA-12.6.0

PROJECT_DIR=/cluster/projects/nn12068k/alexaau/llm-training-experiments
cd $PROJECT_DIR

if [ -d "$PROJECT_DIR/venv" ]; then
    source $PROJECT_DIR/venv/bin/activate
fi

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# --- HuggingFace cache ---
MODEL_CACHE_KEY="Qwen_Qwen2.5-32B-Instruct"
export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface_${MODEL_CACHE_KEY}
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_DATASETS_CACHE

# --- Slingshot / NCCL / libfabric networking (single-node, but vLLM TP still needs NCCL) ---
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CROSS_NIC=1
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:/usr/lib64:${LD_LIBRARY_PATH}
export CPATH=/opt/cray/libfabric/1.22.0/include:${CPATH}
export NCCL_DEBUG=WARN

# --- CUDA paths ---
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
export LIBRARY_PATH=${MATH_LIBS}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${MATH_LIBS}:${LD_LIBRARY_PATH}

# --- Other ---
export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_RS_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

export PYTHONPATH=${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}

# Sanity: refuse to overwrite a pre-existing parquet without explicit consent.
NLL_PATH="data/s1K/s1k_base_nll.parquet"
if [ -f "$NLL_PATH" ]; then
    echo "WARNING: $NLL_PATH already exists. Delete it first if you want to recompute."
    echo "Existing file kept; exiting cleanly."
    exit 0
fi

if [ ! -f "data/s1K/s1k.parquet" ]; then
    echo "ERROR: data/s1K/s1k.parquet missing."
    exit 1
fi

echo "Running compute_base_nll on Qwen2.5-32B-Instruct (TP=4) ..."
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

python -m s1.pruning.compute_base_nll \
    --input data/s1K/s1k.parquet \
    --output "$NLL_PATH" \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tp 4

echo ""
echo "=========================================="
echo "Quick stats on the produced parquet"
echo "=========================================="
python - <<'PY'
import pyarrow.parquet as pq
import statistics
t = pq.read_table("data/s1K/s1k_base_nll.parquet").to_pydict()
n = len(t["index"])
total = t["total_nll"]
mean = t["mean_nll"]
toks = t["response_tokens"]
def fmt(xs):
    xs = sorted(xs)
    return f"min={xs[0]:.3f} p10={xs[n//10]:.3f} median={statistics.median(xs):.3f} p90={xs[9*n//10]:.3f} max={xs[-1]:.3f}"
print(f"rows: {n}")
print(f"response_tokens: {fmt(toks)}")
print(f"total_nll      : {fmt(total)}")
print(f"mean_nll       : {fmt(mean)}")
zero = sum(1 for x in mean if x == 0.0)
print(f"rows with mean_nll=0 (parse failure or empty response): {zero}")
PY

echo ""
echo "End: $(date)"
echo "=== Done ==="
