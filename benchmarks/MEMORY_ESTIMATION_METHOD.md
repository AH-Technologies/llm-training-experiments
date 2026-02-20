# LLM Training Memory Estimation: Methodology & Formulas

## 1. Goals

Estimate **peak GPU memory** for:
- **SFT** (Supervised Fine-Tuning)
- **GRPO** (Group Relative Policy Optimization, via the verl library)

across arbitrary model sizes (0.5B–1000B+ parameters), with and without FSDP sharding, assuming CPU offloading is always available (GH200 unified memory with 900 GB/s NVLink-C2C bandwidth).

---

## 2. Configurable Parameters

| Symbol | Description | Typical Values |
|--------|-------------|----------------|
| `P` | Total model parameters | 0.5B – 1000B |
| `h` | Hidden dimension | 896 – 18432 |
| `L` | Number of transformer layers | 24 – 126 |
| `a` | Number of attention heads | 14 – 128 |
| `V` | Vocabulary size | 32K – 152K |
| `N` | Number of GPUs (FSDP world size) | 1 – 256 |
| `B` | Micro-batch size per GPU | 1 – 128 |
| `S` | Total sequence length (prompt + response) | 512 – 8192 |
| `S_p` | Prompt length (GRPO) | 256 – 2048 |
| `S_r` | Response length (GRPO) | 256 – 4096 |
| `G` | GRPO rollout group size (responses per prompt) | 4 – 16 |
| `GC` | Gradient checkpointing enabled | True / False |
| `bp` | Bytes per parameter (bf16=2, fp32=4) | 2 |

### Architecture Reference Table

These are **verified values** from official model config.json files on HuggingFace (retrieved Feb 2026):

| Family | Size | P (B) | h | L | a | a_kv | d_ff | V |
|--------|------|-------|------|-----|-----|------|-------|--------|
| **Qwen2.5** | 0.5B | 0.49 | 896 | 24 | 14 | 2 | 4864 | 151936 |
| | 1.5B | 1.54 | 1536 | 28 | 12 | 2 | 8960 | 151936 |
| | 3B | 3.09 | 2048 | 36 | 16 | 2 | 11008 | 151936 |
| | 7B | 7.62 | 3584 | 28 | 28 | 4 | 18944 | 152064 |
| | 14B | 14.77 | 5120 | 48 | 40 | 8 | 13824 | 152064 |
| | 32B | 32.76 | 5120 | 64 | 40 | 8 | 27648 | 152064 |
| | 72B | 72.71 | 8192 | 80 | 64 | 8 | 29568 | 152064 |
| **Llama 3.x** | 1B | 1.24 | 2048 | 16 | 32 | 8 | 8192 | 128256 |
| | 3B | 3.21 | 3072 | 28 | 24 | 8 | 8192 | 128256 |
| | 8B | 8.03 | 4096 | 32 | 32 | 8 | 14336 | 128256 |
| | 70B | 70.55 | 8192 | 80 | 64 | 8 | 28672 | 128256 |
| | 405B | 405.0 | 16384 | 126 | 128 | 8 | 53248 | 128256 |

Where: `h` = hidden_size, `L` = num_layers, `a` = attention heads, `a_kv` = KV heads (GQA), `d_ff` = intermediate/FFN size, `V` = vocab_size.

### Architecture Scaling Heuristics

When only `P` is known (and no specific model is selected), we use interpolation from the table above. Key observed patterns:

**1. Hidden dimension scales roughly as `h ∝ P^0.4`:**

```
h ≈ 1500 × (P_billions)^0.4      (rough fit for 1B–400B range)
```

Spot-check: P=7B → h ≈ 1500 × 7^0.4 ≈ 1500 × 2.43 ≈ 3645 (actual Qwen 7B: 3584, Llama 8B: 4096) ✓

**2. Number of layers scales as `L ∝ P^0.35`:**

```
L ≈ 16 × (P_billions)^0.35       (rough fit)
```

Spot-check: P=70B → L ≈ 12 × 70^0.35 ≈ 12 × 6.65 ≈ 80 (actual: 80) ✓

**3. Attention heads: `a = h / d_head` where `d_head` is typically 64 or 128:**

Most modern models use `d_head = 128`:
```
a ≈ h / 128
```

For smaller models (< 3B), `d_head = 64` is common.

**4. KV heads (GQA ratio):**

- Qwen family: `a_kv = 2` for ≤3B, scales up to 8 for larger models
- Llama 3 family: consistently `a_kv = 8` across all sizes
- **Default heuristic**: `a_kv = max(2, min(8, a / 8))`

**5. FFN intermediate dimension:**

Typically `d_ff ≈ 2.5h to 3.5h` (with SwiGLU, the effective expansion is 2/3 of this due to the gate projection). A reasonable default:
```
d_ff ≈ round_to_256(3.5 × h)
```

Some models (Qwen 14B, 32B) deviate significantly from this due to architectural choices.

**6. Vocabulary size:**

| Family | V |
|--------|------|
| Qwen 2 / 2.5 | 151936–152064 |
| Llama 3 | 128256 |
| Mistral / Mixtral | 32000–32768 |
| DeepSeek V2/V3 | 102400 |
| Gemma 2 | 256000 |
| Generic default | 128000 |

**Embedding parameters** (not included in layer count heuristics):
```
P_embedding = V × h    (often tied input/output embeddings)
```

For large vocabularies this can be significant: V=152K, h=8192 → P_emb ≈ 1.25B params.

>**Note**: We use the vocab size of Qwen for the calcualtions.

> **Note**: For MoE models, `P` refers to total parameters but only a fraction is active per token. This methodology focuses on **dense** models unless stated otherwise. For MoE, substitute `P_active` (the active parameter count per token) in activation calculations, but use `P_total` for model state calculations.

---

## 3. Precision & Bytes

We assume **mixed-precision training** (the standard for modern LLM training):
- Forward/backward computation: **bf16** (2 bytes per value)
- Optimizer states and master weights: **fp32** (4 bytes per value)

| Component | Precision | Bytes per parameter |
|-----------|-----------|---------------------|
| Model weights (training) | bf16 | 2 |
| Gradients | bf16 | 2 |
| Optimizer master weights | fp32 | 4 |
| Optimizer momentum (Adam β₁) | fp32 | 4 |
| Optimizer variance (Adam β₂) | fp32 | 4 |
| **Total "model states"** | mixed | **16** |

---

## 4. Memory Components Overview

Total GPU memory = **Model States** + **Activations** + **Temporary Buffers** + **Framework Overhead**

### 4.1 Model States (per parameter)

Without any sharding or offloading:

```
M_states = P × 16 bytes
         = P × (2 + 2 + 4 + 4 + 4) bytes
              weights  grads  master  momentum  variance
```

For a 7B model: `7 × 10⁹ × 16 = 112 GB`

### 4.2 Activation Memory

Activations are the intermediate tensors saved during the forward pass for use in the backward pass. This is where batch size and sequence length have the biggest impact.

#### Without Gradient Checkpointing

Per transformer layer, the saved activations include (in bf16, 2 bytes each):

| Saved Tensor | Shape | Size (bytes) |
|-------------|-------|--------------|
| Input to layer norm | `B × S × h` | `2BSh` |
| Attention input (Q, K, V) | `3 × B × S × h` | `6BSh` |
| Attention weights (softmax output) | `B × a × S × S` | `2BaS²` |
| Attention output | `B × S × h` | `2BSh` |
| Post-attention residual | `B × S × h` | `2BSh` |
| MLP input | `B × S × h` | `2BSh` |
| MLP intermediate (up + gate) | `2 × B × S × 4h` | `16BSh` |
| MLP output | `B × S × h` | `2BSh` |

**Per-layer activation memory (approximate)**:

```
A_layer ≈ BSh × (32 + 2aS/h) bytes     [Eq. 1]
```

The `2aS²/h` term is the attention score matrix. For long sequences with many heads, this can dominate.

**Simplified per-layer (ignoring attention scores for short sequences)**:

```
A_layer ≈ 32 × B × S × h bytes          [Eq. 1a]
```

**Total activation memory** (all layers):

```
A_total = L × A_layer                     [Eq. 2]
```

#### With Gradient Checkpointing (per-layer)

When gradient checkpointing is applied per layer (the standard in verl):
- Only the **input activation** of each layer is saved: `2BSh` bytes per layer
- During backward, each layer's internal activations are **recomputed** from the saved input
- Peak activation = stored boundaries + recomputation peak for one layer
- **Important**: During backward recomputation, the GPU briefly holds *both* the checkpointed layer input *and* the recomputed intermediate activations simultaneously. The true peak is therefore closer to **2 × A_layer**, not 1 × A_layer.

```
A_checkpoint = L × 2BSh + 2 × A_layer         [Eq. 3]
             = 2LBSh + 2 × BSh(32 + 2aS/h)
             ≈ BSh × (2L + 64 + 4aS/h)
```

**This is a major memory saving.** For Llama-3-8B (L=32, h=4096, a=32) with B=4, S=4096:
- Without checkpointing: ~L × B × S × h × 32 = 32 × 4 × 4096 × 4096 × 32 ≈ 69 GB
- With checkpointing: ~B × S × h × (2L + 64 + 4aS/h) = 4 × 4096 × 4096 × (64 + 64 + 128) ≈ 17.2 GB

> **We always assume gradient checkpointing is enabled** in our estimates, as it is standard practice for training models above ~1B parameters.

### 4.3 Temporary Buffers & Framework Overhead

These include:
- **All-reduce / all-gather communication buffers**: ~`2P/N` to `2P` bytes (FSDP all-gather)
- **Loss computation buffers**: `B × S × V × 2` bytes (logits in bf16)
- **CUDA memory fragmentation**: typically 5–15% overhead
- **PyTorch allocator overhead**: ~200–500 MB constant

We model this as:

```
M_overhead = α × (M_states + A) + M_logits + 500 MB       [Eq. 4]

M_logits = 2 × B × S × V bytes                            [Eq. 4a]
```

Where `α` is the fragmentation/overhead factor:
- **α = 0.15** (recommended default) — accounts for CUDA memory fragmentation, allocator overhead, and small temporary tensors
- **α = 0.20** for large distributed jobs with dynamic shapes (e.g., variable-length sequences in GRPO), where fragmentation can be significantly worse
- **α = 0.10** only for small single-GPU runs with fixed shapes

> **Note**: The logits tensor (`B × S × V`) can be significant. For V=152K, B=4, S=4096: `4 × 4096 × 152000 × 2 ≈ 4.7 GB`. verl uses chunked loss computation to mitigate this, but it's still a consideration.

---

## 5. SFT Memory Estimation

### 5.1 Without FSDP (Single GPU)

```
M_SFT = M_states + A_checkpoint + M_overhead              [Eq. 5]
      = 16P + BSh(2L + 32 + 2aS/h) + overhead
```

### 5.2 With FSDP (N GPUs, no offloading)

FSDP shards model states across N GPUs:

```
M_states_fsdp = 16P / N                                    [Eq. 6]
```

**But**: during forward/backward, FSDP performs **all-gather** to reconstruct full parameters for each FSDP unit. With `reshard_after_forward=True`, only **one FSDP unit** is fully materialized at a time:

```
M_allgather_peak = 2 × P_unit                              [Eq. 7]
```

Where `P_unit` is the number of parameters in the largest FSDP wrapping unit. With `min_num_params=0` (wrap every module), P_unit ≈ parameters of largest single layer ≈ `12h²` (for a standard transformer layer with 4× MLP expansion).

Activations are **NOT sharded** — each GPU computes its own micro-batch:

```
M_SFT_FSDP = 16P/N + A_checkpoint(B_micro) + 2×P_unit + M_overhead   [Eq. 8]
```

Where `B_micro = B_global / N` (micro-batch size per GPU).

### 5.3 With FSDP + CPU Offloading

**Optimizer offloading** (most impactful):
- Moves master weights, momentum, variance to CPU: saves `12P/N` bytes on GPU
- GPU retains: bf16 weights `2P/N` + bf16 gradients `2P/N`

```
M_SFT_FSDP_optim_offload = 4P/N + A_checkpoint + 2×P_unit + M_overhead   [Eq. 9]
```

**Full param + optimizer offloading**:
- All model states on CPU, loaded on demand
- GPU only needs: current FSDP unit params + activations + gradients for current unit

```
M_SFT_FSDP_full_offload = 2×P_unit + A_checkpoint + grad_buffer + M_overhead   [Eq. 10]
```

This is extremely memory-efficient but slower (though GH200's C2C bandwidth mitigates the speed penalty significantly).

---

## 6. GRPO Memory Estimation (verl)

### 6.1 verl Phase-Based Execution

verl runs GRPO in **sequential phases**, swapping models on/off GPU between phases. The peak GPU memory is determined by the **most expensive phase**, including the weight sync transition:

```
M_GRPO = max(M_rollout, M_ref, M_actor_update, M_weight_sync)   [Eq. 11]
```

#### Phase 1: Rollout Generation (vLLM)

vLLM generates `G` responses per prompt using the current policy.

```
M_rollout = M_vllm_weights + M_kv_cache + M_vllm_overhead   [Eq. 12]
```

- **vLLM model weights** (bf16): `2P` bytes (full model, no sharding during inference)
- **KV cache**: Controlled by `gpu_memory_utilization` (default 0.7). vLLM allocates:
  ```
  M_kv_cache = gpu_mem_util × GPU_total - M_vllm_weights - M_vllm_overhead   [Eq. 12a]
  ```

  Per-token KV cache size (per layer):
  ```
  kv_per_token_per_layer = 2 × 2 × h_kv bytes              [Eq. 12b]
                           K+V  bf16  kv_head_dim
  ```
  Where `h_kv = h / a × a_kv` (for GQA models, `a_kv < a`). For MHA, `h_kv = h`.

  Max concurrent tokens in KV cache:
  ```
  max_tokens = M_kv_cache / (L × kv_per_token_per_layer)   [Eq. 12c]
  ```

- **vLLM overhead**: ~1–2 GB for framework, CUDA graphs, etc.

> **Critical: FSDP vs TP asymmetry.** vLLM does **not** support FSDP — it only supports tensor parallelism (TP). This means the rollout phase shards across `T` GPUs (TP degree), while the actor update phase shards across `N` GPUs (FSDP world size). Typically `T << N`. For very large models, this makes the rollout phase the **hard memory bottleneck**: you need `2P/T` bytes per GPU for weights alone, and T is usually limited to 1–8. Even if you have 64 GPUs for FSDP, the rollout still only splits across T of them. This is the key constraint that determines the minimum TP degree for a given model size.

Per-GPU rollout memory with TP:
```
M_rollout_per_gpu = (2P + M_kv_cache) / T + overhead       [Eq. 12d]
```

**Automatic TP degree selection:**

  When `--tp-degree auto` (the default), the estimator finds the minimum
  TP degree from {1, 2, 4, 8} that makes the GRPO pipeline fit within
  both GPU and CPU memory limits for each model size. This reflects
  realistic deployment: small models use T=1 (no tensor parallelism),
  while larger models automatically scale up to T=2, 4, or 8 as needed.

  The selection rule is:

      T* = min { T ∈ {1, 2, 4, 8} : T ≤ N  and  M_gpu_peak(T) ≤ GPU_limit
                                             and  M_cpu_offloaded ≤ CPU_limit }

  If no candidate fits, the largest viable T is used and the model is
  reported as not fitting. A fixed T can still be specified with
  `--tp-degree <int>` for single-model analysis.

#### Phase 2: Reference Model Log-Probabilities

The frozen reference model computes log-probs for the generated responses.

```
M_ref = M_ref_weights + A_ref_forward + M_ref_overhead     [Eq. 13]
```

- **Reference weights**: `2P` bytes (bf16), or `2P/N` with FSDP param sharding
- **Forward-only activations** (no gradient storage needed): much smaller than training
  ```
  A_ref_forward ≈ B_micro × S × h × 2 bytes + M_logits    [Eq. 13a]
  ```
  (Only need current layer's activations + output logits, no saving for backward)

With **CPU param offloading** for reference model (verl default):
```
M_ref_offloaded = A_ref_forward + 2×P_unit + M_ref_overhead   [Eq. 13b]
```

This is typically very memory-efficient since the reference model weights stay on CPU and are streamed in per-layer.

#### Phase 3: Actor Update (Training Step)

This is equivalent to an SFT step on the rollout data, plus KL divergence computation:

```
M_actor = M_actor_states + A_checkpoint + M_kl_buffers + M_overhead   [Eq. 14]
```

- **Actor model states**: Same as SFT (Eq. 6 or 9 depending on offloading)
- **Activations**: Gradient checkpointing on, same as Eq. 3
- **KL buffers**: Store old and new log-probs, advantages
  ```
  M_kl = B_ppo × S_r × 4 × 2 bytes                        [Eq. 14a]
            (old_logprob, new_logprob, advantages, rewards — bf16)
  ```
  This is typically negligible compared to model states and activations.

- **verl uses `ppo_max_token_len_per_gpu`** to control peak activation memory during the actor update. This limits the total tokens processed per GPU in one micro-step.

### 6.2 GRPO Peak Memory Summary

| Phase | Primary Memory Consumer | Typical Bottleneck |
|-------|------------------------|-------------------|
| Rollout (vLLM) | Model weights + KV cache | KV cache for long sequences / many rollouts |
| Reference log-probs | Forward activations + logits | Logits tensor for large vocab |
| Actor update | Optimizer states + activations | Activations for large batch × seq |

In practice, the **actor update phase** is usually the memory bottleneck for small-to-medium models (optimizer states dominate), while the **rollout phase** can become the bottleneck for very large models (full model weights must fit on GPU for vLLM, no FSDP sharding during generation — unless tensor parallel is used).

### 6.3 GRPO with FSDP + CPU Offloading

Best-case per-GPU memory for each phase:

```
M_rollout_TP = (2P + KV_cache) / T + overhead              [Eq. 15a]

M_ref_offloaded = B_micro × S × h × 2 + M_logits + overhead   [Eq. 15b]

M_actor_FSDP_offload = 4P/N + A_checkpoint(B_micro) + overhead   [Eq. 15c]
    (with optimizer offloading, params on GPU)

M_actor_FSDP_full_offload = 2×P_unit + A_checkpoint(B_micro) + overhead   [Eq. 15d]
    (with full param + optimizer offloading)
```

```
M_GRPO_peak = max(M_rollout_TP, M_ref_offloaded, M_actor_FSDP_offload, M_weight_sync)   [Eq. 16]
```

See Section 7.3 for the weight sync spike formula (Eq. 18–19).

---

## 7. Special Considerations

### 7.1 verl's Dynamic Batch Sizing

verl supports `use_dynamic_bsz=True` with `ppo_max_token_len_per_gpu` which dynamically adjusts the micro-batch size during the actor update to fit within a token budget. This makes activation memory more predictable:

```
A_dynamic ≈ ppo_max_token_len × h × (2L + 64 + 4aS/h) / S   [Eq. 17]
```

### 7.2 Gradient Accumulation

If using gradient accumulation steps `GA`:
- Micro-batch per GPU: `B_micro = B_global / (N × GA)`
- Activation memory scales with `B_micro`, not `B_global`
- Model states unchanged

### 7.3 vLLM ↔ Training Weight Sync Spike (verl)

In verl, the vLLM rollout engine and the training actor **share weights** through a weight-sync mechanism. After the actor update, updated weights are synced from the FSDP-sharded training representation to vLLM's inference format.

**This sync creates a transient memory spike** that can exceed any individual phase:

During the sync, both representations coexist briefly on GPU:
- FSDP-sharded actor weights: `2P/N` bytes per GPU
- vLLM model weights being assembled: up to `2P/T` bytes per GPU (TP-sharded)
- Temporary all-gather buffers for FSDP → full weight reconstruction

```
M_weight_sync = 2P/N + 2P/T + allgather_buffer             [Eq. 18]
```

For configurations where `T << N` (e.g., N=32 FSDP GPUs, T=2 TP GPUs), the vLLM copy dominates: `2P/T` can be much larger than `2P/N`.

**This spike can be the true peak memory** rather than any individual training or inference phase. The estimation script should compute this alongside the three phases:

```
M_GRPO_true_peak = max(M_rollout, M_ref, M_actor, M_weight_sync)   [Eq. 19]
```

> In practice, verl mitigates this by streaming the sync and not materializing all weights simultaneously. The actual spike depends on the sync implementation, but budgeting for it is prudent.

### 7.4 Hardware: GH200 Node Configuration

Verified from cluster SLURM configuration (`sinfo`):

| Property | Per GPU | Per Node (4 GPUs) |
|----------|---------|-------------------|
| GPU type | H200 (GH200 superchip) | 4× H200 |
| GPU HBM3 | **96 GB** | 384 GB |
| CPU memory (LPDDR5X) | **~202 GB** | ~808 GB |
| CPU cores (ARM Grace) | 72 | 288 |
| CPU↔GPU bandwidth | **900 GB/s** (NVLink-C2C) | — |

> The 900 GB/s NVLink-C2C bandwidth between Grace CPU and Hopper GPU means CPU offloading has a much lower performance penalty than on PCIe systems (~12.5 GB/s → 900 GB/s, ~72× improvement). **Recommendation**: Always enable CPU offloading on GH200.

### 7.5 Feasibility Criteria: "Does It Fit?"

A training configuration **does not fit** if either of these per-GPU constraints is violated:

**Constraint 1 — GPU HBM (hard limit):**
```
M_gpu_peak ≤ 96 GB                                         [Eq. 20]
```
Where `M_gpu_peak` is the peak per-GPU memory from Eq. 5/8/9/10 (SFT) or Eq. 19 (GRPO). This is the memory that **must** reside on GPU: activations, currently-active weights, communication buffers.

**Constraint 2 — CPU memory for offloaded data:**
```
M_cpu_offloaded ≤ 202 GB                                   [Eq. 21]
```
What gets offloaded per GPU:
- Optimizer states (if optimizer offload): `12P/N` bytes
- Model parameters (if full param offload): `2P/N` bytes
- Reference model weights (if ref offload): up to `2P` bytes (full model, not FSDP-sharded, streamed but buffered)

```
M_cpu_offloaded = 12P/N                        (optimizer offload only)     [Eq. 21a]
M_cpu_offloaded = 12P/N + 2P/N                 (full offload)              [Eq. 21b]
M_cpu_offloaded = 12P/N + 2P                   (GRPO: optimizer + ref)     [Eq. 21c]
```

> **Note on Eq. 21c**: The reference model in GRPO is typically NOT FSDP-sharded for offloading — each GPU streams the full model from CPU. This means the ref model's `2P` bytes must fit in **each GPU's** CPU memory share, making it the tighter constraint for large models.

**Combined constraint — does the training configuration fit on K nodes?**

```
Fits if and only if:
  1. M_gpu_peak(N=4K, T) ≤ 96 GB        (per GPU)
  2. M_cpu_offloaded(N=4K) ≤ 202 GB      (per GPU)
```

Where `K` = number of nodes, `N = 4K` GPUs total, and `T` = tensor parallelism degree for rollout.

**Example**: GRPO with a 70B model, 4 nodes (N=16), TP=4:
- Actor optimizer offload: `12 × 70B / 16 = 52.5 GB` CPU per GPU ✓
- Ref model offload: `2 × 70B = 140 GB` CPU per GPU ← **leaves only 62 GB for optimizer!**
- Total CPU: `52.5 + 140 = 192.5 GB` per GPU ≤ 202 GB ✓ (barely fits)
- Rollout: `2 × 70B / 4 = 35 GB` + KV cache GPU ✓
- Actor update: `4 × 70B / 16 = 17.5 GB` + activations GPU ✓

### 7.6 Memory for Multi-Node Training

For multi-node setups with N_total = N_nodes × GPUs_per_node:
- FSDP shards across all N_total GPUs
- Communication buffers increase (inter-node bandwidth is lower)
- Add ~5% overhead for NCCL buffers in multi-node

---

## 8. Summary: Quick Estimation Formulas

All formulas use:
- `α = 0.15` (overhead factor, use 0.20 for large distributed jobs with dynamic shapes)
- Gradient checkpointing enabled (per-layer, with 2× recomputation peak)
- Activation term: `A = BSh × (2L + 64 + 4aS/h)` [from Eq. 3]

### SFT (Single GPU, no FSDP)
```
M_SFT ≈ (1 + α) × 16P + A + 2BSV + 0.5 GB
```

### SFT (FSDP, N GPUs, optimizer offload)
```
M_SFT ≈ (1 + α) × 4P/N + A(B_micro) + 2×12h² + 2×B_micro×S×V + 0.5 GB
```

### SFT (FSDP, N GPUs, full offload)
```
M_SFT ≈ 2×12h² + A(B_micro) + 2×B_micro×S×V + 0.5 GB
```

### GRPO (verl, FSDP, optimizer offload, ref offloaded)
```
M_GRPO ≈ max(
    (2P + KV_cache) / T + 2 GB,                               -- rollout phase
    B_micro × S × h × 2 + 2×B_micro×S×V + 1 GB,              -- ref phase
    (1+α) × 4P/N + A(B_micro) + 2×B_micro×S×V + 0.5 GB,      -- actor phase
    2P/N + 2P/T + allgather_buffer                             -- weight sync spike
)
```

> **Key insight**: For large models, the rollout phase (constrained by TP, not FSDP) and the weight sync spike are often the true bottlenecks, not the actor update.

### Rules of Thumb

| Model Size | Min GPUs (SFT, FSDP+offload) | Min GPUs (GRPO, 96GB/GPU) |
|------------|-------------------------------|---------------------------|
| 1.5B | 1 | 1 (TP=1) |
| 7B | 1 | 1–2 (TP=1) |
| 14B | 2 | 4 (TP=1–2) |
| 32B | 4 | 8 (TP=2–4) |
| 70B | 8 | 16 (TP=4–8) |
| 140B | 16 | 32 (TP=8) |
| 405B | 48 | 96+ (TP=8) |

> These are approximate lower bounds. Actual requirements depend heavily on batch size and sequence length.

---

## 9. Notation Reference

| Symbol | Meaning |
|--------|---------|
| P | Total model parameters |
| h | Hidden dimension |
| L | Number of transformer layers |
| a | Number of attention heads |
| V | Vocabulary size |
| N | Number of GPUs (FSDP sharding) |
| T | Tensor parallelism degree |
| B | Global batch size |
| B_micro | Micro-batch size per GPU = B / (N × GA) |
| S | Sequence length (total) |
| S_p, S_r | Prompt / response length |
| G | GRPO group size (rollouts per prompt) |
| GA | Gradient accumulation steps |
| GC | Gradient checkpointing flag |
| P_unit | Params in largest FSDP wrapping unit |
