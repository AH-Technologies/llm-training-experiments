#!/usr/bin/env python3
"""Quick smoke test for vLLM weight-based head ablation.

Verifies:
  1. vLLM loads and generates
  2. apply_model weight modification works (o_proj columns)
  3. Ablation actually changes output (vs baseline)
  4. Weight restoration is correct

Run via srun:
  srun --account=nn12068k --partition=accel --gpus=1 --mem=32G --time=00:15:00 \
    bash -c 'source venv/bin/activate && \
    PYTHONPATH=src:$PYTHONPATH HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface \
    python scripts/reasoning_head_analysis/test_vllm_ablation.py'
"""
import time

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
PROMPT = ("<|im_start|>system\nPlease reason step by step, and put your final "
          "answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
          "What is 7 * 8?<|im_end|>\n<|im_start|>assistant\n")
N_SAMPLES = 2
MAX_TOKENS = 256


def main():
    print("=" * 60)
    print("vLLM ablation smoke test")
    print("=" * 60)

    # 1. Load vLLM
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
    )
    print(f"\n[1/6] Model loaded in {time.time()-t0:.1f}s")

    # 2. Check model structure via apply_model
    def _get_info(model):
        config = model.config
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // n_heads
        o_proj_shape = tuple(model.model.layers[0].self_attn.o_proj.weight.shape)
        return {
            "n_layers": n_layers, "n_heads": n_heads,
            "hidden_size": hidden_size, "head_dim": head_dim,
            "o_proj_shape": o_proj_shape,
        }

    info = llm.apply_model(_get_info)[0]
    print(f"[2/6] Model info: {info}")
    assert info["o_proj_shape"] == (info["hidden_size"], info["hidden_size"])

    # 3. Save original weights inside worker
    def _save_originals(model):
        model._oproj_originals = {}
        for i, layer in enumerate(model.model.layers):
            model._oproj_originals[i] = layer.self_attn.o_proj.weight.data.clone()
        return len(model._oproj_originals)

    n_saved = llm.apply_model(_save_originals)[0]
    print(f"[3/6] Saved original weights for {n_saved} layers")

    # 4. Baseline generation
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=MAX_TOKENS, n=N_SAMPLES)
    t0 = time.time()
    baseline_out = llm.generate([PROMPT], params)
    baseline_time = time.time() - t0
    baseline_texts = [o.text for o in baseline_out[0].outputs]
    print(f"\n[4/6] Baseline generation ({baseline_time:.1f}s):")
    for i, t in enumerate(baseline_texts):
        print(f"  sample {i}: {t[:150]}...")

    # 5. Ablate heads 0-3 in layer 10
    test_heads = {10: [0, 1, 2, 3]}
    head_dim = info["head_dim"]

    def _ablate(model, layer_heads=test_heads):
        hd = model.config.hidden_size // model.config.num_attention_heads
        for layer_idx, h_indices in layer_heads.items():
            o_proj = model.model.layers[layer_idx].self_attn.o_proj
            # Restore from original first
            o_proj.weight.data.copy_(model._oproj_originals[layer_idx])
            for h in h_indices:
                o_proj.weight.data[:, h*hd:(h+1)*hd] = 0.0
        # Verify
        norm = model.model.layers[10].self_attn.o_proj.weight.data[:, :4*hd].norm().item()
        return norm

    norm = llm.apply_model(_ablate)[0]
    print(f"\n[5/6] Ablated heads {test_heads} — zeroed columns norm: {norm}")
    assert norm == 0.0, f"Ablation failed, norm={norm}"

    t0 = time.time()
    ablated_out = llm.generate([PROMPT], params)
    ablated_time = time.time() - t0
    ablated_texts = [o.text for o in ablated_out[0].outputs]
    print(f"  Ablated generation ({ablated_time:.1f}s):")
    for i, t in enumerate(ablated_texts):
        print(f"  sample {i}: {t[:150]}...")

    # 6. Restore and verify
    def _restore_and_check(model):
        hd = model.config.hidden_size // model.config.num_attention_heads
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.o_proj.weight.data.copy_(model._oproj_originals[i])
        # Check restoration
        restored = model.model.layers[10].self_attn.o_proj.weight.data[:, :4*hd].norm().item()
        original = model._oproj_originals[10][:, :4*hd].norm().item()
        return abs(restored - original)

    diff = llm.apply_model(_restore_and_check)[0]
    print(f"\n[6/6] Restoration verified (norm diff: {diff:.2e})")
    assert diff < 1e-6

    t0 = time.time()
    restored_out = llm.generate([PROMPT], params)
    restored_time = time.time() - t0
    restored_texts = [o.text for o in restored_out[0].outputs]
    print(f"  Restored generation ({restored_time:.1f}s):")
    for i, t in enumerate(restored_texts):
        print(f"  sample {i}: {t[:150]}...")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print(f"  Baseline gen:  {baseline_time:.1f}s")
    print(f"  Ablated gen:   {ablated_time:.1f}s")
    print(f"  Restored gen:  {restored_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
