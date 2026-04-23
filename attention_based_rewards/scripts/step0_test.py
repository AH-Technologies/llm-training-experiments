"""Step 0: Environment compatibility tests for circuit-guided credit assignment."""
import torch
import sys

print("=" * 60)
print("TEST 1: Load Qwen2.5-Math-1.5B-Instruct via TransformerLens")
print("=" * 60)

from transformer_lens import HookedTransformer

# Qwen2.5-Math-1.5B-Instruct is NOT in TL's official list,
# but it has the same architecture as Qwen2.5-1.5B-Instruct which IS supported.
# Strategy: try from_pretrained_no_processing first, then fall back.

model = None

# Attempt 1: from_pretrained_no_processing (loads any HF model without TL's name lookup)
print("Attempt 1: from_pretrained_no_processing with Qwen2.5-Math-1.5B-Instruct...")
try:
    model = HookedTransformer.from_pretrained_no_processing(
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
        device="cuda",
        dtype=torch.float16,
    )
    print("SUCCESS with from_pretrained_no_processing!")
except Exception as e:
    print(f"  Failed: {e}")

# Attempt 2: Use the supported Qwen2.5-1.5B-Instruct name but load Math weights
if model is None:
    print("\nAttempt 2: from_pretrained with Qwen2.5-1.5B-Instruct (non-Math, same arch)...")
    try:
        model = HookedTransformer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            device="cuda",
            dtype=torch.float16,
        )
        print("SUCCESS with Qwen2.5-1.5B-Instruct (fallback, same architecture)")
    except Exception as e:
        print(f"  Failed: {e}")

if model is None:
    print("FAILED: Could not load any suitable model")
    sys.exit(1)

print(f"  Layers: {model.cfg.n_layers}")
print(f"  Heads per layer: {model.cfg.n_heads}")
print(f"  Model dim: {model.cfg.d_model}")
print(f"  Head dim: {model.cfg.d_head}")
print(f"  GPU memory used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

print()
print("=" * 60)
print("TEST 2: Run inference")
print("=" * 60)

try:
    tokens = model.to_tokens("3 + 5 =")
    print(f"  Input tokens shape: {tokens.shape}")
    logits = model(tokens)
    print(f"  Output logits shape: {logits.shape}")
    top_pred = model.to_string(logits[0, -1].argmax())
    print(f"  Top prediction for '3 + 5 =': '{top_pred}'")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("TEST 3: Cache activations")
print("=" * 60)

try:
    logits, cache = model.run_with_cache(tokens)
    print(f"  Cache keys: {len(cache)}")

    hook_name = "blocks.0.attn.hook_result"
    if hook_name in cache:
        print(f"  Head output shape ({hook_name}): {cache[hook_name].shape}")
    else:
        print(f"  WARNING: {hook_name} not found")
        attn_hooks = [k for k in cache.keys() if "attn" in k][:10]
        print(f"  Available attn hooks: {attn_hooks}")

    pattern_hook = "blocks.0.attn.hook_pattern"
    if pattern_hook in cache:
        print(f"  Attention pattern shape ({pattern_hook}): {cache[pattern_hook].shape}")
    else:
        print(f"  WARNING: {pattern_hook} not found")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("TEST 4: EAP-IG Graph construction")
print("=" * 60)

try:
    from eap.graph import Graph
    graph = Graph.from_model(model)
    print(f"  Graph nodes: {len(graph.nodes)}")
    print(f"  Graph edges: {len(graph.edges)}")

    # Print a few node names to understand naming convention
    node_names = [str(n) for n in list(graph.nodes)[:10]]
    print(f"  Sample node names: {node_names}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
