#!/usr/bin/env python3
"""Patch model config to remove flash_attention_2 requirement."""
import json
import os
import sys
from pathlib import Path

def patch_model_config(model_name: str, cache_dir: str = None):
    """Remove attn_implementation from cached model config."""
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    # Find the model's config.json in the cache
    hub_dir = Path(cache_dir) / "hub"

    # Model name to cache dir name (replace / with --)
    model_cache_name = "models--" + model_name.replace("/", "--")
    model_dir = hub_dir / model_cache_name

    if not model_dir.exists():
        print(f"Model not found in cache: {model_dir}")
        print("Downloading model first...")
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

    # Find snapshots directory
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        print(f"Snapshots directory not found: {snapshots_dir}")
        return False

    # Patch all config.json files in snapshots
    patched = False
    for snapshot in snapshots_dir.iterdir():
        config_path = snapshot / "config.json"
        if config_path.exists():
            print(f"Patching: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Remove or change attn_implementation
            changed = False
            if "_attn_implementation" in config:
                print(f"  Removing _attn_implementation: {config['_attn_implementation']}")
                del config["_attn_implementation"]
                changed = True
            if "attn_implementation" in config:
                print(f"  Removing attn_implementation: {config['attn_implementation']}")
                del config["attn_implementation"]
                changed = True

            if changed:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print("  Patched successfully!")
                patched = True
            else:
                print("  No flash attention config found, skipping.")

    return patched

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
    cache_dir = os.environ.get("HF_HOME")

    print(f"Patching model config for: {model_name}")
    print(f"Cache dir: {cache_dir}")

    success = patch_model_config(model_name, cache_dir)
    if success:
        print("\nModel config patched successfully!")
    else:
        print("\nNo patches needed or model not found.")
