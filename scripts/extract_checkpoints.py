"""Extract RLVR grokking checkpoints: merge FSDP shards → HF safetensors → upload to HF Hub.

Discovers all global_step_* dirs dynamically, so new steps from running training are picked up.
Idempotent: uses .uploaded markers to skip already-uploaded steps.

Usage:
    python3.12 scripts/extract_checkpoints.py
"""

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Load HF token from .env
PROJECT_DIR = Path("/cluster/projects/nn12068k/alexaau/llm-training-experiments")
env_file = PROJECT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

CKPT_BASE = PROJECT_DIR / "checkpoints" / "rlvr-grokking"
MERGED_BASE = PROJECT_DIR / "checkpoints" / "rlvr-grokking-merged"
HF_USER = "alexauren"


def merge_fsdp_to_hf(actor_dir: Path, output_dir: Path):
    """Merge FSDP-sharded model weights into a single HF model directory."""
    # Read FSDP config
    with open(actor_dir / "fsdp_config.json") as f:
        fsdp_config = json.load(f)
    world_size = fsdp_config["world_size"]

    # Load all shards in parallel
    shards = [None] * world_size

    def load_shard(rank):
        path = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        shards[rank] = torch.load(path, map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=min(world_size, os.cpu_count() or 4)) as pool:
        list(pool.map(load_shard, range(world_size)))

    # Check if DTensor or plain tensor
    sample_key = sorted(shards[0].keys())[0]
    sample_val = shards[0][sample_key]

    try:
        from torch.distributed._tensor import DTensor
    except ImportError:
        DTensor = None

    is_dtensor = DTensor is not None and isinstance(sample_val, DTensor)

    # Merge shards
    merged = {}
    for key in sorted(shards[0].keys()):
        tensors = []
        placement = None
        for rank in range(world_size):
            t = shards[rank].pop(key)
            if is_dtensor and isinstance(t, DTensor):
                if placement is None:
                    placement = t.placements[0]
                tensors.append(t._local_tensor.to(torch.bfloat16))
            else:
                tensors.append(t.to(torch.bfloat16))

        if placement is not None and placement.is_shard():
            merged[key] = torch.cat(tensors, dim=placement.dim).contiguous()
        elif placement is not None and placement.is_replicate():
            merged[key] = tensors[0]
        else:
            # Non-DTensor: FSDP1 shards along dim 0
            merged[key] = torch.cat(tensors, dim=0).contiguous()

    del shards

    # Load model config from the huggingface subfolder and save
    hf_config_dir = actor_dir / "huggingface"
    config = AutoConfig.from_pretrained(hf_config_dir)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, state_dict=merged)
    del merged, model

    # Copy tokenizer files
    tokenizer = AutoTokenizer.from_pretrained(hf_config_dir)
    tokenizer.save_pretrained(output_dir)


def upload_to_hub(local_dir: Path, repo_id: str, subfolder: str):
    """Upload a merged checkpoint to HF Hub as a subfolder."""
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        path_in_repo=subfolder,
        repo_type="model",
    )


def main():
    print("=" * 50)
    print("RLVR Grokking Checkpoint Extraction")
    print("=" * 50)

    MERGED_BASE.mkdir(parents=True, exist_ok=True)

    for run_dir in sorted(CKPT_BASE.glob("grpo_*")):
        run_name = run_dir.name
        hf_repo = f"{HF_USER}/{run_name}"
        print(f"\n>>> Run: {run_name}  →  {hf_repo}")

        for step_dir in sorted(run_dir.glob("global_step_*")):
            step_name = step_dir.name
            actor_dir = step_dir / "actor"
            merged_dir = MERGED_BASE / run_name / step_name

            if not (actor_dir / "fsdp_config.json").exists():
                print(f"  [SKIP] {step_name} - incomplete")
                continue

            if (merged_dir / ".uploaded").exists():
                print(f"  [DONE] {step_name} - already uploaded")
                continue

            print(f"  [MERGE] {step_name}...")
            merge_fsdp_to_hf(actor_dir, merged_dir)

            print(f"  [UPLOAD] {step_name}...")
            upload_to_hub(merged_dir, hf_repo, step_name)

            # Mark done and clean up large files
            (merged_dir / ".uploaded").touch()
            for f in merged_dir.glob("*.safetensors"):
                f.unlink()
            for f in merged_dir.glob("*.bin"):
                f.unlink()
            print(f"  [OK] {step_name}")

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
