"""Quick dependency check for s1 SFT training. Run on a compute node:

    srun --account=nn12068k --partition=accel --gpus=1 --time=00:02:00 --mem=8G \\
        bash -c "source venv/bin/activate && python scripts/check_s1_deps.py"

Or from an interactive session:
    source venv/bin/activate && python scripts/check_s1_deps.py
"""

import importlib
import sys

REQUIRED = [
    # (module, min_version_or_None)
    ("torch", None),
    ("torchao", None),
    ("torchtune", None),
    ("torchtune.models.qwen2_5", None),
    ("torchtune.training", None),
    ("torchtune.modules", None),
    ("transformers", None),
    ("datasets", None),
    ("huggingface_hub", None),
    ("wandb", None),
    ("safetensors", None),
    ("pyarrow", None),
]

# Also check that the actual trainer imports work
IMPORT_CHECKS = [
    "from benchmarks.sft_torchtune_worker import get_model_builder, setup_distributed, _load_from_rank0_broadcast, IGNORE_INDEX",
    "from torchtune.training import FullModelHFCheckpointer, OffloadActivations, get_shard_conditions, prepare_mha_for_tp, set_activation_checkpointing",
    "from torchtune.modules import TransformerSelfAttentionLayer",
    "from benchmarks.qwen2_5_tp_plan import base_qwen2_5_tp_plan",
    "from huggingface_hub import HfApi",
]


def main():
    ok = True

    print("Checking modules:")
    for mod, _ in REQUIRED:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "")
            print(f"  OK  {mod} {v}")
        except ImportError as e:
            print(f"  FAIL {mod}: {e}")
            ok = False

    print("\nChecking imports:")
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    for stmt in IMPORT_CHECKS:
        try:
            exec(stmt)
            print(f"  OK  {stmt.split('import')[0].strip()}")
        except Exception as e:
            print(f"  FAIL {stmt[:60]}...\n       {e}")
            ok = False

    print()
    if ok:
        print("All checks passed. Ready to train.")
    else:
        print("Some checks FAILED. Fix before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
