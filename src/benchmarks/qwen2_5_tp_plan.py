"""Tensor Parallel plan for Qwen2.5 models in torchtune.

Qwen2 uses the same attention structure as Llama but different MLP naming:
- Llama MLP: w1 (gate), w2 (down), w3 (up)
- Qwen2 MLP: gate_proj, down_proj, up_proj
"""

from typing import Dict

from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel.style import ParallelStyle

BASE_QWEN2_5_TP_PLAN = {
    "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
    "output": ColwiseParallel(output_layouts=Replicate()),
    "layers.*.attn.q_proj": ColwiseParallel(),
    "layers.*.attn.k_proj": ColwiseParallel(),
    "layers.*.attn.v_proj": ColwiseParallel(),
    "layers.*.attn.output_proj": RowwiseParallel(),
    "layers.*.mlp.gate_proj": ColwiseParallel(),
    "layers.*.mlp.down_proj": RowwiseParallel(),
    "layers.*.mlp.up_proj": ColwiseParallel(),
}


def base_qwen2_5_tp_plan() -> Dict[str, ParallelStyle]:
    """Return the base tensor parallel plan for Qwen2.5 models."""
    return BASE_QWEN2_5_TP_PLAN
