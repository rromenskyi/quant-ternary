"""Names of the calibrated tensors: gptq_inkling2.py writes transformers'
names; the published checkpoint and mlx-lm use others. The fused experts'
gate / up rows: transformers splits them ([gate; up]), the checkpoint and
mlx-lm interleave them (gate i at row 2i, up i at row 2i + 1)."""

from __future__ import annotations

import re

# transformers module in a decoder layer -> (checkpoint, mlx-lm)
MODULES = {
    "self_attn.q_proj.linear": ("attn.wq_du", "self_attn.q_proj"),
    "self_attn.k_proj.linear": ("attn.wk_dv", "self_attn.k_proj"),
    "self_attn.v_proj.linear": ("attn.wv_dv", "self_attn.v_proj"),
    "self_attn.relative_position_bias.linear": ("attn.wr_du", "self_attn.r_proj"),
    "self_attn.o_proj.linear": ("attn.wo_ud", "self_attn.o_proj"),
    "self_attn.o_proj.gate_proj": ("attn.gate_attn", "self_attn.gate_proj"),
    "mlp.experts.gate_up_proj": ("mlp.experts.w13", "mlp.experts.gate_up_proj"),
    "mlp.experts.down_proj": ("mlp.experts.w2", "mlp.experts.down_proj"),
}
KEY = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+?)(\.weight)?$")


def split(key: str) -> tuple[int, str]:
    """A gptq_inkling2.py tensor name -> (layer, transformers module)."""
    m = KEY.match(key)
    if not m or m[2] not in MODULES:
        raise KeyError(key)
    return int(m[1]), m[2]


def checkpoint_name(key: str) -> str:
    layer, module = split(key)
    name = MODULES[module][0]
    return f"model.llm.layers.{layer}.{name}" + ("_weight" if ".experts." in name else ".weight")


def mlx_prefix(key: str) -> str:
    layer, module = split(key)
    return f"model.language_model.layers.{layer}.{MODULES[module][1]}"


def interleave(t, fused: bool):
    """[E, 2I, ...] split gate / up rows -> interleaved (torch or mlx)."""
    if not fused:
        return t
    half = t.shape[1] // 2
    gate, up = t[:, :half], t[:, half:]
    try:
        import torch
        if isinstance(t, torch.Tensor):
            return torch.stack([gate, up], dim=2).reshape(t.shape)
    except ImportError:
        pass
    import mlx.core as mx
    return mx.stack([gate, up], axis=2).reshape(t.shape)
