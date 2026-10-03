"""Shared pieces of the Gemma 4 ternary distillation pipeline.

- Ternary fake-quantization on MLX's 2-bit affine grid: per group of
  --group input columns, scale s = mean|w|, codes {-1, 0, +1}. In MLX terms
  that is bias = -s, scale = s, codes {0, 1, 2}, so a trained model exports
  to stock 2-bit MLX losslessly (no custom kernel, no rotation).
- TernaryLinear: straight-through estimator over a latent bf16 weight.
- SRAdamW: AdamW with bf16 weights and bf16 moments, stochastically rounded
  after every update, so updates far below bf16's resolution still move the
  latent weights in expectation (fp32 master weights don't fit 121 GB).
- Atomic JSON/npy writes, shared by every resumable step.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_LINEAR = r"language_model\.layers\.\d+\..*_proj$"


def ternary_q(w: torch.Tensor, group: int) -> torch.Tensor:
    out_f, in_f = w.shape
    g = w.float().view(out_f, in_f // group, group)
    s = g.abs().mean(-1, keepdim=True).clamp_min(1e-8)
    return ((g / s).round().clamp(-1, 1) * s).view(out_f, in_f).to(w.dtype)


def affine_q(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """MLX's affine grid (scale = (max-min)/(2^b-1), bias = min) -- reference
    points for the eval (4-bit and 2-bit round-to-nearest)."""
    out_f, in_f = w.shape
    g = w.float().view(out_f, in_f // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    s = ((hi - lo) / (2**bits - 1)).clamp_min(1e-8)
    return (((g - lo) / s).round().clamp(0, 2**bits - 1) * s + lo).view(out_f, in_f).to(w.dtype)


_ternary_q_compiled = torch.compile(ternary_q, dynamic=False)


class _TernarySTE(torch.autograd.Function):
    """Ternary weight forward, identity gradient backward (straight-through).
    The quantizer is one fused kernel: read bf16, write bf16."""

    @staticmethod
    def forward(ctx, w, group):
        return _ternary_q_compiled(w, group)

    @staticmethod
    def backward(ctx, g):
        return g, None


def q4_0(w: torch.Tensor) -> torch.Tensor:
    """llama.cpp q4_0 (the grid Gemma 4's QAT trained for): groups of 32,
    d = (signed max-|w| value) / -8, codes 0..15 around 8."""
    out_f, in_f = w.shape
    g = w.float().view(out_f, in_f // 32, 32)
    d = g.gather(-1, g.abs().argmax(-1, keepdim=True)) / -8
    q = (g / torch.where(d != 0, d, torch.ones_like(d))).round().add(8).clamp(0, 15)
    return ((q - 8) * d).view(out_f, in_f).to(w.dtype)


def ref_quantizer(name: str):
    """'q4_0' or 'rtn<bits>g<group>' (MLX affine round-to-nearest)."""
    if name == "q4_0":
        return q4_0
    m = re.fullmatch(r"rtn(\d+)g(\d+)", name)
    if not m:
        raise ValueError(f"unknown reference {name!r}")
    bits, group = int(m[1]), int(m[2])
    return lambda w: affine_q(w, bits, group)


class TernaryLinear(nn.Module):
    def __init__(self, lin: nn.Linear, group: int):
        super().__init__()
        self.group = group
        self.weight = lin.weight
        self.bias = lin.bias
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        return F.linear(x, _TernarySTE.apply(self.weight, self.group), self.bias)


def ternarize(model: nn.Module, group: int, pattern: str = TEXT_LINEAR) -> list[str]:
    """Swap every matching nn.Linear for a TernaryLinear sharing its weight."""
    rx = re.compile(pattern)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and rx.search(n)]
    for n in names:
        parent, _, child = n.rpartition(".")
        lin = model.get_submodule(n)
        if lin.in_features % group:
            raise ValueError(f"{n}: in_features {lin.in_features} not divisible by group {group}")
        setattr(model.get_submodule(parent), child, TernaryLinear(lin, group))
    return names


@torch.no_grad()
def bake(model: nn.Module, names: list[str], fn) -> None:
    """Replace the named Linears' weights by fn(weight) in place (eval references)."""
    for n in names:
        m = model.get_submodule(n)
        m.weight.copy_(fn(m.weight))


def load_text_model(master: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(master, dtype=torch.bfloat16, device_map=device)
    tok = AutoTokenizer.from_pretrained(master)
    return model, tok


def sr_bf16(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 with stochastic rounding (unbiased)."""
    xi = x.contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, xi.shape, device=x.device, dtype=torch.int32)
    return ((xi + r) & -65536).view(torch.float32).to(torch.bfloat16)


def _adamw_sr(p, g, m, v, lr, b1, b2, bc1, bc2, eps, wd):
    g = g.float()
    mf = m.float() * b1 + g * (1 - b1)
    vf = v.float() * b2 + g * g * (1 - b2)
    pf = p.float() * (1 - lr * wd) - lr * (mf / bc1) / ((vf / bc2).sqrt() + eps)
    return sr_bf16(pf), sr_bf16(mf), sr_bf16(vf)


_adamw_sr_compiled = torch.compile(_adamw_sr, dynamic=False)


class SRAdamW:
    def __init__(self, params, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.b1, self.b2 = betas
        self.eps, self.wd = eps, weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, lr: float) -> None:
        self.t += 1
        dev = self.params[0].device
        sc = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)  # tensors: no recompile per step
        args = [sc(x) for x in (lr, self.b1, self.b2, 1 - self.b1**self.t, 1 - self.b2**self.t, self.eps, self.wd)]
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            np_, nm, nv = _adamw_sr_compiled(p, p.grad, m, v, *args)
            p.copy_(np_); m.copy_(nm); v.copy_(nv)
            p.grad = None

    def state_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        out = {}
        for n, m, v in zip(names, self.m, self.v):
            out[f"m.{n}"], out[f"v.{n}"] = m, v
        return out

    def load_state_tensors(self, names: list[str], tensors: dict[str, torch.Tensor]) -> None:
        for n, m, v in zip(names, self.m, self.v):
            m.copy_(tensors[f"m.{n}"])
            v.copy_(tensors[f"v.{n}"])


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)
