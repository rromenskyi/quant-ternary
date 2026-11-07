"""Pure PyTorch reference implementation of NVIDIA Dory 1.0 (looped
nGPT-style, dense). Transformers has no modeling file for it (model_type
"dory" is only implemented in vLLM and llama.cpp); this is original code
written against the math in our own already-tested `mlx_lm/models/dory.py`
(ipsupport-llc/mlx-lm, merged 2026-11-07), not copied from vLLM's or
llama.cpp's Dory code.

Keeps q_proj/k_proj/v_proj/g_proj and gate_proj/up_proj/down_proj as
separate nn.Linear modules (the checkpoint's own layout) instead of the MLX
model's fused qkvg_proj/gate_up_proj, so gptq_dory.py can hook and quantize
each one individually, matching the account's usual per-component recipe.

    python -c "from dory_reference import load_dory; m, cfg = load_dory('/path/to/snapshot')"
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

_EPS = 1e-6


def _jn32(xf: torch.Tensor) -> torch.Tensor:
    return xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + _EPS)


def justnorm(x: torch.Tensor) -> torch.Tensor:
    return _jn32(x.float()).to(x.dtype)


def slerp_norm(res: torch.Tensor, out: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a, b = _jn32(res.float()), _jn32(out.float())
    return _jn32(a + alpha * (b - a)).to(res.dtype)


def loop_inject(state: torch.Tensor, input_h: torch.Tensor) -> torch.Tensor:
    return _jn32(state.float() + input_h.float()).to(state.dtype)


def qk_prep(qk: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (_jn32(qk.float()) * scale).to(qk.dtype)


def gate_out(o: torch.Tensor, g: torch.Tensor, sg2: torch.Tensor) -> torch.Tensor:
    y = o.float() * torch.sigmoid(g.float() * sg2)
    return _jn32(y).to(o.dtype)


def mlp_act(g: torch.Tensor, u: torch.Tensor, sg: torch.Tensor, su: torch.Tensor) -> torch.Tensor:
    gf, uf = g.float() * sg, u.float() * su
    return _jn32(F.silu(gf) * uf).to(g.dtype)


@dataclass
class DoryConfig:
    vocab_size: int = 131072
    hidden_size: int = 2560
    intermediate_size: int = 7168
    num_hidden_layers: int = 72
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    sliding_window: int = 1024
    n_input_layers: int = 16
    n_recurrent_layers: int = 40
    n_output_layers: int = 16
    n_recurrent_loops: int = 2
    ngpt_alpha_init: float = 0.05
    initializer_range: float = 0.0177
    attention_output_gate_logit_scale_init: Optional[float] = None
    theta_sliding: float = 10000.0
    theta_full: float = 10000000.0
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                "mlp" if i % 2 else ("full_attention" if i % 8 == 6 else "sliding_attention")
                for i in range(self.num_hidden_layers)
            ]

    @classmethod
    def from_json(cls, path: Path) -> "DoryConfig":
        raw = json.loads(path.read_text())
        rp = raw.get("rope_parameters") or {}
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            sliding_window=raw["sliding_window"],
            n_input_layers=raw["n_input_layers"],
            n_recurrent_layers=raw["n_recurrent_layers"],
            n_output_layers=raw["n_output_layers"],
            n_recurrent_loops=raw["n_recurrent_loops"],
            ngpt_alpha_init=raw.get("ngpt_alpha_init", 0.05),
            initializer_range=raw.get("initializer_range", 0.0177),
            attention_output_gate_logit_scale_init=raw.get("attention_output_gate_logit_scale_init"),
            theta_sliding=float(rp.get("sliding_attention", {}).get("rope_theta", 10000.0)),
            theta_full=float(rp.get("full_attention", {}).get("rope_theta", 10000000.0)),
            layer_types=raw.get("layer_types"),
        )


class DoryAttention(nn.Module):
    def __init__(self, cfg: DoryConfig, sliding: bool):
        super().__init__()
        self.cfg, self.sliding = cfg, sliding
        H, K, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, H * d, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, K * d, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, K * d, bias=False)
        self.g_proj = nn.Linear(cfg.hidden_size, H * d, bias=False)
        self.o_proj = nn.Linear(H * d, cfg.hidden_size, bias=False)
        self.sqk = nn.Parameter(torch.ones(H * d))
        self.s_gate = nn.Parameter(torch.ones(H * d))
        theta = cfg.theta_sliding if sliding else cfg.theta_full
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        std = cfg.initializer_range
        gate_init = cfg.attention_output_gate_logit_scale_init or math.sqrt(cfg.hidden_size)
        self._sqk_factor = math.sqrt(d) / std
        self._gate_factor = math.sqrt(gate_init) / std

    def _scales(self):
        H, K, d = self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim
        sqk = self.sqk.float() * self._sqk_factor
        sg = self.s_gate.float() * self._gate_factor
        qk_scale = torch.cat(
            [
                (sqk * sqk).view(1, H, 1, d),
                torch.ones(1, K, 1, d, dtype=torch.float32, device=sqk.device),
            ],
            dim=1,
        )
        return qk_scale, sg * sg

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Non-traditional (NeoX-style, split-half) RoPE, matching mlx.nn.RoPE(traditional=False).
        d = x.shape[-1]
        freqs = positions[:, None].float() * self.inv_freq[None, :].to(x.device)
        cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
        xf = x.float()
        x1, x2 = xf[..., : d // 2], xf[..., d // 2 :]
        rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rot.to(x.dtype)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, K, d = self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim
        qk_scale, sg2 = self._scales()
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, K, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, K, d).transpose(1, 2)
        g = self.g_proj(x)
        q, k = self._rope(q, positions), self._rope(k, positions)
        qk = qk_prep(torch.cat([q, k], dim=1), qk_scale.to(q.dtype))
        q, k = qk[:, :H], qk[:, H:]
        rep = H // K
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(d))
        o = o.transpose(1, 2).reshape(B, T, H * d)
        return self.o_proj(gate_out(o, g, sg2))


class DoryMLP(nn.Module):
    def __init__(self, cfg: DoryConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.suv_gate = nn.Parameter(torch.ones(cfg.intermediate_size))
        self.suv_up = nn.Parameter(torch.ones(cfg.intermediate_size))
        self._s = math.sqrt(cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sg, su = self.suv_gate.float() * self._s, self.suv_up.float() * self._s
        return self.down_proj(mlp_act(self.gate_proj(x), self.up_proj(x), sg, su))


class DoryLayer(nn.Module):
    def __init__(self, cfg: DoryConfig, layer_type: str):
        super().__init__()
        self.is_attn = layer_type != "mlp"
        self._alpha_scale = cfg.ngpt_alpha_init / cfg.initializer_range
        if self.is_attn:
            self.attn_alpha = nn.Parameter(torch.ones(cfg.hidden_size))
            self.mixer = DoryAttention(cfg, layer_type == "sliding_attention")
        else:
            self.mlp_alpha = nn.Parameter(torch.ones(cfg.hidden_size))
            self.mixer = DoryMLP(cfg)

    def forward(self, h: torch.Tensor, *mixer_args) -> torch.Tensor:
        a = self.attn_alpha if self.is_attn else self.mlp_alpha
        alpha = (a.float() * self._alpha_scale).abs()
        return slerp_norm(h, self.mixer(h, *mixer_args), alpha)


class DoryModel(nn.Module):
    def __init__(self, cfg: DoryConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DoryLayer(cfg, t) for t in cfg.layer_types)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.logit_scale = nn.Parameter(torch.ones(cfg.vocab_size))

    def _masks(self, T: int, device, dtype):
        causal = torch.zeros(T, T, device=device, dtype=torch.float32)
        causal.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), 1), float("-inf"))
        pos = torch.arange(T, device=device)
        too_far = pos[None, :] <= pos[:, None] - self.cfg.sliding_window
        sliding = causal.masked_fill(too_far, float("-inf"))
        return causal.to(dtype), sliding.to(dtype)

    def _run(self, i: int, h: torch.Tensor, positions: torch.Tensor, causal, sliding) -> torch.Tensor:
        layer = self.layers[i]
        if layer.is_attn:
            mask = sliding if layer.mixer.sliding else causal
            return layer(h, positions, mask)
        return layer(h)

    def forward(self, input_ids: Optional[torch.Tensor] = None, input_embeddings: Optional[torch.Tensor] = None):
        cfg = self.cfg
        h = self.embeddings(input_ids) if input_embeddings is None else input_embeddings
        B, T, _ = h.shape
        positions = torch.arange(T, device=h.device)
        causal, sliding = self._masks(T, h.device, h.dtype)
        lo, hi = cfg.n_input_layers, cfg.n_input_layers + cfg.n_recurrent_layers
        for i in range(lo):
            h = self._run(i, h, positions, causal, sliding)
        input_h = h
        state = input_h
        for _ in range(cfg.n_recurrent_loops):
            h = loop_inject(state, input_h)
            for i in range(lo, hi):
                h = self._run(i, h, positions, causal, sliding)
            state = justnorm(h)
        h = state
        for i in range(hi, cfg.num_hidden_layers):
            h = self._run(i, h, positions, causal, sliding)
        ls = self.logit_scale.float() / cfg.initializer_range
        return self.lm_head(h).float() * ls


def _strip(key: str) -> str:
    if key == "lm_head.logit_scale":
        return "logit_scale"
    if key.startswith("backbone."):
        return key[len("backbone.") :]
    return key


def load_dory(snapshot: str, device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
    """Loads a Dory snapshot directory (config.json + model.safetensors, the
    checkpoint's own separate q/k/v/g_proj and gate/up/down_proj names) into
    this reference model, unmodified (no fusing: gptq_dory.py quantizes each
    projection on its own, same recipe granularity as every other model)."""
    root = Path(snapshot)
    cfg = DoryConfig.from_json(root / "config.json")
    model = DoryModel(cfg)
    state = load_file(str(root / "model.safetensors"))
    mapped = {_strip(k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    if missing or unexpected:
        raise SystemExit(f"load_dory: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device=device, dtype=dtype), cfg
