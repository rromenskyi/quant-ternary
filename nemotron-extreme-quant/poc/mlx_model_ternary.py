"""Custom mlx_lm architecture file for our packed ternary+rotation+salient
MoE model (see poc/pack_mlx.py). Referenced via config.json's "model_file"
key, loaded by mlx_lm.utils.load_model() with trust_remote_code=True.

Reuses mlx_lm's own NemotronHMamba2Mixer/NemotronHAttention/NemotronHMLP/
MoEGate UNCHANGED -- those projections are plain 8-bit affine-quantized by
our packer (same convention mlx_lm's generic nn.quantize()/QuantizedLinear
already knows how to load, auto-detected via the presence of a matching
"{path}.scales" key), so no custom code is needed for them. Only the routed
experts (fundamentally different: ternary base + rotation + sparse salient
overlay, not a stock affine quant) need a custom module, built from
poc/rotated_switch_linear.py's RotatedTernarySwitchLinear.

This file is copied into the packed model's output directory alongside
rotation_mlx.py/sparse_salient_mlx.py/rotated_switch_linear.py (see
pack_mlx.py's main()) -- mlx_lm loads it via importlib.util.spec_from_file_
location, which does NOT add the model directory to sys.path automatically,
so we do that ourselves before importing the sibling modules below.
"""

from __future__ import annotations

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from typing import Any, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.nemotron_h import (
    ModelArgs,
    MoEGate,
    NemotronHAttention,
    NemotronHMamba2Mixer,
    NemotronHMLP,
)

from rotated_switch_linear import RotatedTernarySwitchLinear
from sparse_salient_mlx import build_rank_index


def _salient_k(fraction: float, out_features: int, padded_in: int) -> int:
    # Must match poc/methods.py's _select_salient_mask exactly (int()
    # truncation, not round()) -- this is how the packer derived its own
    # fixed per-expert salient count, see pack_mlx.py's quantize_moe_block_for_mlx.
    return max(1, int(fraction * out_features * padded_in))


class TernarySwitchMLP(nn.Module):
    """Analogous to mlx_lm.models.switch_layers.SwitchMLP, but fc1/fc2 are
    RotatedTernarySwitchLinear instead of plain SwitchLinear. NemotronH's
    routed experts are a single (non-gated) MLP: down(act(up(x))), matching
    poc/pack_mlx.py's calibration forward pass (act_fn(F.linear(x, W_up))).
    """

    def __init__(self, config: ModelArgs, pm: dict):
        super().__init__()
        self.fc1 = RotatedTernarySwitchLinear(
            input_dims=config.hidden_size,
            output_dims=pm["fc1"]["out_features"],
            num_experts=config.n_routed_experts,
            rotation_block_size=pm["rotation_block_size"],
            group_size=pm["moe_group_size"],
            bits=pm["moe_bits"],
            salient_k=_salient_k(pm["salient_fraction"], pm["fc1"]["out_features"], pm["fc1"]["padded_in"]),
        )
        self.fc2 = RotatedTernarySwitchLinear(
            input_dims=pm["fc1"]["out_features"],
            output_dims=pm["fc2"]["out_features"],
            num_experts=config.n_routed_experts,
            rotation_block_size=pm["rotation_block_size"],
            group_size=pm["moe_group_size"],
            bits=pm["moe_bits"],
            salient_k=_salient_k(pm["salient_fraction"], pm["fc2"]["out_features"], pm["fc2"]["padded_in"]),
        )
        self.activation = nn.ReLU2()

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        x = self.fc1(x, indices)
        x = self.activation(x)
        x = self.fc2(x, indices)
        return x.squeeze(-2)


class NemotronHMoETernary(nn.Module):
    def __init__(self, config: ModelArgs, pm: dict):
        super().__init__()
        self.config = config
        self.switch_mlp = TernarySwitchMLP(config, pm)
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = NemotronHMLP(
                config, intermediate_size=config.moe_shared_expert_intermediate_size
            )

    def __call__(self, x: mx.array) -> mx.array:
        residuals = x
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(residuals)
        return y


class NemotronHBlockTernary(nn.Module):
    def __init__(self, args: ModelArgs, block_type: str, pm: dict):
        super().__init__()
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.block_type = block_type
        if block_type == "M":
            self.mixer = NemotronHMamba2Mixer(args)
        elif block_type == "*":
            self.mixer = NemotronHAttention(args)
        elif block_type == "-":
            self.mixer = NemotronHMLP(args)
        elif block_type == "E":
            self.mixer = NemotronHMoETernary(args, pm)

    def __call__(self, x, mask: Optional[mx.array] = None, cache: Optional[Any] = None):
        hidden_states = self.norm(x)
        if self.block_type in ("M", "*"):
            hidden_states = self.mixer(hidden_states, mask=mask, cache=cache)
        else:
            hidden_states = self.mixer(hidden_states)
        return x + hidden_states


class NemotronHModelTernary(nn.Module):
    def __init__(self, args: ModelArgs, pm: dict):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            NemotronHBlockTernary(args, block_type, pm) for block_type in args.hybrid_override_pattern
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.fa_idx = 0
        self.ssm_idx = 0
        for b in args.hybrid_override_pattern:
            if b == "*":
                break
            elif b == "M":
                self.fa_idx += 1
        for b in args.hybrid_override_pattern:
            if b == "*":
                self.ssm_idx += 1
            elif b == "M":
                break

    def __call__(self, inputs, cache: Optional[Any] = None):
        hidden_states = self.embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        attn_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        cache_counter = 0
        for layer in self.layers:
            if layer.block_type in ("M", "*"):
                c = cache[cache_counter]
                cache_counter += 1
            else:
                c = None
            mask = attn_mask if layer.block_type == "*" else ssm_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        with open(os.path.join(_THIS_DIR, "mlx_packing_config.json")) as f:
            self.pm = json.load(f)

        self.args = args
        self.backbone = NemotronHModelTernary(args, self.pm)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.model_type = args.model_type

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        out = self.backbone(inputs, cache=cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.block_type == "M":
                caches.append(ArraysCache(size=2))
            elif l.block_type == "*":
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: dict) -> dict:
        # PyTorch's Conv1d weight layout is [out_channels, 1, kernel_size];
        # MLX's nn.Conv1d expects [out_channels, kernel_size, 1] -- same fix
        # as mlx_lm's stock nemotron_h.py Model.sanitize().
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = mx.moveaxis(v, 2, 1)

        # Decode pack_mlx.py's on-disk (bitmap, values) salient storage into
        # the explicit (row, col, value) arrays RotatedTernarySwitchLinear's
        # kernel needs -- a one-time cost at load, not per forward pass (see
        # sparse_salient_mlx.decode_salient_bitmap's docstring).
        for i, block_type in enumerate(self.args.hybrid_override_pattern):
            if block_type != "E":
                continue
            # The source HF checkpoint has no e_score_correction_bias (unlike
            # mlx_lm's MoEGate, which always allocates one) -- zeros are a
            # no-op addition to the routing scores, matching "no correction".
            bias_key = f"backbone.layers.{i}.mixer.gate.e_score_correction_bias"
            if bias_key not in weights:
                weights[bias_key] = mx.zeros((self.args.n_routed_experts,))
            prefix = f"backbone.layers.{i}.mixer.switch_mlp"
            for proj_name, dims in (("fc1", self.pm["fc1"]), ("fc2", self.pm["fc2"])):
                bitmap_key = f"{prefix}.{proj_name}.salient_bitmap"
                if bitmap_key not in weights:
                    continue
                # Keep the bitmap itself as-is (it already matches
                # RotatedTernarySwitchLinear's salient_bitmap attribute name/
                # shape) and only add the small rank checkpoint index built
                # from it -- no explicit (row, col) decode at all, which is
                # the whole point of salient_correction_bitmap (see its
                # docstring and docs/session_findings_2026-09-11.md for why
                # decode_salient_bitmap's old approach doesn't fit in 24GB).
                bitmap_np = np.array(weights[bitmap_key])
                checkpoint = build_rank_index(bitmap_np, chunk_words=RotatedTernarySwitchLinear.CHUNK_WORDS)
                weights[f"{prefix}.{proj_name}.salient_checkpoint"] = mx.array(checkpoint)

                # Version-tolerant loading: older packs (built before the
                # int8-values change) store salient_val as plain float16
                # with no salient_val_scale key. Quantize on the fly here
                # (same per-expert abs-max/127 scheme as poc/pack_mlx.py's
                # extract_salient) so one loader handles both on-disk
                # formats -- callers shouldn't need to know which pack_mlx.py
                # version produced a given model directory.
                scale_key = f"{prefix}.{proj_name}.salient_val_scale"
                if scale_key not in weights:
                    val_key = f"{prefix}.{proj_name}.salient_val"
                    vals = np.array(weights[val_key]).astype(np.float32)
                    scale = np.maximum(np.abs(vals).max(axis=-1) / 127.0, 1e-8)
                    vals_int8 = np.round(vals / scale[:, None]).clip(-127, 127).astype(np.int8)
                    weights[val_key] = mx.array(vals_int8)
                    weights[scale_key] = mx.array(scale.astype(np.float16))
        return weights
