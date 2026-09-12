"""Pre-flight RAM estimate for an MLX model + context length, BEFORE
downloading anything -- point this at a (possibly remote) config.json plus a
quantization scheme to check whether a candidate model/context combination
even has a chance of fitting on a given machine.

For this project's NemotronH hybrid Mamba2+Attention+MoE architecture, the
key fact that makes this worth automating: only a handful of layers are
actual attention (KV-cache scales with context length), while Mamba2 layers
carry a FIXED-size recurrent state independent of context length -- so this
architecture's context-length memory cost is tiny compared to a same-size
pure transformer. This script makes that trade-off explicit and reusable for
other context lengths / bit-widths without redoing the arithmetic by hand.

Usage:
    python poc/estimate_ram.py --config /path/to/config.json --bits 3 --group-size 64 --context-len 8192
    python poc/estimate_ram.py --config /path/to/config.json --bits 4 --group-size 64 --context-len 32768
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

BYTES_PER_QUANT_SCALE_ENTRY = 2 + 2  # fp16 scale + fp16 bias per group (mx.quantize's affine mode)
KV_CACHE_DTYPE_BYTES = 2  # fp16, unless --kv-bits is used (see --kv-bits flag)
MAMBA_STATE_DTYPE_BYTES = 4  # fp32, matches transformers' NemotronH mamba cache dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the model's config.json")
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--total-params", type=float, required=True, help="e.g. 31577937344 for this project's 30B-A3B model")
    parser.add_argument("--kv-bits", type=int, default=None, help="if the KV cache will be quantized (e.g. mlx_lm.server's --kv-bits)")
    parser.add_argument("--overhead-gb", type=float, default=1.5, help="fudge factor for Python/MLX runtime + activation buffers")
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    block_types = Counter(cfg["layers_block_type"])
    n_attention = block_types.get("attention", 0)
    n_mamba = block_types.get("mamba", 0)
    n_moe = block_types.get("moe", 0)

    total_params = args.total_params

    bits_per_weight = args.bits + (16 * 2) / args.group_size  # affine quant: N-bit codes + fp16 scale+bias per group
    weights_gb = total_params * bits_per_weight / 8 / 1e9

    kv_bytes_per_elem = (args.kv_bits / 8 + 0.1) if args.kv_bits else KV_CACHE_DTYPE_BYTES
    attn_kv_gb = (
        2 * cfg["num_key_value_heads"] * cfg["head_dim"] * args.context_len * kv_bytes_per_elem * n_attention
    ) / 1e9

    mamba_state_gb = (
        cfg["mamba_num_heads"] * cfg["mamba_head_dim"] * cfg["ssm_state_size"] * MAMBA_STATE_DTYPE_BYTES * n_mamba
    ) / 1e9

    total_gb = weights_gb + attn_kv_gb + mamba_state_gb + args.overhead_gb

    print(f"blocks: {n_mamba} mamba, {n_moe} moe, {n_attention} attention (of {sum(block_types.values())} total)")
    print(f"weights ({args.bits}-bit, group_size={args.group_size}): {weights_gb:.2f} GB")
    print(f"attention KV-cache @ context_len={args.context_len}: {attn_kv_gb * 1000:.1f} MB")
    print(f"mamba2 recurrent state (fixed, context-independent): {mamba_state_gb * 1000:.1f} MB")
    print(f"runtime/activation overhead (fudge factor): {args.overhead_gb:.2f} GB")
    print(f"--- ESTIMATED TOTAL: {total_gb:.2f} GB ---")


if __name__ == "__main__":
    main()
