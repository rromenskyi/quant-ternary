"""Full pipeline: PyTorch GPTQ + rotation + salient quantization of a
NemotronH MoE model's routed experts, packed DIRECTLY into MLX's on-disk
format for poc/rotated_switch_linear.py's RotatedTernarySwitchLinear --
no intermediate HF checkpoint, no second MLX-side quantization pass.

Unlike quantize_full_moe_model.py (which calls unrotate() so the result is
a standard dense HF checkpoint, testable via a normal forward pass but not
literally 2-bit packable), this script keeps the *rotated* ternary values
and salient mask exactly as poc/methods.py's rot_salient_ternary_batched_
packable produces them, and packs them into MLX's affine-quantized layout
by hand (mx.quantize's own affine RTN search does NOT reproduce our
{-scale, 0, +scale} 3-level convention if run naively on already-ternary
data -- see docs/session_findings_2026-09-11.md for why -- so packing is
done directly from our known scale/sign values, verified byte-for-byte
against mx.quantize/mx.dequantize's bit layout on a synthetic round-trip
test before being used here).

Everything except the routed experts (mamba, attention, shared-expert,
embeddings, norms, router) is kept in the original bf16/fp16 dtype --
these are a small fraction of this model's total parameters (~1.5B of
~30B), so leaving them uncompressed is a simple, safe choice that still
lands well under the size target once the ~29B routed-expert parameters
are ternary-packed. A later pass could additionally MLX-quantize these to
8-bit for a modest extra size reduction (~1.4GB) -- not done here to avoid
a second (Mac-side, MLX-only) processing stage.

This script runs on the PyTorch/CUDA side (the RunPod A100), producing a
complete MLX model directory that only needs `mlx.core`-side sanity
testing on the user's Mac -- it does not itself import mlx.

Usage:
    python poc/pack_mlx.py \
        --model /root/nemotron30b-bf16-src --output /root/lightning30b-mlx \
        --wikitext /root/llama.cpp/wikitext-2-raw/wiki.train.raw \
        --calib-chunks 24 --calib-chunk-tokens 512 --moe-subbatch 12 \
        --salient-fraction 0.10
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptq import gptq_nbit
from methods import rot_salient_ternary_batched_packable
from quantize_full_moe_model import (
    ATTN_PROJECTIONS,
    MAMBA_PROJECTIONS,
    NBIT_BITS,
    NBIT_GROUP_SIZE,
    capture_all_activations,
    capture_single_block_activations,
    load_calibration_chunks,
)

MOE_GROUP_SIZE = 64
ROTATION_BLOCK_SIZE = 64
MOE_BITS = 2


def pack_ternary_codes(w_hat_rotated: torch.Tensor, group_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """w_hat_rotated: [E, out, padded_in] with values in {-scale, 0,
    +scale} per group (exactly poc/methods.py's rot_salient_ternary_
    batched_packable output BEFORE unrotate -- salient-pinned entries are
    NOT necessarily on this 3-level grid, see note in main() about zeroing
    them before packing and applying them via the sparse overlay instead).

    Returns (packed_uint32, scales_f16, biases_f16) in MLX's exact affine
    layout: 16 2-bit codes per uint32, LSB-first, code c -> value
    c*scale + bias, verified against mx.quantize/mx.dequantize's bit
    layout on a synthetic round-trip (see docs/session_findings_2026-09-11.md).
    """
    E, out, padded_in = w_hat_rotated.shape
    num_groups = padded_in // group_size
    w_groups = w_hat_rotated.reshape(E, out, num_groups, group_size)
    scale = w_groups.abs().amax(dim=-1).clamp_min(1e-8)  # [E, out, num_groups]

    sign = torch.sign(w_groups)
    code = (sign + 1).to(torch.int64)  # -1,0,1 -> 0,1,2 ; unused code 3 never emitted
    codes_flat = code.reshape(E, out, padded_in).numpy().astype(np.uint32)

    codes_per_word = 32 // 2  # bits=2 -> 16 codes/uint32
    num_words = padded_in // codes_per_word
    codes_grouped = codes_flat.reshape(E, out, num_words, codes_per_word)
    packed = np.zeros((E, out, num_words), dtype=np.uint32)
    for j in range(codes_per_word):
        packed |= (codes_grouped[:, :, :, j] & 0x3) << (2 * j)

    scales_np = scale.numpy().astype(np.float16)
    biases_np = (-scale).numpy().astype(np.float16)
    return packed, scales_np, biases_np


def pack_nbit_codes(
    w_hat: torch.Tensor, scale: torch.Tensor, group_size: int, bits: int = NBIT_BITS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """w_hat: [out, in] dequantized GPTQ output (gptq_nbit's "W_hat"), scale:
    [out, num_groups] the *exact* per-group scale gptq_nbit used internally
    (gptq_nbit's "scale" -- NOT recomputed from w_hat.abs().max(), which
    would silently pick a smaller-than-intended scale for any group where
    no weight happened to hit the extremal code, corrupting the round-trip).

    Returns (packed_uint32, scales_mlx, biases_mlx) in MLX's affine layout
    (codes_per_word = 32 // bits per uint32, LSB-first -- same convention
    validated for the 2-bit ternary case in pack_ternary_codes). Our code
    range is signed [-levels, levels] (levels = 2**(bits-1)-1); MLX's
    affine codes are unsigned [0, 2**bits-1], so shift by 2**(bits-1) and
    fold the shift into bias: value = code*step = (mlx_code - 2**(bits-1))
    * step, i.e. scale_mlx = step, bias_mlx = -2**(bits-1) * step.
    """
    out_features, in_features = w_hat.shape
    num_groups = in_features // group_size
    levels = 2 ** (bits - 1) - 1
    half = 2 ** (bits - 1)

    step = (scale / levels).clamp_min(1e-12)  # [out, num_groups]
    w_groups = w_hat.reshape(out_features, num_groups, group_size)
    step_expanded = step.unsqueeze(-1)
    mlx_code = torch.round(w_groups / step_expanded) + half
    mlx_code = mlx_code.clamp(0, 2 ** bits - 1).to(torch.int64)

    codes_flat = mlx_code.reshape(out_features, in_features).numpy().astype(np.uint32)
    codes_per_word = 32 // bits
    num_words = in_features // codes_per_word
    codes_grouped = codes_flat.reshape(out_features, num_words, codes_per_word)
    packed = np.zeros((out_features, num_words), dtype=np.uint32)
    for j in range(codes_per_word):
        packed |= (codes_grouped[:, :, j] & ((1 << bits) - 1)) << (bits * j)

    scales_mlx = step.numpy().astype(np.float16)
    biases_mlx = (-half * step).numpy().astype(np.float16)
    return packed, scales_mlx, biases_mlx


def extract_salient(w_true_rotated: torch.Tensor, salient_mask: torch.Tensor, k: int):
    """w_true_rotated: [E, out, padded_in] the ORIGINAL rotated weight
    (pre-ternary), salient_mask: same shape, boolean. Every expert has the
    same fixed salient count k (see poc/methods.py's _select_salient_mask
    -- global top-k per expert's flattened tensor, now fixed to always
    return exactly k, see that function's docstring).

    Returns (bitmap[E, ceil(out*padded_in/32)] uint32, value_int8[E, k]
    int8, value_scale[E] float16). Storing explicit (row, col) int32/int16
    coordinates per salient entry is surprisingly expensive at this
    project's salient fractions: at 10%, 2x int16 + 1x float16 = 6
    bytes/entry works out to *0.6 bytes of overhead per weight in the whole
    tensor* -- more than double the 0.25 bytes/weight the ternary base
    itself costs, which would have made a 10%-salient MoE model bigger than
    plain GGUF Q4_K_M (see docs/session_findings_2026-09-11.md for the full
    size blowup this was caught before actually happening). A packed bitmap
    instead costs a *fixed* 1 bit/weight regardless of fraction (0.125
    bytes/weight); the bitmap is read directly by a rank/select Metal
    kernel at inference (poc/sparse_salient_mlx.py's salient_correction_
    bitmap), never decoded into explicit (row, col) arrays at all -- see
    that module's docstring for why (the decode-everything approach still
    blows past a 24GB Mac's usable memory even with a compact on-disk
    bitmap, because the *decoded* form isn't compact).

    Values are int8-quantized with one scale per expert (`value.abs().max()
    / 127`, a single float per expert -- negligible overhead) instead of
    stored as float16: halves the values array's resident cost (1 byte/entry
    instead of 2), which matters because unlike the bitmap this array's
    size scales with salient_fraction directly. Dequantized on the fly
    inside the same kernel that reads the bitmap (int8 * scale), not at
    load time, so this doesn't cost anything extra at inference either.
    """
    E, out, padded_in = w_true_rotated.shape
    numel = out * padded_in
    bitmap_words = (numel + 31) // 32
    bitmap = np.zeros((E, bitmap_words), dtype=np.uint32)
    vals_int8 = np.zeros((E, k), dtype=np.int8)
    val_scale = np.zeros((E,), dtype=np.float16)
    mask_flat = salient_mask.reshape(E, numel)
    w_flat = w_true_rotated.reshape(E, numel)
    for e in range(E):
        idx = torch.nonzero(mask_flat[e], as_tuple=False).squeeze(-1)  # [k], ascending flat index
        n = idx.shape[0]
        if n != k:
            raise ValueError(f"expert {e}: expected exactly {k} salient entries, got {n}")
        vals_e = w_flat[e, idx].numpy().astype(np.float32)
        scale_e = max(float(np.abs(vals_e).max()) / 127.0, 1e-8)
        vals_int8[e] = np.round(vals_e / scale_e).clip(-127, 127).astype(np.int8)
        val_scale[e] = np.float16(scale_e)
        idx_np = idx.numpy()
        words = idx_np // 32
        bits = idx_np % 32
        np.bitwise_or.at(bitmap[e], words, (np.uint32(1) << bits.astype(np.uint32)))
    return bitmap, vals_int8, val_scale


def quantize_moe_block_for_mlx(
    block, expert_inputs, shared_inputs, min_expert_tokens: int, gptq_device: str, subbatch: int,
    salient_fraction: float,
) -> dict:
    experts_module = block.mixer.experts
    num_experts = experts_module.num_experts
    act_fn = experts_module.act_fn

    valid_experts = [
        i for i in range(num_experts)
        if expert_inputs.get(i) is not None and expert_inputs[i].shape[0] >= min_expert_tokens
    ]
    if len(valid_experts) != num_experts:
        missing = set(range(num_experts)) - set(valid_experts)
        raise ValueError(
            f"pack_mlx.py requires every expert to have calibration data (need a fixed salient "
            f"count per expert); experts {sorted(missing)} had none. Increase --calib-chunks."
        )

    up_param = experts_module.up_proj
    down_param = experts_module.down_proj

    up_pad = (-up_param.shape[-1]) % ROTATION_BLOCK_SIZE
    down_pad = (-down_param.shape[-1]) % ROTATION_BLOCK_SIZE
    up_padded_in = up_param.shape[-1] + up_pad
    down_padded_in = down_param.shape[-1] + down_pad
    # Must match _select_salient_mask's exact formula (poc/methods.py):
    # max(1, int(fraction * W.numel())) -- int() truncates, round() doesn't,
    # and the two disagree by 1 for some shapes, which fails extract_salient's
    # exact-count check below.
    up_k = max(1, int(salient_fraction * up_param.shape[-2] * up_padded_in))
    down_k = max(1, int(salient_fraction * down_param.shape[-2] * down_padded_in))

    up_numel = up_param.shape[-2] * up_padded_in
    down_numel = down_param.shape[-2] * down_padded_in
    up_bitmap_words = (up_numel + 31) // 32
    down_bitmap_words = (down_numel + 31) // 32

    up_packed = np.zeros((num_experts, up_param.shape[-2], up_padded_in // 16), dtype=np.uint32)
    up_scales = np.zeros((num_experts, up_param.shape[-2], up_padded_in // MOE_GROUP_SIZE), dtype=np.float16)
    up_biases = np.zeros_like(up_scales)
    up_sbitmap = np.zeros((num_experts, up_bitmap_words), dtype=np.uint32)
    up_sval = np.zeros((num_experts, up_k), dtype=np.int8)
    up_sval_scale = np.zeros((num_experts,), dtype=np.float16)

    down_packed = np.zeros((num_experts, down_param.shape[-2], down_padded_in // 16), dtype=np.uint32)
    down_scales = np.zeros((num_experts, down_param.shape[-2], down_padded_in // MOE_GROUP_SIZE), dtype=np.float16)
    down_biases = np.zeros_like(down_scales)
    down_sbitmap = np.zeros((num_experts, down_bitmap_words), dtype=np.uint32)
    down_sval = np.zeros((num_experts, down_k), dtype=np.int8)
    down_sval_scale = np.zeros((num_experts,), dtype=np.float16)

    t_up_total = t_down_total = 0.0
    for start in range(0, num_experts, subbatch):
        sub = list(range(start, min(start + subbatch, num_experts)))

        t0 = time.time()
        W_up_batch = up_param.data[sub].detach().to(torch.float32).cpu()
        X_up_list = [expert_inputs[i] for i in sub]
        with torch.no_grad():
            X_down_list = [act_fn(F.linear(X_up_list[j], W_up_batch[j])) for j in range(len(sub))]

        result_up = rot_salient_ternary_batched_packable(
            W_up_batch, X_up_list, group_size=MOE_GROUP_SIZE, salient_fraction=salient_fraction, device=gptq_device,
        )
        w_ternary = result_up["W_hat_rotated"].clone()
        mask = result_up["salient_mask"]
        w_ternary[mask] = 0.0  # zero the base so ternary+overlay don't double-count
        p, s, b = pack_ternary_codes(w_ternary, MOE_GROUP_SIZE)
        up_packed[sub], up_scales[sub], up_biases[sub] = p, s, b
        # true (pre-ternary) rotated weight at salient positions, for the overlay:
        from rotation import rotate as rotate_torch
        W_up_rot_true = rotate_torch(W_up_batch, MOE_GROUP_SIZE)
        bmp, v, vscale = extract_salient(W_up_rot_true, mask, up_k)
        up_sbitmap[sub], up_sval[sub], up_sval_scale[sub] = bmp, v, vscale
        t_up_total += time.time() - t0

        t1 = time.time()
        W_down_batch = down_param.data[sub].detach().to(torch.float32).cpu()
        result_down = rot_salient_ternary_batched_packable(
            W_down_batch, X_down_list, group_size=MOE_GROUP_SIZE, salient_fraction=salient_fraction, device=gptq_device,
        )
        w_ternary_d = result_down["W_hat_rotated"].clone()
        mask_d = result_down["salient_mask"]
        w_ternary_d[mask_d] = 0.0
        pd, sd, bd = pack_ternary_codes(w_ternary_d, MOE_GROUP_SIZE)
        down_packed[sub], down_scales[sub], down_biases[sub] = pd, sd, bd
        W_down_rot_true = rotate_torch(W_down_batch, MOE_GROUP_SIZE)
        bmp_d, vd, vd_scale = extract_salient(W_down_rot_true, mask_d, down_k)
        down_sbitmap[sub], down_sval[sub], down_sval_scale[sub] = bmp_d, vd, vd_scale
        t_down_total += time.time() - t1

    # Shared expert: dense, protected at 8-bit, packed for real (not stored fp16).
    shared_out = {}
    for proj_name in ("up_proj", "down_proj"):
        module = getattr(block.mixer.shared_experts, proj_name)
        W = module.weight.detach().to(torch.float32).cpu()
        X = shared_inputs[proj_name]
        result = gptq_nbit(W, X, bits=NBIT_BITS, group_size=NBIT_GROUP_SIZE, device="cpu")
        p, s, b = pack_nbit_codes(result["W_hat"], result["scale"], NBIT_GROUP_SIZE, NBIT_BITS)
        shared_out[proj_name] = (p, s, b)

    return {
        "up_packed": up_packed, "up_scales": up_scales, "up_biases": up_biases,
        "up_salient_bitmap": up_sbitmap, "up_salient_val": up_sval, "up_salient_val_scale": up_sval_scale,
        "up_out": up_param.shape[-2], "up_padded_in": up_padded_in,
        "down_packed": down_packed, "down_scales": down_scales, "down_biases": down_biases,
        "down_salient_bitmap": down_sbitmap, "down_salient_val": down_sval, "down_salient_val_scale": down_sval_scale,
        "down_out": down_param.shape[-2], "down_padded_in": down_padded_in,
        "shared_up": shared_out["up_proj"], "shared_down": shared_out["down_proj"],
        "t_up": t_up_total, "t_down": t_down_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wikitext", required=True)
    parser.add_argument("--calib-chunks", type=int, default=24)
    parser.add_argument("--calib-chunk-tokens", type=int, default=512)
    parser.add_argument("--min-expert-tokens", type=int, default=1)
    parser.add_argument("--moe-subbatch", type=int, default=12)
    parser.add_argument("--gptq-device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--salient-fraction", type=float, default=0.03)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--end-block", type=int, default=None, help="exclusive; default = num_hidden_layers")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="capture each block's calibration activations against the model with all prior blocks "
        "already quantized in place, instead of a single one-shot pass against the pristine model "
        "for every block at once -- see quantize_full_moe_model.py's capture_single_block_activations.",
    )
    args = parser.parse_args()

    torch.set_num_threads(16)
    device = "cuda"
    print(f"Loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=False, device_map=device
    )
    model.eval()

    with open(f"{args.model}/config.json") as f:
        config = json.load(f)
    block_types = config["layers_block_type"]
    end_block = args.end_block if args.end_block is not None else len(block_types)
    active_types = ["_"] * args.start_block + block_types[args.start_block : end_block]
    active_types += ["_"] * (len(block_types) - end_block)

    calib_ids = load_calibration_chunks(args.wikitext, tokenizer, args.calib_chunks, args.calib_chunk_tokens)
    total_tokens = sum(ids.shape[1] for ids in calib_ids)
    print(f"Calibration: {len(calib_ids)} chunks, {total_tokens} tokens total", flush=True)

    dense_acts, moe_acts = {}, {}
    if not args.sequential:
        print("Running one-shot calibration forward pass ...", flush=True)
        t_cap = time.time()
        dense_acts, moe_acts = capture_all_activations(model, active_types, calib_ids, device)
        print(f"Calibration capture done in {time.time() - t_cap:.0f}s", flush=True)
    else:
        print("Sequential mode: capturing + quantizing block by block ...", flush=True)

    tensors = {}
    mlx_packing_meta = {
        "rotation_block_size": ROTATION_BLOCK_SIZE,
        "moe_group_size": MOE_GROUP_SIZE,
        "moe_bits": MOE_BITS,
        "nbit_group_size": NBIT_GROUP_SIZE,
        "nbit_bits": NBIT_BITS,
        "salient_fraction": args.salient_fraction,
    }
    run_start = time.time()
    for i in range(args.start_block, end_block):
        kind = block_types[i]
        block = model.model.layers[i]
        t_block = time.time()

        if args.sequential:
            captured = capture_single_block_activations(model, block, kind, calib_ids, device)
            if kind == "moe":
                moe_acts[i] = captured
            elif kind in ("mamba", "attention"):
                dense_acts[i] = captured
        prefix = f"backbone.layers.{i}.mixer"
        tensors[f"backbone.layers.{i}.norm.weight"] = block.norm.weight.detach().to(torch.float16).cpu().numpy()

        if kind == "moe":
            expert_inputs, shared_inputs = moe_acts[i]
            result = quantize_moe_block_for_mlx(
                block, expert_inputs, shared_inputs, args.min_expert_tokens, args.gptq_device,
                args.moe_subbatch, args.salient_fraction,
            )
            tensors[f"{prefix}.switch_mlp.fc1.weight"] = result["up_packed"]
            tensors[f"{prefix}.switch_mlp.fc1.scales"] = result["up_scales"]
            tensors[f"{prefix}.switch_mlp.fc1.biases"] = result["up_biases"]
            tensors[f"{prefix}.switch_mlp.fc1.salient_bitmap"] = result["up_salient_bitmap"]
            tensors[f"{prefix}.switch_mlp.fc1.salient_val"] = result["up_salient_val"]
            tensors[f"{prefix}.switch_mlp.fc1.salient_val_scale"] = result["up_salient_val_scale"]
            tensors[f"{prefix}.switch_mlp.fc2.weight"] = result["down_packed"]
            tensors[f"{prefix}.switch_mlp.fc2.scales"] = result["down_scales"]
            tensors[f"{prefix}.switch_mlp.fc2.biases"] = result["down_biases"]
            tensors[f"{prefix}.switch_mlp.fc2.salient_bitmap"] = result["down_salient_bitmap"]
            tensors[f"{prefix}.switch_mlp.fc2.salient_val"] = result["down_salient_val"]
            tensors[f"{prefix}.switch_mlp.fc2.salient_val_scale"] = result["down_salient_val_scale"]
            (sp, ss, sb), (dp, ds, db) = result["shared_up"], result["shared_down"]
            tensors[f"{prefix}.shared_experts.up_proj.weight"] = sp
            tensors[f"{prefix}.shared_experts.up_proj.scales"] = ss
            tensors[f"{prefix}.shared_experts.up_proj.biases"] = sb
            tensors[f"{prefix}.shared_experts.down_proj.weight"] = dp
            tensors[f"{prefix}.shared_experts.down_proj.scales"] = ds
            tensors[f"{prefix}.shared_experts.down_proj.biases"] = db
            # Router/gate weights: copy through untouched, full precision.
            tensors[f"{prefix}.gate.weight"] = block.mixer.gate.weight.detach().to(torch.float16).cpu().numpy()
            mlx_packing_meta.setdefault(
                "fc1", {"out_features": result["up_out"], "padded_in": result["up_padded_in"]}
            )
            mlx_packing_meta.setdefault(
                "fc2", {"out_features": result["down_out"], "padded_in": result["down_padded_in"]}
            )
            print(
                f"[block {i}] moe: up {result['t_up']:.1f}s, down {result['t_down']:.1f}s, "
                f"total_elapsed={time.time() - run_start:.0f}s", flush=True,
            )
        elif kind in ("mamba", "attention"):
            proj_names = MAMBA_PROJECTIONS if kind == "mamba" else ATTN_PROJECTIONS
            activations = dense_acts.get(i, {})
            for proj_name in proj_names:
                module = getattr(block.mixer, proj_name, None)
                if module is None or proj_name not in activations:
                    continue
                W = module.weight.detach().to(torch.float32).cpu()
                X = activations[proj_name]
                result = gptq_nbit(W, X, bits=NBIT_BITS, group_size=NBIT_GROUP_SIZE, device="cpu")
                p, s, b = pack_nbit_codes(result["W_hat"], result["scale"], NBIT_GROUP_SIZE, NBIT_BITS)
                tensors[f"{prefix}.{proj_name}.weight"] = p
                tensors[f"{prefix}.{proj_name}.scales"] = s
                tensors[f"{prefix}.{proj_name}.biases"] = b
            print(f"[block {i}] {kind}: block_time={time.time() - t_block:.1f}s", flush=True)

        # Everything else in this block not explicitly quantized above
        # (norms, mamba conv1d/ssm scalar params, router bias, etc.):
        # copy through untouched at fp16 -- these are small (no out_features
        # x in_features weight matrices among them) so leaving them
        # uncompressed costs negligible size.
        quantized_leaves = {k[len(prefix) + 1 :] for k in tensors if k.startswith(prefix + ".")}
        # MoE's raw HF parameter names (experts.up_proj/down_proj, direct
        # nn.Parameters not nested under a .weight attribute) don't share a
        # prefix with the switch_mlp.fc1/fc2 keys we replace them with above
        # -- without this, the catch-all below silently re-copies the full,
        # unquantized [128, out, in] expert tensors on top of the packed
        # ones (this happened: 4-block test came out 7.8GB instead of the
        # expected ~1.5GB, entirely due to this one bug).
        raw_handled_prefixes = ("experts.up_proj", "experts.down_proj") if kind == "moe" else ()
        for name, param in block.named_parameters():
            if "mixer." not in name:
                continue
            leaf = name.split("mixer.", 1)[-1]
            if leaf in quantized_leaves or any(leaf.startswith(q.rsplit(".", 1)[0]) for q in quantized_leaves):
                continue
            if any(leaf.startswith(p) for p in raw_handled_prefixes):
                continue
            tensors[f"{prefix}.{leaf}"] = param.detach().to(torch.float16).cpu().numpy()

    # Embeddings, final norm, lm_head, and all remaining top-level params: copy through.
    # PyTorch's in-memory naming (confirmed via named_parameters()) is
    # "model.embeddings.weight"/"model.norm_f.weight"/"lm_head.weight" --
    # different from the on-disk HF checkpoint AND the target MLX convention
    # (both "backbone.embeddings.weight"/"backbone.norm_f.weight"), which is
    # what mlx_lm's model code and the stock MLX conversion actually use.
    # lm_head sits directly on the root (no "model." prefix) in both.
    for name, param in model.named_parameters():
        if "mixer." in name or ".layers." in name:
            continue
        out_name = "backbone." + name[len("model.") :] if name.startswith("model.") else name
        tensors[out_name] = param.detach().to(torch.float16).cpu().numpy()

    import os
    os.makedirs(args.output, exist_ok=True)
    save_file(tensors, f"{args.output}/model.safetensors")
    with open(f"{args.model}/config.json") as f:
        out_config = json.load(f)
    # Wire up mlx_lm's custom-architecture loading (trust_remote_code=True):
    # model_file points at our Model/ModelArgs; quantization tells its
    # generic nn.quantize() pass to 8-bit-convert every plain nn.Linear that
    # has a matching ".scales" key (mamba/attention/shared-expert -- see
    # mlx_model_ternary.py's docstring) while leaving our custom
    # RotatedTernarySwitchLinear (no to_quantized method) and the
    # unquantized embeddings/lm_head untouched.
    out_config["model_file"] = "mlx_model_ternary.py"
    out_config["quantization"] = {"group_size": NBIT_GROUP_SIZE, "bits": NBIT_BITS, "mode": "affine"}
    with open(f"{args.output}/config.json", "w") as f:
        json.dump(out_config, f, indent=2)
    with open(f"{args.output}/mlx_packing_config.json", "w") as f:
        json.dump(mlx_packing_meta, f, indent=2)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ("mlx_model_ternary.py", "rotated_switch_linear.py", "rotation_mlx.py", "sparse_salient_mlx.py"):
        shutil.copy(f"{this_dir}/{fname}", f"{args.output}/{fname}")
    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = f"{args.model}/{fname}"
        if os.path.exists(src):
            shutil.copy(src, f"{args.output}/{fname}")

    print(f"\nSaved to {args.output}. Total time: {time.time() - run_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
