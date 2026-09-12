"""GPTQ-calibrate a NemotronH model at a single uniform bit-width, with NO
rotation and NO salient-weight overlay, then save a standard HF checkpoint
whose weights already sit exactly on the target affine quantization grid --
so a subsequent stock `mlx_lm.convert -q --q-bits <bits> --q-group-size
<group_size>` (or any other standard N-bit RTN quantizer using the same
group_size/scale convention) reproduces these Hessian-calibrated codes
instead of re-deriving them via naive round-to-nearest.

Why no rotation/salient here: those exist in this project to make *very*
low-bit (1-2 bit) ternary/binary quantization survivable -- at 3+ bits the
naive stock RTN conversion already works (see docs/session_findings_
2026-09-11.md's PPL measurement against F16), so plain GPTQ error
compensation (no incoherence processing, no full-precision outlier pins)
is expected to *improve* on stock RTN's quality at the exact same size and
inference speed (same MLX-native quantized-linear/gather_qmm path, no
custom kernel, no custom model-loading code needed at all).

All bit-width/group-size choices are CLI flags, not hardcoded constants --
this script is meant to be reusable across bit-widths (and, per the disk-
size back-of-envelope this session did for a hypothetical 70B model,
across model sizes) without editing source.

Usage:
    python poc/gptq_stock_convert.py \
        --model /root/nemotron30b-bf16-src --output /root/lightning30b-gptq3bit-src \
        --wikitext /root/llama.cpp/wikitext-2-raw/wiki.train.raw \
        --bits 3 --group-size 64 --calib-chunks 24 --calib-chunk-tokens 512 --moe-subbatch 12

Then, separately (stock mlx_lm, no custom code):
    mlx_lm.convert --hf-path /root/lightning30b-gptq3bit-src \
        --mlx-path /root/lightning30b-gptq3bit-mlx -q --q-bits 3 --q-group-size 64
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptq import gptq_nbit, gptq_nbit_batched
from quantize_full_moe_model import (
    ATTN_PROJECTIONS,
    MAMBA_PROJECTIONS,
    capture_all_activations,
    capture_single_block_activations,
    load_calibration_chunks,
)


def quantize_dense_block(block, kind: str, activations: dict[str, torch.Tensor], bits: int, group_size: int) -> list[str]:
    proj_names = MAMBA_PROJECTIONS if kind == "mamba" else ATTN_PROJECTIONS
    quantized = []
    for proj_name in proj_names:
        module = getattr(block.mixer, proj_name, None)
        if module is None or proj_name not in activations:
            continue
        W = module.weight.detach().to(torch.float32).cpu()
        X = activations[proj_name]
        result = gptq_nbit(W, X, bits=bits, group_size=group_size, device="cpu")
        module.weight.data.copy_(result["W_hat"].to(module.weight.dtype))
        quantized.append(proj_name)
    return quantized


def quantize_moe_block(
    block, expert_inputs: dict, shared_inputs: dict, min_expert_tokens: int, gptq_device: str,
    subbatch: int, bits: int, group_size: int,
) -> dict:
    experts_module = block.mixer.experts
    num_experts = experts_module.num_experts
    act_fn = experts_module.act_fn

    valid_experts = [
        i for i in range(num_experts)
        if expert_inputs.get(i) is not None and expert_inputs[i].shape[0] >= min_expert_tokens
    ]
    skipped = num_experts - len(valid_experts)

    up_param = experts_module.up_proj
    down_param = experts_module.down_proj
    t_up_total = t_down_total = 0.0

    for start in range(0, num_experts, subbatch):
        sub = [i for i in range(start, min(start + subbatch, num_experts)) if i in valid_experts]
        if not sub:
            continue

        t0 = time.time()
        W_up_batch = up_param.data[sub].detach().to(torch.float32).cpu()
        X_up_list = [expert_inputs[i] for i in sub]
        n_max = max(x.shape[0] for x in X_up_list)
        Xp_up = torch.zeros(len(sub), n_max, W_up_batch.shape[-1], dtype=torch.float32)
        for j, x in enumerate(X_up_list):
            Xp_up[j, : x.shape[0]] = x
        result_up = gptq_nbit_batched(W_up_batch, Xp_up, bits=bits, group_size=group_size, device=gptq_device)
        up_param.data[sub] = result_up["W_hat"].to(up_param.dtype)
        with torch.no_grad():
            X_down_list = [act_fn(F.linear(X_up_list[j], W_up_batch[j])) for j in range(len(sub))]
        t_up_total += time.time() - t0

        t1 = time.time()
        W_down_batch = down_param.data[sub].detach().to(torch.float32).cpu()
        n_max_d = max(x.shape[0] for x in X_down_list)
        Xp_down = torch.zeros(len(sub), n_max_d, W_down_batch.shape[-1], dtype=torch.float32)
        for j, x in enumerate(X_down_list):
            Xp_down[j, : x.shape[0]] = x
        result_down = gptq_nbit_batched(W_down_batch, Xp_down, bits=bits, group_size=group_size, device=gptq_device)
        down_param.data[sub] = result_down["W_hat"].to(down_param.dtype)
        t_down_total += time.time() - t1

    for proj_name in ("up_proj", "down_proj"):
        module = getattr(block.mixer.shared_experts, proj_name)
        W = module.weight.detach().to(torch.float32).cpu()
        X = shared_inputs[proj_name]
        result = gptq_nbit(W, X, bits=bits, group_size=group_size, device="cpu")
        module.weight.data.copy_(result["W_hat"].to(module.weight.dtype))

    return {
        "quantized_experts": len(valid_experts), "skipped": skipped,
        "t_up": t_up_total, "t_down": t_down_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wikitext", required=True)
    parser.add_argument("--bits", type=int, required=True, help="uniform bit-width for every linear layer quantized here")
    parser.add_argument("--group-size", type=int, required=True, help="must match the group_size the downstream mlx_lm.convert -q run uses")
    parser.add_argument("--calib-chunks", type=int, default=24)
    parser.add_argument("--calib-chunk-tokens", type=int, default=512)
    parser.add_argument("--min-expert-tokens", type=int, default=8)
    parser.add_argument("--moe-subbatch", type=int, default=12)
    parser.add_argument("--gptq-device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--end-block", type=int, default=None, help="exclusive; default = num_hidden_layers")
    parser.add_argument(
        "--sequential", action="store_true",
        help="capture each block's calibration activations against the model with all prior blocks "
        "already quantized in place, instead of one-shot capture against the pristine model.",
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
    print(f"Calibration: {len(calib_ids)} chunks, {total_tokens} tokens total, bits={args.bits} group_size={args.group_size}", flush=True)

    dense_acts, moe_acts = {}, {}
    if not args.sequential:
        print("Running one-shot calibration forward pass ...", flush=True)
        t_cap = time.time()
        dense_acts, moe_acts = capture_all_activations(model, active_types, calib_ids, device)
        print(f"Calibration capture done in {time.time() - t_cap:.0f}s", flush=True)
    else:
        print("Sequential mode: capturing + quantizing block by block ...", flush=True)

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

        if kind == "moe":
            expert_inputs, shared_inputs = moe_acts[i]
            stats = quantize_moe_block(
                block, expert_inputs, shared_inputs, args.min_expert_tokens, args.gptq_device,
                args.moe_subbatch, args.bits, args.group_size,
            )
            print(
                f"[block {i}/{end_block - 1}] moe: quantized={stats['quantized_experts']} "
                f"skipped={stats['skipped']} (up {stats['t_up']:.1f}s, down {stats['t_down']:.1f}s), "
                f"total_elapsed={time.time() - run_start:.0f}s", flush=True,
            )
        elif kind in ("mamba", "attention"):
            activations = dense_acts.get(i, {})
            quantized = quantize_dense_block(block, kind, activations, args.bits, args.group_size)
            print(
                f"[block {i}/{end_block - 1}] {kind}: {quantized} -> {args.bits}bit, "
                f"block_time={time.time() - t_block:.1f}s, total_elapsed={time.time() - run_start:.0f}s", flush=True,
            )
        else:
            print(f"[block {i}/{end_block - 1}] unknown kind {kind!r}, skipping", flush=True)

    model = model.to("cpu")
    model.generation_config.do_sample = True
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nSaved to {args.output}. Total quantization time: {time.time() - run_start:.0f}s", flush=True)
    print(
        f"Next step (stock mlx_lm, no custom code): mlx_lm.convert --hf-path {args.output} "
        f"--mlx-path {args.output}-mlx -q --q-bits {args.bits} --q-group-size {args.group_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
