"""Full-model GPTQ quantization for NemotronH hybrid MoE models (e.g.
Nemotron-3.5-Lightning-30B-A3B), driven by config.json's layers_block_type
instead of guessing mixer kind from module class names (collect_acts.py's
classify_mixer doesn't recognize "NemotronHMoE" or this model's Mamba2 mixer
class name).

Strategy (matches pack_gguf.py's tensor-type mixed-precision philosophy,
generalized to a from-scratch GPTQ pass instead of packing an existing
GGUF):
  - moe blocks: routed experts (up_proj/down_proj), batched across experts
    in sub-batches of --moe-subbatch -> literal ternary via
    gptq_ternary_batched called directly (no rotation, no salient mask) so
    the stored W_hat is exactly {-scale, 0, +scale} per group in the
    standard basis -- required to pack into a real 2-bit format afterward
    (rot_gptq_salient's unrotate() leaves a dense reconstruction instead,
    see docs/stage_b_prep.md). Shared expert (dense, every token) -> 8-bit
    gptq_nbit (protected; the dense 4B model's ablations showed always-on
    dense paths need more than 1-2 bits -- see nano4b-mamba-q8 in
    experiments_log.csv).
  - mamba blocks: in_proj/out_proj -> 8-bit gptq_nbit (protected; skipmamba/
    mamba-q8/mamba-q4 ablations all showed Mamba is highly sensitive).
  - attention blocks: q/k/v/o_proj -> 8-bit gptq_nbit (protected; skipattn
    was the single worst ablation result on the dense model).

One-shot calibration, NOT sequential: this model's Mamba mixer requires a
CUDA-only causal_conv1d kernel with no CPU fallback (transformers'
use_kernel_func_from_hub_with_fallback always prefers the installed
`causal_conv1d` package over the pure-PyTorch reference, with no runtime
device check -- confirmed by a CPU-device_map run hard-crashing with
"Expected x.is_cuda() to be true"), so the model must stay resident on the
GPU for every forward pass. A 65GB bf16 model already consumes ~77GB of the
A100's 80GB, leaving only ~2-3GB free -- not enough for a sequential
per-block forward pass loop (quantize_sequential.py's approach) AND a
batched multi-expert Hessian in the same step. One-shot (single pass over
all calibration chunks, every target module hooked simultaneously) avoids
re-running the forward pass once per block, and --moe-subbatch bounds the
batched-Hessian memory spike per MoE block. Trade-off: quantization error
from an already-quantized block cannot be compensated for by a not-yet-
quantized downstream block's calibration (the win sequential normally
gives) -- acceptable here because non-expert tensors stay at 8-bit (low
reconstruction error) and MoE's sparse routing + residual stream already
buffer against a single sub-layer's error more than a dense stack does.

Calibration corpus: wikitext-2-raw/wiki.train.raw chunked into
--calib-chunks chunks of --calib-chunk-tokens tokens each, instead of the
10-prompt CALIBRATION_PROMPTS (~1076 chars total) used elsewhere in this
repo -- that starves per-expert calibration badly on a 128-expert model
(block-1 test with the small prompt set left 16/128 experts under the
8-token floor, and running median was only 12 tokens/expert).

Usage:
    python poc/quantize_full_moe_model.py \
        --model /root/nemotron30b-bf16-src --output /root/lightning30b-ours \
        --wikitext /root/llama.cpp/wikitext-2-raw/wiki.train.raw \
        --calib-chunks 24 --calib-chunk-tokens 512 --moe-subbatch 12
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptq import gptq_nbit
from methods import rot_salient_ternary_batched_packable
from rotation import unrotate

ATTN_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAMBA_PROJECTIONS = ["in_proj", "out_proj"]
MOE_GROUP_SIZE = 64  # matches mlx-lm's default --q-group-size, so the eventual
# MLX packer can lift {scale, code} straight out of each 64-wide group
# without needing to reconcile a different quantization group size.
NBIT_GROUP_SIZE = 32
NBIT_BITS = 4
SALIENT_FRACTION = 0.03  # matches the recipe that got 1.006x PPL on block 1 (see experiments_log.csv)


def load_calibration_chunks(path: str, tokenizer, num_chunks: int, chunk_tokens: int) -> list[torch.Tensor]:
    text = open(path, encoding="utf-8").read()
    approx_chars_per_chunk = chunk_tokens * 6  # generous upper bound, ~4 chars/token for English text
    stride = max(1, len(text) // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = i * stride
        raw = text[start : start + approx_chars_per_chunk]
        ids = tokenizer(raw, return_tensors="pt", truncation=True, max_length=chunk_tokens)["input_ids"]
        if ids.shape[1] > 0:
            chunks.append(ids)
    return chunks


def capture_all_activations(model, block_types: list[str], calib_ids: list[torch.Tensor], device: str):
    """Single pass: hook every target module in every block simultaneously,
    against the pristine (not-yet-quantized) model. Returns per-block dicts
    keyed exactly like the per-block quantize functions expect.
    """
    dense_captured: dict[int, dict[str, list[torch.Tensor]]] = {}
    moe_expert_inputs: dict[int, dict[int, list[torch.Tensor]]] = {}
    moe_shared_inputs: dict[int, dict[str, list[torch.Tensor]]] = {}
    handles = []

    for i, kind in enumerate(block_types):
        block = model.model.layers[i]
        if kind in ("mamba", "attention"):
            proj_names = MAMBA_PROJECTIONS if kind == "mamba" else ATTN_PROJECTIONS
            targets = {
                name: getattr(block.mixer, name)
                for name in proj_names
                if isinstance(getattr(block.mixer, name, None), nn.Linear)
            }
            dense_captured[i] = {name: [] for name in targets}

            def make_hook(block_idx, name):
                def hook(module, inputs):
                    dense_captured[block_idx][name].append(
                        inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]).cpu()
                    )

                return hook

            for name, module in targets.items():
                handles.append(module.register_forward_pre_hook(make_hook(i, name)))

        elif kind == "moe":
            experts_module = block.mixer.experts
            num_experts = experts_module.num_experts
            moe_expert_inputs[i] = {e: [] for e in range(num_experts)}
            moe_shared_inputs[i] = {"up_proj": [], "down_proj": []}

            def make_experts_hook(block_idx, n_experts):
                def hook(module, args, kwargs):
                    hidden_states = args[0] if args else kwargs["hidden_states"]
                    top_k_index = args[1] if len(args) > 1 else kwargs["top_k_index"]
                    hidden_states = hidden_states.detach().to(torch.float32)
                    with torch.no_grad():
                        expert_mask = F.one_hot(top_k_index, num_classes=n_experts).permute(2, 1, 0)
                        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
                    for expert_idx in expert_hit:
                        expert_idx = expert_idx.item()
                        _, token_idx = torch.where(expert_mask[expert_idx])
                        if token_idx.numel() == 0:
                            continue
                        moe_expert_inputs[block_idx][expert_idx].append(hidden_states[token_idx].cpu())

                return hook

            handles.append(
                experts_module.register_forward_pre_hook(make_experts_hook(i, num_experts), with_kwargs=True)
            )

            def make_shared_hook(block_idx, name):
                def hook(module, inputs):
                    moe_shared_inputs[block_idx][name].append(
                        inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]).cpu()
                    )

                return hook

            handles.append(
                block.mixer.shared_experts.up_proj.register_forward_pre_hook(make_shared_hook(i, "up_proj"))
            )
            handles.append(
                block.mixer.shared_experts.down_proj.register_forward_pre_hook(make_shared_hook(i, "down_proj"))
            )

    with torch.no_grad():
        for n, ids in enumerate(calib_ids):
            model(input_ids=ids.to(device), use_cache=False)
            print(f"  calibration pass {n + 1}/{len(calib_ids)} done", flush=True)

    for h in handles:
        h.remove()

    dense_out = {
        i: {name: torch.cat(v, dim=0) for name, v in d.items() if v} for i, d in dense_captured.items()
    }
    moe_out = {}
    for i in moe_expert_inputs:
        expert_inputs = {e: torch.cat(v, dim=0) for e, v in moe_expert_inputs[i].items() if v}
        shared_inputs = {k: torch.cat(v, dim=0) for k, v in moe_shared_inputs[i].items()}
        moe_out[i] = (expert_inputs, shared_inputs)

    return dense_out, moe_out


class _EarlyExit(Exception):
    pass


def capture_single_block_activations(model, block, kind: str, calib_ids: list[torch.Tensor], device: str):
    """Sequential-calibration counterpart to capture_all_activations: hooks
    only ONE block, plus an early-exit hook right after it, and runs the
    calibration chunks through whatever the model currently is -- so if
    blocks 0..i-1 have already been quantized in place by the time this is
    called for block i, this block's captured inputs reflect that, exactly
    like quantize_sequential.py's approach for the dense model. Costs a
    forward pass through blocks 0..i for every block (instead of one pass
    through all 52 for every calibration chunk, done once) -- more compute,
    but accounts for compounding quantization error across blocks, which
    the one-shot capture in capture_all_activations cannot.
    """
    handles = []

    if kind in ("mamba", "attention"):
        proj_names = MAMBA_PROJECTIONS if kind == "mamba" else ATTN_PROJECTIONS
        targets = {
            name: getattr(block.mixer, name)
            for name in proj_names
            if isinstance(getattr(block.mixer, name, None), nn.Linear)
        }
        captured = {name: [] for name in targets}

        def make_hook(name):
            def hook(module, inputs):
                captured[name].append(inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]).cpu())

            return hook

        for name, module in targets.items():
            handles.append(module.register_forward_pre_hook(make_hook(name)))

        result = None  # dense_out, filled after the forward loop below

    elif kind == "moe":
        experts_module = block.mixer.experts
        num_experts = experts_module.num_experts
        expert_inputs = {e: [] for e in range(num_experts)}
        shared_inputs = {"up_proj": [], "down_proj": []}

        def experts_hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            top_k_index = args[1] if len(args) > 1 else kwargs["top_k_index"]
            hidden_states = hidden_states.detach().to(torch.float32)
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
            for expert_idx in expert_hit:
                expert_idx = expert_idx.item()
                _, token_idx = torch.where(expert_mask[expert_idx])
                if token_idx.numel() == 0:
                    continue
                expert_inputs[expert_idx].append(hidden_states[token_idx].cpu())

        handles.append(experts_module.register_forward_pre_hook(experts_hook, with_kwargs=True))

        def make_shared_hook(name):
            def hook(module, inputs):
                shared_inputs[name].append(
                    inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]).cpu()
                )

            return hook

        handles.append(block.mixer.shared_experts.up_proj.register_forward_pre_hook(make_shared_hook("up_proj")))
        handles.append(block.mixer.shared_experts.down_proj.register_forward_pre_hook(make_shared_hook("down_proj")))

    else:
        return None

    def stop_hook(module, inputs, output):
        raise _EarlyExit()

    handles.append(block.register_forward_hook(stop_hook))

    with torch.no_grad():
        for ids in calib_ids:
            try:
                model(input_ids=ids.to(device), use_cache=False)
            except _EarlyExit:
                pass

    for h in handles:
        h.remove()

    if kind in ("mamba", "attention"):
        return {name: torch.cat(v, dim=0) for name, v in captured.items() if v}
    expert_inputs_out = {e: torch.cat(v, dim=0) for e, v in expert_inputs.items() if v}
    shared_inputs_out = {k: torch.cat(v, dim=0) for k, v in shared_inputs.items()}
    return expert_inputs_out, shared_inputs_out


def quantize_dense_block(block, kind: str, activations: dict[str, torch.Tensor]) -> list[str]:
    # Deliberately hardcoded to CPU, unlike the batched MoE path: a single
    # (non-batched) GPTQ call's per-column loop is ~2700 sequential tiny ops,
    # and on GPU that's dominated by kernel-launch latency rather than
    # compute -- measured slower than CPU for this exact shape class (see
    # quantize_sequential.py's --gptq-device docstring, and reproduced here:
    # a mamba block that finishes in ~20s on CPU instead spun for 70+ CPU-
    # minutes at 0% GPU utilization when routed to CUDA).
    proj_names = MAMBA_PROJECTIONS if kind == "mamba" else ATTN_PROJECTIONS
    quantized = []
    for proj_name in proj_names:
        module = getattr(block.mixer, proj_name, None)
        if not isinstance(module, nn.Linear) or proj_name not in activations:
            continue
        W = module.weight.detach().to(torch.float32).cpu()
        X = activations[proj_name]
        result = gptq_nbit(W, X, bits=NBIT_BITS, group_size=NBIT_GROUP_SIZE, device="cpu")
        module.weight.data.copy_(result["W_hat"].to(module.weight.dtype).to(module.weight.device))
        quantized.append(proj_name)
    return quantized


def quantize_moe_block(
    block, expert_inputs, shared_inputs, min_expert_tokens: int, gptq_device: str, subbatch: int,
    salient_fraction: float = SALIENT_FRACTION,
) -> dict:
    experts_module = block.mixer.experts
    num_experts = experts_module.num_experts
    act_fn = experts_module.act_fn

    valid_experts = [
        i
        for i in range(num_experts)
        if expert_inputs.get(i) is not None and expert_inputs[i].shape[0] >= min_expert_tokens
    ]
    skipped_no_data = sum(1 for i in range(num_experts) if expert_inputs.get(i) is None)
    skipped_too_few = num_experts - len(valid_experts) - skipped_no_data

    up_param = experts_module.up_proj
    down_param = experts_module.down_proj

    t_up_total = 0.0
    t_down_total = 0.0
    for start in range(0, len(valid_experts), subbatch):
        sub = valid_experts[start : start + subbatch]

        t0 = time.time()
        W_up_batch = up_param.data[sub].detach().to(torch.float32).cpu()
        X_up_list = [expert_inputs[i] for i in sub]
        with torch.no_grad():
            X_down_list = [act_fn(F.linear(X_up_list[j], W_up_batch[j])) for j in range(len(sub))]

        # rot_salient_ternary_batched_packable takes the raw (unpadded) X list itself
        # (it rotates internally); unrotate() here reproduces rot_gptq_salient_batched's
        # end-to-end behavior for a standard nn.Linear-style forward pass, so this
        # checkpoint is testable immediately -- the *rotated* (result_up/result_down)
        # values are what an eventual MLX packer would use instead of these dense ones.
        result_up = rot_salient_ternary_batched_packable(
            W_up_batch, X_up_list, group_size=MOE_GROUP_SIZE, salient_fraction=salient_fraction, device=gptq_device,
        )
        W_up_hat = unrotate(result_up["W_hat_rotated"], MOE_GROUP_SIZE, result_up["in_features"])
        up_param.data[sub] = W_up_hat.to(up_param.dtype).to(up_param.device)
        t_up_total += time.time() - t0

        t1 = time.time()
        W_down_batch = down_param.data[sub].detach().to(torch.float32).cpu()

        result_down = rot_salient_ternary_batched_packable(
            W_down_batch, X_down_list, group_size=MOE_GROUP_SIZE, salient_fraction=salient_fraction, device=gptq_device,
        )
        W_down_hat = unrotate(result_down["W_hat_rotated"], MOE_GROUP_SIZE, result_down["in_features"])
        down_param.data[sub] = W_down_hat.to(down_param.dtype).to(down_param.device)
        t_down_total += time.time() - t1

        if gptq_device == "cuda":
            torch.cuda.empty_cache()

    for proj_name in ("up_proj", "down_proj"):
        # Non-batched (single tensor), so CPU per quantize_dense_block's note above.
        module = getattr(block.mixer.shared_experts, proj_name)
        W = module.weight.detach().to(torch.float32).cpu()
        X = shared_inputs[proj_name]
        result = gptq_nbit(W, X, bits=NBIT_BITS, group_size=NBIT_GROUP_SIZE, device="cpu")
        module.weight.data.copy_(result["W_hat"].to(module.weight.dtype))

    return {
        "quantized_experts": len(valid_experts),
        "skipped_no_data": skipped_no_data,
        "skipped_too_few": skipped_too_few,
        "t_up": t_up_total,
        "t_down": t_down_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wikitext", required=True)
    parser.add_argument("--calib-chunks", type=int, default=24)
    parser.add_argument("--calib-chunk-tokens", type=int, default=512)
    parser.add_argument("--min-expert-tokens", type=int, default=8)
    parser.add_argument("--moe-subbatch", type=int, default=12, help="experts per batched GPTQ call (memory bound)")
    parser.add_argument("--salient-fraction", type=float, default=SALIENT_FRACTION)
    parser.add_argument("--gptq-device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--end-block", type=int, default=None, help="exclusive; default = num_hidden_layers")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="capture each block's calibration activations against the model with all prior blocks "
        "already quantized in place (one extra truncated forward pass per block), instead of a single "
        "one-shot pass against the pristine model for every block at once. Slower (~26x more forward-"
        "pass compute for 52 blocks x 24 calibration chunks) but accounts for compounding quantization "
        "error across blocks -- see capture_single_block_activations's docstring.",
    )
    args = parser.parse_args()

    # This host has 252 CPU threads; torch's default is one thread pool per
    # process sized to match, which is great for a few huge matmuls but
    # catastrophic for the dense-quantize path's tight per-column loop
    # (~2700 tiny ops in sequence, each paying full 252-way thread-pool
    # sync overhead) -- reproduced: a single mamba block that finishes in
    # ~20s on GPU took 7+ CPU-minutes and counting at this default before
    # even printing a result. Cap it once, globally, for the whole process.
    torch.set_num_threads(16)
    device = "cuda"  # model must stay on GPU: Mamba's causal_conv1d kernel has no CPU fallback in this env
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

    print(
        f"Loading calibration corpus from {args.wikitext} "
        f"({args.calib_chunks} chunks x {args.calib_chunk_tokens} tokens) ...",
        flush=True,
    )
    calib_ids = load_calibration_chunks(args.wikitext, tokenizer, args.calib_chunks, args.calib_chunk_tokens)
    total_tokens = sum(ids.shape[1] for ids in calib_ids)
    print(f"  {len(calib_ids)} chunks, {total_tokens} tokens total", flush=True)

    dense_acts, moe_acts = {}, {}
    if not args.sequential:
        print("Running one-shot calibration forward pass (all target blocks hooked at once) ...", flush=True)
        t_cap = time.time()
        dense_acts, moe_acts = capture_all_activations(model, active_types, calib_ids, device)
        print(f"Calibration capture done in {time.time() - t_cap:.0f}s", flush=True)
    else:
        print("Sequential mode: capturing + quantizing block by block ...", flush=True)

    run_start = time.time()
    for i in range(args.start_block, end_block):
        block = model.model.layers[i]
        kind = block_types[i]
        t_block = time.time()

        if args.sequential:
            captured = capture_single_block_activations(model, block, kind, calib_ids, device)
            if kind == "moe":
                moe_acts[i] = captured
            elif kind in ("mamba", "attention"):
                dense_acts[i] = captured

        if kind == "moe":
            expert_inputs, shared_inputs = moe_acts[i]
            hit = len(expert_inputs)
            num_experts = block.mixer.experts.num_experts
            stats = quantize_moe_block(
                block, expert_inputs, shared_inputs, args.min_expert_tokens, args.gptq_device, args.moe_subbatch,
                args.salient_fraction,
            )
            print(
                f"[block {i}/{end_block - 1}] moe: {hit}/{num_experts} experts hit, "
                f"quantized={stats['quantized_experts']} skipped_no_data={stats['skipped_no_data']} "
                f"skipped_too_few={stats['skipped_too_few']} "
                f"(up {stats['t_up']:.1f}s, down {stats['t_down']:.1f}s), shared_experts=8bit, "
                f"block_time={time.time() - t_block:.1f}s, total_elapsed={time.time() - run_start:.0f}s",
                flush=True,
            )
        elif kind in ("mamba", "attention"):
            activations = dense_acts.get(i, {})
            quantized = quantize_dense_block(block, kind, activations)
            print(
                f"[block {i}/{end_block - 1}] {kind}: {quantized} -> 8bit, "
                f"block_time={time.time() - t_block:.1f}s, total_elapsed={time.time() - run_start:.0f}s",
                flush=True,
            )
        else:
            print(f"[block {i}/{end_block - 1}] unknown kind {kind!r}, skipping", flush=True)

    model = model.to("cpu")
    model.generation_config.do_sample = True
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nSaved to {args.output}. Total quantization time: {time.time() - run_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
