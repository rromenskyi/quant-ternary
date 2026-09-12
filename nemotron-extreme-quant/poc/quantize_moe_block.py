"""Stage B first look: quantize one MoE block's routed experts and shared
expert, using whatever calibration data a small prompt set actually routes
to each expert.

This is deliberately the naive version — no per-expert usage threshold, no
larger calibration corpus, no router statistics. The whole point is to see
how badly under-served most experts are with Stage A's calibration set size,
before building the real fix (docs/stage_b_prep.md's calibration-volume
section). Expect this to look worse than Stage A's dense-model result; that
gap *is* the finding.

NemotronH's MoE block (transformers' modeling_nemotron_h.py) stores expert
weights as batched parameters, not one nn.Linear per expert:
  - block.mixer.experts.up_proj:   [num_experts, moe_intermediate_size, moe_latent_size or hidden_size]
  - block.mixer.experts.down_proj: [num_experts, moe_latent_size or hidden_size, moe_intermediate_size]
  - block.mixer.shared_experts: a plain NemotronHMLP (dense, every token) — handled like Stage A's MLP layers
  - block.mixer.gate: the router (NemotronHTopkRouter) — protected, never quantized here
  - block.mixer.fc1_latent_proj / fc2_latent_proj: dense Linear if moe_latent_size is set, else nn.Identity

Usage:
    python poc/quantize_moe_block.py --model cache/models/<moe-model> \
        --output cache/models/<moe-model>-moe-block-poc --block-index 5 \
        --min-expert-tokens 8
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_acts import CALIBRATION_PROMPTS
from methods import rot_gptq_salient, rot_gptq_salient_batched

GROUP_SIZE = 128


class _EarlyExit(Exception):
    pass


def capture_moe_block(model, tokenizer, block, device: str = "cpu"):
    """Returns (expert_inputs: dict[int, Tensor], shared_inputs: dict[str, Tensor])."""
    experts_module = block.mixer.experts
    num_experts = experts_module.num_experts

    expert_inputs = {i: [] for i in range(num_experts)}
    shared_inputs = {"up_proj": [], "down_proj": []}
    handles = []

    def experts_pre_hook(module, args, kwargs):
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

    handles.append(experts_module.register_forward_pre_hook(experts_pre_hook, with_kwargs=True))

    def make_shared_hook(name):
        def hook(module, inputs):
            shared_inputs[name].append(inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]).cpu())

        return hook

    handles.append(block.mixer.shared_experts.up_proj.register_forward_pre_hook(make_shared_hook("up_proj")))
    handles.append(block.mixer.shared_experts.down_proj.register_forward_pre_hook(make_shared_hook("down_proj")))

    def stop_hook(module, inputs, output):
        raise _EarlyExit()

    handles.append(block.register_forward_hook(stop_hook))

    with torch.no_grad():
        for prompt in CALIBRATION_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=192)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            try:
                model(**inputs, use_cache=False)
            except _EarlyExit:
                pass

    for h in handles:
        h.remove()

    expert_inputs = {i: torch.cat(v, dim=0) for i, v in expert_inputs.items() if v}
    shared_inputs = {k: torch.cat(v, dim=0) for k, v in shared_inputs.items()}
    return expert_inputs, shared_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--min-expert-tokens", type=int, default=8, help="skip (leave BF16) experts with fewer")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--gptq-device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="device for the GPTQ Hessian/Cholesky/column-loop math (defaults to --device). With the "
        "cholesky_inverse fix in gptq.py, GPU is ~36x faster than CPU per expert on this workload's "
        "matrix sizes (measured: 10.4s vs 378s for a 2688-dim Hessian) — pass --gptq-device cuda.",
    )
    parser.add_argument("--salient-fraction", type=float, default=0.03)
    parser.add_argument(
        "--batched",
        action="store_true",
        help="quantize all eligible routed experts' up_proj (then down_proj) in one batched GPTQ "
        "call instead of a 128-iteration Python loop of single-expert calls — see gptq.py's "
        "batched section for why this cuts kernel-launch overhead ~128x on GPU.",
    )
    args = parser.parse_args()
    gptq_device = args.gptq_device or args.device

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    # device_map streams shards directly to the target device; from_pretrained(...).to(device)
    # instead fully materializes on CPU first, which for a 60GB checkpoint is dramatically slower.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=False, device_map=args.device
    )
    model.eval()

    block = model.model.layers[args.block_index]
    if type(block.mixer).__name__ != "NemotronHMoE":
        raise SystemExit(f"block {args.block_index} is not an MoE block (got {type(block.mixer).__name__})")

    num_experts = block.mixer.experts.num_experts
    print(f"Block {args.block_index}: MoE with {num_experts} routed experts, capturing calibration ...")
    expert_inputs, shared_inputs = capture_moe_block(model, tokenizer, block, args.device)

    hit_counts = {i: x.shape[0] for i, x in expert_inputs.items()}
    print(f"{len(hit_counts)}/{num_experts} experts received at least 1 token")
    if hit_counts:
        counts = sorted(hit_counts.values())
        print(f"  token counts per hit expert: min={counts[0]} median={counts[len(counts)//2]} max={counts[-1]}")

    act_fn = block.mixer.experts.act_fn
    import time

    valid_experts = [
        i for i in range(num_experts)
        if expert_inputs.get(i) is not None and expert_inputs[i].shape[0] >= args.min_expert_tokens
    ]
    skipped_no_data = sum(1 for i in range(num_experts) if expert_inputs.get(i) is None)
    skipped_too_few = num_experts - len(valid_experts) - skipped_no_data
    quantized = len(valid_experts)

    if args.batched:
        print(f"Batched mode: quantizing {quantized} eligible experts in 2 calls (up_proj, down_proj) ...")
        up_param = block.mixer.experts.up_proj
        down_param = block.mixer.experts.down_proj

        t0 = time.time()
        W_up_batch = up_param.data[valid_experts].detach().to(torch.float32).cpu()  # [Nv, out, in]
        X_up_list = [expert_inputs[i] for i in valid_experts]
        with torch.no_grad():
            # down_proj's real input, computed per-expert against the *original*
            # (not-yet-quantized) up_proj weight, before up_proj is overwritten.
            X_down_list = [
                act_fn(F.linear(X_up_list[j], W_up_batch[j])) for j in range(quantized)
            ]

        result_up = rot_gptq_salient_batched(
            W_up_batch, X_up_list, GROUP_SIZE, salient_fraction=args.salient_fraction,
            criterion="activation_weighted", device=gptq_device, progress_label="up_proj",
        )
        up_param.data[valid_experts] = result_up["W_hat"].to(up_param.dtype).to(up_param.device)
        print(f"  up_proj batch done in {time.time() - t0:.1f}s", flush=True)

        t1 = time.time()
        W_down_batch = down_param.data[valid_experts].detach().to(torch.float32).cpu()
        result_down = rot_gptq_salient_batched(
            W_down_batch, X_down_list, GROUP_SIZE, salient_fraction=args.salient_fraction,
            criterion="activation_weighted", device=gptq_device, progress_label="down_proj",
        )
        down_param.data[valid_experts] = result_down["W_hat"].to(down_param.dtype).to(down_param.device)
        print(f"  down_proj batch done in {time.time() - t1:.1f}s", flush=True)
        print(f"Batched total: {time.time() - t0:.1f}s for {quantized} experts", flush=True)
    else:
        t_start = time.time()
        for n, expert_idx in enumerate(valid_experts):
            X = expert_inputs[expert_idx]
            t0 = time.time()
            # down_proj's real input is act_fn(up_proj(x)), computed here with the
            # *original* (not-yet-quantized) up_proj weight, before up_proj itself
            # is overwritten below.
            up_param = block.mixer.experts.up_proj
            W_up = up_param.data[expert_idx].detach().to(torch.float32).cpu()
            with torch.no_grad():
                X_down = act_fn(F.linear(X, W_up))

            result_up = rot_gptq_salient(
                W_up, X, GROUP_SIZE, salient_fraction=args.salient_fraction, criterion="activation_weighted",
                device=gptq_device,
            )
            up_param.data[expert_idx] = result_up["W_hat"].to(up_param.dtype)

            down_param = block.mixer.experts.down_proj
            W_down = down_param.data[expert_idx].detach().to(torch.float32).cpu()
            result_down = rot_gptq_salient(
                W_down, X_down, GROUP_SIZE, salient_fraction=args.salient_fraction, criterion="activation_weighted",
                device=gptq_device,
            )
            down_param.data[expert_idx] = result_down["W_hat"].to(down_param.dtype)

            elapsed = time.time() - t0
            total_elapsed = time.time() - t_start
            eta = (total_elapsed / (n + 1)) * (len(valid_experts) - n - 1)
            print(
                f"[expert {expert_idx}/{num_experts - 1}] done in {elapsed:.1f}s "
                f"({X.shape[0]} tokens), total {total_elapsed:.0f}s, ETA {eta:.0f}s",
                flush=True,
            )

    print(
        f"\nExperts quantized (up_proj + down_proj): {quantized}, "
        f"skipped (0 tokens): {skipped_no_data}, skipped (<{args.min_expert_tokens} tokens): {skipped_too_few}"
    )

    # Shared expert: every token passes through it, same as a Stage A dense MLP layer.
    print("Quantizing shared expert (dense, full calibration coverage) ...")
    for proj_name in ("up_proj", "down_proj"):
        module = getattr(block.mixer.shared_experts, proj_name)
        W = module.weight.detach().to(torch.float32).cpu()
        X = shared_inputs[proj_name]
        result = rot_gptq_salient(
            W, X, GROUP_SIZE, salient_fraction=args.salient_fraction, criterion="activation_weighted",
            device=gptq_device,
        )
        module.weight.data.copy_(result["W_hat"].to(module.weight.dtype))
        print(f"  shared_experts.{proj_name}: {tuple(W.shape)} X{tuple(X.shape)} bpw={result['bits_per_weight']:.3f}")

    model = model.to("cpu")
    model.generation_config.do_sample = True
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
