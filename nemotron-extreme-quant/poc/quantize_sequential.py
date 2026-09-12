"""True sequential (block-by-block) quantization: quantize block i, then
capture block i+1's calibration activations by running the calibration set
through the model with blocks 0..i *already quantized* — instead of
capturing every layer's activations in one pass against the untouched
original model (what quantize_full_model.py does, and which round 1 showed
compounds badly across 42 blocks: see poc/quality_report.md).

Each block's target Linear layers are quantized in place immediately after
its activations are captured, so by the time we move to block i+1, its
inputs already reflect every upstream quantization decision — exactly what
GPTQ assumes, and what a one-shot calibration pass does not give you.

To keep this affordable, each round's forward pass exits right after the
target block finishes (via a hook that raises), so quantizing block i only
costs ~i/num_layers of a full forward pass, not a full pass every round.

Usage:
    python poc/quantize_sequential.py \
        --model cache/models/Nemotron-3-Nano-4B-BF16 \
        --output cache/models/Nemotron-3-Nano-4B-BF16-sequential-poc \
        --num-blocks 5
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_acts import CALIBRATION_PROMPTS, classify_mixer
from collect_acts_full import ELIGIBLE_PROJECTIONS
from gptq import gptq_binary, gptq_nbit, gptq_ternary
from methods import rot_gptq_salient

GROUP_SIZE = 128


class _EarlyExit(Exception):
    pass


def _block_targets(block) -> dict[str, nn.Linear]:
    kind = classify_mixer(block.mixer)
    if kind is None:
        return {}
    targets = {}
    for proj_name in ELIGIBLE_PROJECTIONS[kind]:
        module = getattr(block.mixer, proj_name, None)
        if isinstance(module, nn.Linear):
            targets[proj_name] = module
    return targets


def capture_block_activations(
    model, tokenizer, block, targets: dict[str, nn.Linear], device: str = "cpu"
) -> dict[str, torch.Tensor]:
    captured = {name: [] for name in targets}
    handles = []

    def make_hook(name):
        def hook(module, inputs):
            captured[name].append(inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]))

        return hook

    for name, module in targets.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

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

    return {name: torch.cat(tensors, dim=0) for name, tensors in captured.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "mps", "cuda"], help="device for the model forward pass"
    )
    parser.add_argument(
        "--gptq-device",
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="device for the GPTQ Hessian/Cholesky math (defaults to --device); the per-column "
        "loop is many small ops, which on GPU can be *slower* than CPU due to kernel-launch "
        "overhead — pass --gptq-device cpu to keep the model forward pass on GPU without that cost",
    )
    parser.add_argument("--salient-fraction", type=float, default=0.03)
    parser.add_argument("--salient-criterion", default="activation_weighted")
    parser.add_argument("--method", default="binary", choices=["binary", "ternary"])
    parser.add_argument(
        "--group-size",
        type=int,
        default=GROUP_SIZE,
        help="quantization group size; must match the target export format's block size if the "
        "checkpoint will later be packed into a fixed-block format like GGUF's TQ2_0 (256)",
    )
    parser.add_argument(
        "--skip-quantize-for",
        default="",
        help="comma-separated proj_names (e.g. 'q_proj,k_proj') to leave entirely at original "
        "precision (no GPTQ at all) — for ablating which whole tensor TYPES a mixed-precision "
        "GGUF export needs to protect, as opposed to per-weight salient pinning which a fixed "
        "ternary block format like TQ2_0 cannot represent.",
    )
    parser.add_argument(
        "--skip-rotation",
        action="store_true",
        help="bypass the Hadamard rotation wrapper and call gptq_binary/gptq_ternary directly. "
        "rot_gptq_salient's rotate-then-unrotate leaves the *stored* weight as a dense "
        "reconstruction (every entry a linear combination of the rotated-domain ternary values) "
        "which works fine with plain nn.Linear but is NOT literally {-scale,0,scale} in the "
        "standard basis — required for packing into a real fixed-format like GGUF's TQ2_0.",
    )
    parser.add_argument(
        "--nbit-quantize-for",
        default="",
        help="comma-separated proj_names to quantize with standard N-bit uniform GPTQ "
        "(gptq_nbit, Q8_0/Q4_0-style: absmax scale, GPTQ error compensation, no salient "
        "pinning) instead of --method's binary/ternary — for tensors too structurally "
        "critical for 1-2 bit ternary but not worth keeping at full F16 either.",
    )
    parser.add_argument("--nbit-bits", type=int, default=8)
    parser.add_argument("--nbit-group-size", type=int, default=32)
    args = parser.parse_args()
    gptq_device = args.gptq_device or args.device
    group_size = args.group_size
    skip_quantize_for = {s.strip() for s in args.skip_quantize_for.split(",") if s.strip()}
    nbit_quantize_for = {s.strip() for s in args.nbit_quantize_for.split(",") if s.strip()}

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    # device_map streams shards directly to the target device; from_pretrained(...).to(device)
    # instead fully materializes on CPU first, which is dramatically slower for large checkpoints.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=False, device_map=args.device
    )
    model.eval()

    for i in range(args.num_blocks):
        block = model.model.layers[i]
        targets = _block_targets(block)
        if not targets:
            print(f"[block {i}] no eligible linear layers (kind={classify_mixer(block.mixer)}), skipping")
            continue

        activations = capture_block_activations(model, tokenizer, block, targets, args.device)

        kind = classify_mixer(block.mixer)
        for proj_name, module in targets.items():
            if proj_name in skip_quantize_for:
                print(f"[block {i}/{args.num_blocks - 1}] {kind}.{proj_name}: skipped (kept at original precision)")
                continue
            W = module.weight.detach().to(torch.float32).cpu()
            X = activations[proj_name].cpu()
            if proj_name in nbit_quantize_for:
                result = gptq_nbit(
                    W, X, bits=args.nbit_bits, group_size=args.nbit_group_size, device=gptq_device
                )
            elif args.skip_rotation:
                # No rotate()/unrotate() wrapper: the stored W_hat is then literally
                # {-scale, 0, scale} (ternary) or {-scale, scale} (binary) per group in
                # the *standard* basis, packable into a fixed format like GGUF's TQ2_0 —
                # rot_gptq_salient's unrotate() instead leaves a dense reconstruction.
                gptq_fn = gptq_ternary if args.method == "ternary" else gptq_binary
                result = gptq_fn(W, X, group_size=group_size, device=gptq_device)
            else:
                result = rot_gptq_salient(
                    W,
                    X,
                    group_size,
                    salient_fraction=args.salient_fraction,
                    criterion=args.salient_criterion,
                    device=gptq_device,
                    method=args.method,
                )
            module.weight.data.copy_(result["W_hat"].to(module.weight.dtype).to(args.device))
            print(
                f"[block {i}/{args.num_blocks - 1}] {kind}.{proj_name}: "
                f"{tuple(W.shape)}  X{tuple(X.shape)}  bpw={result['bits_per_weight']:.3f}"
            )

    model = model.to("cpu")
    model.generation_config.do_sample = True
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nSaved sequentially-quantized checkpoint (blocks 0-{args.num_blocks - 1}) to {args.output}")


if __name__ == "__main__":
    main()
