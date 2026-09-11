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
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--salient-fraction", type=float, default=0.03)
    parser.add_argument("--salient-criterion", default="activation_weighted")
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=False)
    model = model.to(args.device)
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
            W = module.weight.detach().to(torch.float32).cpu()
            X = activations[proj_name].cpu()
            result = rot_gptq_salient(
                W,
                X,
                GROUP_SIZE,
                salient_fraction=args.salient_fraction,
                criterion=args.salient_criterion,
                device=args.device,
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
