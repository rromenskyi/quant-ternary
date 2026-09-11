"""Capture calibration activations for every quantization-eligible Linear
layer in the model, in a single forward pass. Unlike collect_acts.py (one
representative layer per mixer type, for method comparison), this is meant
to feed full-model quantization.

Protected tensors — embeddings, lm_head, norm layers — are never hooked;
they stay at full precision per docs/spec.md's protected-tensor list.

Usage:
    python poc/collect_acts_full.py --model cache/models/Nemotron-3-Nano-4B-BF16 \
        --output poc/calib_cache_full
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_acts import CALIBRATION_PROMPTS, classify_mixer

ELIGIBLE_PROJECTIONS = {
    "mamba": ["in_proj", "out_proj"],
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["up_proj", "down_proj"],
}


def collect_targets(model) -> dict[str, nn.Linear]:
    targets = {}
    for i, block in enumerate(model.model.layers):
        kind = classify_mixer(block.mixer)
        if kind is None:
            continue
        for proj_name in ELIGIBLE_PROJECTIONS[kind]:
            module = getattr(block.mixer, proj_name, None)
            if isinstance(module, nn.Linear):
                targets[f"layers.{i}.mixer.{proj_name} ({kind})"] = module
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="poc/calib_cache_full")
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=False)
    model.eval()

    targets = collect_targets(model)
    print(f"Hooking {len(targets)} linear layers across {len(model.model.layers)} blocks:")
    kind_counts = {}
    for name in targets:
        kind = name.split("(")[-1].rstrip(")")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    for kind, count in kind_counts.items():
        print(f"  {kind}: {count}")

    captured = {name: [] for name in targets}
    handles = []

    def make_hook(name):
        def hook(module, inputs):
            captured[name].append(inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]))

        return hook

    for name, module in targets.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    with torch.no_grad():
        for i, prompt in enumerate(CALIBRATION_PROMPTS):
            print(f"[{i + 1}/{len(CALIBRATION_PROMPTS)}] forward pass ...")
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            model(**inputs, use_cache=False)

    for h in handles:
        h.remove()

    for name, module in targets.items():
        X = torch.cat(captured[name], dim=0)
        W = module.weight.detach().to(torch.float32).clone()
        safe_name = name.split(" ")[0].replace(".", "_")
        out_path = os.path.join(args.output, f"{safe_name}.pt")
        torch.save({"name": name, "weight": W, "activations": X}, out_path)
        print(f"Saved {name}: W{tuple(W.shape)}, X{tuple(X.shape)} -> {out_path}")


if __name__ == "__main__":
    main()
