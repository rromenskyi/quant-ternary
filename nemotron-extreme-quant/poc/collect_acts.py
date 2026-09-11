"""Stage A: run Nemotron-3-Nano-4B-BF16 on a small calibration set and capture
real input activations for one representative Linear layer of each mixer
type (Mamba, Attention, MLP), plus the corresponding weight tensor.

Usage:
    python poc/collect_acts.py --model cache/models/Nemotron-3-Nano-4B-BF16 \
        --output poc/calib_cache
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CALIBRATION_PROMPTS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# Write a memoized version of this function:\n",
    "Explain step by step how to reverse a linked list in Python, then implement it.",
    "A train leaves station A at 60 mph and another leaves station B, 300 miles away, at 40 mph "
    "towards A. How long until they meet? Show your reasoning.",
    "class BinarySearchTree:\n    def __init__(self):\n        self.root = None\n\n    def insert(self, value):\n",
    "You are an agent with access to a `run_shell(cmd)` tool. Task: find all Python files "
    "larger than 1MB in the current directory. What tool call do you make first?",
    "Prove that the square root of 2 is irrational.",
    "import numpy as np\n\ndef softmax(x):\n    \"\"\"Numerically stable softmax.\"\"\"\n",
    "What is the time complexity of quicksort in the average and worst case, and why?",
    "SELECT customer_id, SUM(amount) FROM orders WHERE status = 'completed' GROUP BY customer_id "
    "HAVING SUM(amount) > 1000 -- explain what this query does and rewrite it using a CTE",
    "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is prime.\"\"\"\n",
]


def classify_mixer(mixer) -> str | None:
    cls_name = type(mixer).__name__
    if "Mamba" in cls_name:
        return "mamba"
    if "Attention" in cls_name:
        return "attention"
    if "MLP" in cls_name:
        return "mlp"
    return None


TARGET_PROJECTIONS = {
    "mamba": ["in_proj", "out_proj"],
    "attention": ["q_proj", "o_proj"],
    "mlp": ["up_proj", "down_proj"],
}


def pick_target_modules(model, min_layer: int = 4):
    """Pick one Linear module per (mixer_type, projection_name), from the
    first layer of that type at index >= min_layer."""
    seen_types = set()
    targets = {}  # full_name -> nn.Linear
    for i, block in enumerate(model.model.layers):
        if i < min_layer:
            continue
        kind = classify_mixer(block.mixer)
        if kind is None or kind in seen_types:
            continue
        seen_types.add(kind)
        for proj_name in TARGET_PROJECTIONS[kind]:
            module = getattr(block.mixer, proj_name, None)
            if module is not None:
                targets[f"layers.{i}.mixer.{proj_name} ({kind})"] = module
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="poc/calib_cache")
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=False
    )
    model.eval()

    targets = pick_target_modules(model)
    print("Hooking:")
    for name in targets:
        print(f"  - {name}")

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
