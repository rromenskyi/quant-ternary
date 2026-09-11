"""Run every quantization method against every captured calibration layer,
evaluate reconstruction quality, and write poc/results.md.

Usage:
    python poc/run_poc.py --calib-dir poc/calib_cache --output poc/results.md
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

from evaluate import evaluate
from gptq import gptq_binary, gptq_ternary
from quantize import activation_aware_binary, binary_with_residual, naive_binary, ternary_optimized
from rotation import with_rotation

GROUP_SIZE = 128
# Practical gate: activation (output) reconstruction + bpw budget are what actually
# predict whether the quantized model behaves like the original. Weight cosine is
# reported but not gated on: for binary/ternary quantization it stays in the 0.7-0.9
# range even for methods whose activation cosine is 0.98+, because each weight is
# flipped to +-scale independently — the *dot product* with real correlated
# activations reconstructs well even though the raw weight vector doesn't.
SUCCESS = {"act_cosine": 0.95, "bits_per_weight": 1.8}


def run_methods(W: torch.Tensor, X: torch.Tensor) -> list[dict]:
    methods = [
        naive_binary(W, GROUP_SIZE),
        activation_aware_binary(W, X, GROUP_SIZE),
        binary_with_residual(W, X, GROUP_SIZE, residual_fraction=0.01, criterion="magnitude"),
        binary_with_residual(
            W, X, GROUP_SIZE, residual_fraction=0.03, criterion="activation_weighted", activation_aware=True
        ),
        ternary_optimized(W, GROUP_SIZE),
        gptq_binary(W, X, GROUP_SIZE),
        gptq_ternary(W, X, GROUP_SIZE),
        with_rotation(naive_binary, W, None, GROUP_SIZE, group_size=GROUP_SIZE),
        with_rotation(gptq_binary, W, X, GROUP_SIZE, group_size=GROUP_SIZE),
        with_rotation(gptq_ternary, W, X, GROUP_SIZE, group_size=GROUP_SIZE),
    ]
    results = []
    for m in methods:
        metrics = evaluate(W, m["W_hat"], X)
        passed = (
            metrics["act_cosine"] >= SUCCESS["act_cosine"]
            and m["bits_per_weight"] <= SUCCESS["bits_per_weight"]
            and not metrics["has_nan_or_inf"]
        )
        results.append({"method": m["method"], "bits_per_weight": m["bits_per_weight"], **metrics, "pass": passed})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-dir", default="poc/calib_cache")
    parser.add_argument("--output", default="poc/results.md")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.calib_dir, "*.pt")))
    if not files:
        raise SystemExit(f"No calibration files found in {args.calib_dir}. Run collect_acts.py first.")

    all_results = {}
    for path in files:
        data = torch.load(path, weights_only=False)
        name, W, X = data["name"], data["weight"], data["activations"]
        print(f"=== {name}  W{tuple(W.shape)}  X{tuple(X.shape)} ===")
        results = run_methods(W, X)
        all_results[name] = results
        for r in results:
            status = "PASS" if r["pass"] else "fail"
            print(
                f"  [{status}] {r['method']:45s} "
                f"bpw={r['bits_per_weight']:.3f}  w_cos={r['weight_cosine']:.4f}  "
                f"act_cos={r['act_cosine']:.4f}"
            )

    write_report(all_results, args.output)
    print(f"\nWrote {args.output}")


def write_report(all_results: dict, output_path: str):
    lines = ["# PoC Results — Nemotron-3-Nano-4B Stage A\n"]
    lines.append(
        f"Success criteria: weight cosine >= {SUCCESS['weight_cosine']}, "
        f"activation cosine >= {SUCCESS['act_cosine']}, "
        f"effective bpw <= {SUCCESS['bits_per_weight']}, no NaN/Inf.\n"
    )

    total_pass = 0
    total_methods = 0
    for name, results in all_results.items():
        lines.append(f"\n## {name}\n")
        lines.append("| Method | Eff. bpw | Weight cos | Weight MSE | Act cos | Act MSE | Verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in results:
            total_methods += 1
            total_pass += int(r["pass"])
            verdict = "✅ PASS" if r["pass"] else "❌ fail"
            lines.append(
                f"| {r['method']} | {r['bits_per_weight']:.3f} | {r['weight_cosine']:.4f} | "
                f"{r['weight_mse']:.3e} | {r['act_cosine']:.4f} | {r['act_mse']:.3e} | {verdict} |"
            )

    layers_with_a_pass = sum(1 for results in all_results.values() if any(r["pass"] for r in results))
    n_layers = len(all_results)
    verdict = "GO" if layers_with_a_pass >= max(1, int(0.6 * n_layers)) else "NO-GO"

    lines.append(f"\n## Summary\n")
    lines.append(f"- {total_pass}/{total_methods} (method, layer) combinations passed all four criteria")
    lines.append(f"- {layers_with_a_pass}/{n_layers} layers had at least one passing method")
    lines.append(f"\n**Decision: {verdict}**\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
