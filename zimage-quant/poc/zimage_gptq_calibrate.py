"""CUDA/CPU-friendly calibration+GPTQ stage for Z-Image-Turbo, split out of
zimage_gptq.py so this half can run on a rented GPU pod (no mlx dependency
at all -- mlx is Apple-only) while the final splice into mflux's MLX
checkpoint happens back on the Mac via zimage_gptq_splice.py.

Same calibration idea as zimage_gptq.py: run ZImagePipeline's own denoising
loop for a handful of real prompts, capture each target Linear's input
activation via a forward-pre-hook, run gptq_nbit (Hessian-corrected,
already on mlx's exact affine grid -- see gptq.py's scheme="affine" note),
and write each batch's corrected weights to --output-dir IMMEDIATELY
(safetensors, one file per batch) with a `gptq_progress.json` sidecar --
same crash-safety reasoning as zimage_gptq.py: a kill mid-run must lose at
most one in-flight batch, not everything.

Usage (on the pod, CUDA):
    python zimage_gptq_calibrate.py \
        --hf-pipeline-dir /workspace/zimage-turbo \
        --output-dir /workspace/zimage-corrected \
        --layers all --device cuda --prompts 4 --steps 9 \
        --bits 8 --group-size 64

Then rsync --output-dir back to the Mac and run zimage_gptq_splice.py
against it plus an mflux-saved-dir template.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gptq import gptq_nbit  # noqa: E402

from diffusers import ZImagePipeline  # noqa: E402

TARGET_SUBMODULES = [
    "attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0",
    "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
]

CALIBRATION_PROMPTS = [
    "A photo of a red apple on a wooden table, natural lighting.",
    "A futuristic city skyline at sunset, digital art.",
    "A golden retriever running on a beach, sunny day.",
    "A cup of coffee next to an open book, cozy morning light.",
    "A mountain landscape with a lake reflecting the sky.",
    "A close-up portrait of an elderly man, black and white photography.",
    "A bowl of fresh fruit on a kitchen counter.",
    "A spaceship flying through a nebula, sci-fi concept art.",
]


def layer_indices(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    return [int(x) for x in spec.split(",")]


def progress_path(output_dir: Path) -> Path:
    return output_dir / "gptq_progress.json"


def load_progress(output_dir: Path) -> set[int]:
    p = progress_path(output_dir)
    if not p.exists():
        return set()
    return set(json.loads(p.read_text())["done_layers"])


def save_progress(output_dir: Path, done_layers: set[int]) -> None:
    progress_path(output_dir).write_text(json.dumps({"done_layers": sorted(done_layers)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-pipeline-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,1", help='comma-separated indices, or "all" for all 30')
    parser.add_argument("--layers-per-batch", type=int, default=3)
    parser.add_argument("--max-rows-per-module", type=int, default=4096)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument(
        "--bits", type=int, default=None,
        help="uniform bit-width for both attention and feed_forward -- mutually exclusive with "
        "--attn-bits/--ffn-bits",
    )
    parser.add_argument(
        "--attn-bits", type=int, default=None,
        help="bit-width for attention.to_q/to_k/to_v/to_out.0 (component-type mixed precision, "
        "e.g. 8-bit attention + 4-bit feed_forward -- attention is the smaller of the two "
        "component types by param count in Z-Image-Turbo, ~1.77B vs feed_forward's ~3.54B "
        "across all 30 layers, so protecting it costs relatively little)",
    )
    parser.add_argument("--ffn-bits", type=int, default=None, help="bit-width for feed_forward.w1/w2/w3")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    args = parser.parse_args()

    if args.bits is not None and (args.attn_bits is not None or args.ffn_bits is not None):
        parser.error("--bits is mutually exclusive with --attn-bits/--ffn-bits")
    if args.bits is None and (args.attn_bits is None or args.ffn_bits is None):
        if args.attn_bits is None and args.ffn_bits is None:
            args.bits = 8  # preserve the old default when nothing is specified
        else:
            parser.error("--attn-bits and --ffn-bits must both be given together")
    attn_bits = args.attn_bits if args.attn_bits is not None else args.bits
    ffn_bits = args.ffn_bits if args.ffn_bits is not None else args.bits

    def bits_for(key: str) -> int:
        return attn_bits if ".attention." in key else ffn_bits

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    done_layers = load_progress(output_dir)
    if done_layers:
        print(f"Resuming: {len(done_layers)} layer(s) already corrected on disk: {sorted(done_layers)}", flush=True)

    print("Loading diffusers ZImagePipeline (bf16) for calibration ...", flush=True)
    pipe = ZImagePipeline.from_pretrained(args.hf_pipeline_dir, torch_dtype=torch.bfloat16)
    pipe = pipe.to(args.device)
    model = pipe.transformer
    model.eval()

    num_layers = len(model.layers)
    requested = layer_indices(args.layers, num_layers)
    layers = [i for i in requested if i not in done_layers]
    skipped = [i for i in requested if i in done_layers]
    if skipped:
        print(f"Skipping already-done layers: {skipped}", flush=True)
    if not layers:
        print("Nothing left to do -- all requested layers already corrected on disk.", flush=True)
        print("ZIMAGE_GPTQ_CALIBRATE_DONE")
        return

    print(f"Calibrating layers {layers} of {num_layers}, {args.layers_per_batch} at a time on {args.device} ...", flush=True)
    batches = [layers[i : i + args.layers_per_batch] for i in range(0, len(layers), args.layers_per_batch)]

    for batch_num, batch in enumerate(batches):
        print(f"\n--- batch {batch_num + 1}/{len(batches)}: layers {batch} ---", flush=True)
        captured: dict[str, list[torch.Tensor]] = {}
        row_counts: dict[str, int] = {}
        handles = []

        def make_hook(key: str):
            def hook(module, inputs):
                if row_counts.get(key, 0) >= args.max_rows_per_module:
                    return
                x = inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]).cpu()
                captured.setdefault(key, []).append(x)
                row_counts[key] = row_counts.get(key, 0) + x.shape[0]
            return hook

        target_modules: dict[str, torch.nn.Linear] = {}
        for i in batch:
            block = model.layers[i]
            for sub in TARGET_SUBMODULES:
                module = block.get_submodule(sub)
                key = f"layers.{i}.{sub}"
                target_modules[key] = module
                handles.append(module.register_forward_pre_hook(make_hook(key)))

        print(f"  Running up to {args.prompts} calibration prompt(s) at {args.steps} steps ...", flush=True)
        with torch.no_grad():
            for i, prompt in enumerate(CALIBRATION_PROMPTS[: args.prompts]):
                print(f"    [{i + 1}/{args.prompts}] {prompt!r}", flush=True)
                pipe(prompt=prompt, num_inference_steps=args.steps, height=512, width=512)
                if all(row_counts.get(k, 0) >= args.max_rows_per_module for k in target_modules):
                    print("    (row cap reached for all modules in this batch, skipping remaining prompts)", flush=True)
                    break

        for h in handles:
            h.remove()

        print("  Running GPTQ correction for this batch ...", flush=True)
        corrected: dict[str, torch.Tensor] = {}
        key_bits: dict[str, int] = {}
        for key, module in target_modules.items():
            bits = bits_for(key)
            X = torch.cat(captured[key], dim=0)
            W = module.weight.detach().to(torch.float32).cpu()
            result = gptq_nbit(
                W, X, bits=bits, group_size=args.group_size, device=args.device, scheme="affine"
            )
            corrected[key] = result["W_hat"].to(torch.bfloat16).contiguous()
            key_bits[key] = bits
            print(f"    {key}: {tuple(W.shape)}, bits={bits}, calib rows={X.shape[0]}", flush=True)

        batch_path = output_dir / f"batch_{'_'.join(str(i) for i in batch)}.safetensors"
        save_file(
            corrected, str(batch_path),
            metadata={"group_size": str(args.group_size), "key_bits": json.dumps(key_bits)},
        )
        print(f"  Wrote {batch_path}", flush=True)
        done_layers.update(batch)
        save_progress(output_dir, done_layers)

        del captured, row_counts, corrected
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

    print("ZIMAGE_GPTQ_CALIBRATE_DONE")


if __name__ == "__main__":
    main()
