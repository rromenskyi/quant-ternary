"""GPTQ-calibrates Z-Image-Turbo's (Tongyi-MAI, diffusers ZImagePipeline)
transformer blocks, then splices the corrected weights into an mflux-saved
MLX checkpoint -- same idea as nemotron-extreme-quant's gptq_stock_convert.py
applied to a diffusion transformer instead of a language model.

Why this works at all: mflux's own `mflux-save --quantize N` output uses
the exact same on-disk convention this project already produces for LLMs --
plain `mx.quantize(mode="affine")` triples (weight/scales/biases, uint32-
packed, group_size derived from shape) under the SAME key names diffusers'
own ZImageTransformer2DModel uses (mflux is a "line-by-line port", per its
own README -- confirmed directly: `layers.0.attention.to_q.weight` etc.
match 1:1 between the two). So a GPTQ-corrected, already-on-grid weight can
be re-quantized with mx.quantize and dropped straight into a copy of
mflux's own safetensors shards, with zero format conversion.

Calibration data: unlike a language model (calibrate on real token
sequences), a diffusion transformer's real workload is a denoising loop --
this runs ZImagePipeline's own forward pass for a handful of real prompts
at the model's own default step count, capturing each target Linear's
input activation via a forward-pre-hook at every step of every prompt, and
pools all of it into one calibration set per layer (one-shot GPTQ, not
sequential -- see the --layers docstring below for why that's a
deliberate, documented simplification for a first version, not an
oversight).

Checkpointing: each batch's GPTQ-corrected weights are spliced into
--output-dir IMMEDIATELY after that batch finishes (not accumulated in
memory and written once at the end) -- a real full-30-layer CPU run costs
hours, and this project has already lost a full batch's worth of compute
once to a memory-pressure kill because nothing had reached disk yet. A
`gptq_progress.json` sidecar records which layer indices are already
corrected on disk, so re-running the same --output-dir after a crash/kill
skips finished layers instead of redoing them.

Usage (small-subset validation run):
    python poc/zimage_gptq.py \
        --hf-pipeline-dir ~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/<sha> \
        --mflux-saved-dir /tmp/zimage-8bit-test \
        --output-dir /tmp/zimage-8bit-gptq \
        --layers 0,1 --prompts 2 --bits 8 --group-size 64

Usage (full 30-layer run, resumable -- re-run the same command after a kill):
    python poc/zimage_gptq.py \
        --hf-pipeline-dir ~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/<sha> \
        --mflux-saved-dir /tmp/zimage-8bit-test \
        --output-dir /tmp/zimage-8bit-gptq \
        --layers all --prompts 8 --bits 8 --group-size 64
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nemotron-extreme-quant" / "poc"))
from gptq import gptq_nbit  # noqa: E402

from diffusers import ZImagePipeline  # noqa: E402

# Every diffusers Linear this project's GPTQ knows how to correct, inside
# one ZImageTransformerBlock -- attention q/k/v/o and the gated-MLP
# w1/w2/w3, mirroring exactly the attention+MLP shape gptq_stock_convert.py
# already handles for language models. adaLN_modulation (diffusion-specific
# conditioning) is deliberately excluded: tiny (15360x256 vs 3840x3840+),
# and not part of mflux's own --quantize sweep either (confirmed: still
# present as a plain f32/bf16 tensor, not weight/scales/biases, in the
# mflux-save output this script reads).
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


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


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
    parser.add_argument(
        "--hf-pipeline-dir", required=True,
        help="real diffusers pipeline dir (has model_index.json) -- e.g. the HF cache snapshot "
        "mflux itself already downloaded, used here for calibration (PyTorch, bf16)",
    )
    parser.add_argument(
        "--mflux-saved-dir", required=True,
        help="output of `mflux-save --model z-image-turbo` (MLX, already quantized) -- the "
        "template this script copies (once) and patches with GPTQ-corrected weights",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,1", help='comma-separated indices, or "all" for all 30')
    parser.add_argument(
        "--layers-per-batch", type=int, default=3,
        help="how many layers' hooks are live at once -- bounds peak activation memory to "
        "roughly this many layers' worth regardless of --layers, at the cost of re-running "
        "the calibration denoising loop once per batch instead of once total. Each batch's "
        "result is written to --output-dir immediately, so this also bounds how much work a "
        "kill/crash mid-run can lose.",
    )
    parser.add_argument(
        "--max-rows-per-module", type=int, default=4096,
        help="cap on calibration rows kept per Linear layer -- without this, capturing every "
        "patch position at every step of every prompt scales to tens of GB per module "
        "(confirmed live: this is what actually exhausted memory, not layer count by itself); "
        "GPTQ's Hessian estimate doesn't need every token, just a representative sample. NOTE: "
        "this only stops the hook from *recording* more rows -- it does NOT shorten the "
        "diffusion loop itself (confirmed live: a capped module still sits through the "
        "remaining steps of whatever prompt is in flight). Use --steps/--prompts to actually "
        "bound wall-clock time.",
    )
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument(
        "--steps", type=int, default=9,
        help="denoising steps per calibration prompt -- GPTQ needs a representative activation "
        "sample, not a finished-quality image, so this can be cut well below the model's own "
        "generation default (9) to trade calibration precision for wall-clock time",
    )
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = parser.parse_args()

    saved_dir = Path(args.mflux_saved_dir)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Copying template checkpoint {saved_dir} -> {output_dir} ...", flush=True)
        shutil.copytree(saved_dir, output_dir)

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
        print("ZIMAGE_GPTQ_DONE")
        return

    print(f"Calibrating layers {layers} of {num_layers}, {args.layers_per_batch} at a time ...", flush=True)

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
                # Once every target module in this batch has hit its row
                # cap, further prompts add nothing -- stop early rather
                # than paying for a whole extra denoising loop whose
                # activations the hooks above would just discard.
                if all(row_counts.get(k, 0) >= args.max_rows_per_module for k in target_modules):
                    print("    (row cap reached for all modules in this batch, skipping remaining prompts)", flush=True)
                    break

        for h in handles:
            h.remove()

        print("  Running GPTQ correction for this batch ...", flush=True)
        corrected: dict[str, torch.Tensor] = {}
        for key, module in target_modules.items():
            X = torch.cat(captured[key], dim=0)
            W = module.weight.detach().to(torch.float32).cpu()
            result = gptq_nbit(
                W, X, bits=args.bits, group_size=args.group_size, device="cpu", scheme="affine"
            )
            corrected[key] = result["W_hat"].to(torch.bfloat16)
            print(f"    {key}: {tuple(W.shape)}, calib rows={X.shape[0]}", flush=True)

        print(f"  Writing batch {batch_num + 1}/{len(batches)} to {output_dir} ...", flush=True)
        apply_corrections_to_checkpoint(output_dir, corrected, args.bits, args.group_size)
        done_layers.update(batch)
        save_progress(output_dir, done_layers)

        # Explicitly drop this batch's activations/corrections before the
        # next batch starts accumulating -- an explicit del + gc.collect()
        # (and MPS's own cache, which holds its own pool of recently-freed
        # GPU buffers rather than returning them to the OS immediately) is
        # what actually keeps peak memory bounded across batches instead of
        # monotonically climbing.
        del captured, row_counts, corrected
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    print("ZIMAGE_GPTQ_DONE")


def apply_corrections_to_checkpoint(
    output_dir: Path, corrected: dict[str, torch.Tensor], bits: int, group_size: int
) -> None:
    import mlx.core as mx

    transformer_dir = output_dir / "transformer"
    index_path = transformer_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    # Group corrected keys by which shard they currently live in, so each
    # shard is only opened/rewritten once.
    by_shard: dict[str, list[str]] = {}
    for key in corrected:
        shard = weight_map[f"{key}.weight"]
        by_shard.setdefault(shard, []).append(key)

    for shard_name, keys in by_shard.items():
        shard_path = transformer_dir / shard_name
        # mx.load() silently drops the safetensors header's own
        # `__metadata__` entry (confirmed live: it's only visible by
        # reading the raw header directly) -- mflux's own weight loader
        # reads that per-shard metadata (quantization_level, mflux_version)
        # to decide which modules to reconstruct as QuantizedLinear at all.
        # Re-saving via mx.save_safetensors() without passing it back
        # through silently drops it, and mflux then treats every tensor in
        # that shard as unquantized -- reproduced live: a plain nn.Linear
        # loading a still-packed uint32 weight tensor straight into
        # `.weight`, crashing on the first matmul with a bogus shape.
        original_metadata = read_header(shard_path).get("__metadata__")
        tensors = dict(mx.load(str(shard_path)))
        for key in keys:
            # bfloat16 isn't a native numpy dtype -- torch.Tensor.numpy()
            # rejects it directly, so bridge through float32 first.
            w = mx.array(corrected[key].to(torch.float32).numpy()).astype(mx.bfloat16)
            wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits, mode="affine")
            tensors[f"{key}.weight"] = wq
            tensors[f"{key}.scales"] = scales
            tensors[f"{key}.biases"] = biases
        mx.eval(*tensors.values())
        mx.save_safetensors(str(shard_path), tensors, metadata=original_metadata)


if __name__ == "__main__":
    main()
