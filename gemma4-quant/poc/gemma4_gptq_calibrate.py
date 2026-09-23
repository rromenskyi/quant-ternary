"""GPTQ calibration for google/gemma-4-E4B-it across all three real
components (text decoder, vision tower, audio tower) -- no mlx dependency,
meant to run on a rented CUDA pod (mirrors zimage-quant's calibrate/splice
split: this half runs anywhere with torch+transformers, the mlx-only
splice step happens back on a Mac).

Real submodule names (confirmed directly against the real
model.safetensors header, not guessed):
  text:   model.model.language_model.layers[i].self_attn.{q,k,v,o}_proj
          model.model.language_model.layers[i].mlp.{gate,up,down}_proj
          -- plain nn.Linear, all 42 layers have independent q/k/v/o_proj
          (an earlier assumption that num_kv_shared_layers=18 means those
          layers lack their own k/v_proj weights was WRONG -- verified by
          reading the real safetensors header: every layer has full-size
          k_proj/v_proj, the only per-layer difference is head_dim
          (256 local / 512 global) depending on layer_types).
  vision: model.model.vision_tower.encoder.layers[i].self_attn.{q,k,v,o}_proj.linear
          model.model.vision_tower.encoder.layers[i].mlp.{gate,up,down}_proj.linear
  audio:  model.model.audio_tower.layers[i].self_attn.{q,k,v,post}.linear
          model.model.audio_tower.layers[i].feed_forward{1,2}.ffw_layer_{1,2}.linear
          model.model.audio_tower.layers[i].lconv1d.{linear_start,linear_end}.linear

Vision/audio Linears are wrapped in Gemma4ClippableLinear (see
transformers' modeling_gemma4.py): the real nn.Linear lives at
`<name>.linear`, with sibling scalar buffers input_min/input_max/
output_min/output_max used only for activation clamping (numerical
stability, not quantization) -- hooking `.linear` directly captures the
exact input the real weight matrix sees (post input-clamp), and those
scalar buffers are copied through unchanged by the splice step since
GPTQ never touches them.

Usage (on the pod, CUDA):
    python gemma4_gptq_calibrate.py \
        --model-dir google/gemma-4-E4B-it \
        --output-dir /workspace/gemma4-corrected \
        --component text --layers all --device cuda \
        --bits 8 --group-size 64

    python gemma4_gptq_calibrate.py --component vision ... (same flags)
    python gemma4_gptq_calibrate.py --component audio ...
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

TEXT_SUBMODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
VISION_SUBMODULES = [
    "self_attn.q_proj.linear", "self_attn.k_proj.linear", "self_attn.v_proj.linear", "self_attn.o_proj.linear",
    "mlp.gate_proj.linear", "mlp.up_proj.linear", "mlp.down_proj.linear",
]
AUDIO_SUBMODULES = [
    "self_attn.q_proj.linear", "self_attn.k_proj.linear", "self_attn.v_proj.linear", "self_attn.post.linear",
    "feed_forward1.ffw_layer_1.linear", "feed_forward1.ffw_layer_2.linear",
    "feed_forward2.ffw_layer_1.linear", "feed_forward2.ffw_layer_2.linear",
    "lconv1d.linear_start.linear", "lconv1d.linear_end.linear",
]

TEXT_PROMPTS = [
    "The history of the Roman Empire spans over a thousand years, beginning with the "
    "traditional founding of Rome in 753 BC and continuing through the fall of the "
    "Western Empire in 476 AD.",
    "Photosynthesis is the process by which green plants and some other organisms use "
    "sunlight to synthesize foods from carbon dioxide and water.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]\n    return left",
    "Q: What is the capital of France?\nA: The capital of France is Paris, a city known "
    "for its art, culture, and history.",
    "Climate change refers to long-term shifts in temperatures and weather patterns, "
    "mainly caused by human activities, especially the burning of fossil fuels.",
    "Once upon a time, in a small village nestled between two mountains, there lived an "
    "old woman who could speak to birds.",
    "The stock market experienced significant volatility today as investors reacted to "
    "new inflation data released by the Federal Reserve.",
    "To install this package, run pip install followed by the package name, then import "
    "it in your Python script before use.",
]

# Real, stable, widely-used COCO validation images (the same handful
# appear across countless HF/transformers doc examples) -- genuinely
# varied photographic content, not synthetic noise.
IMAGE_URLS = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # two cats on a couch
    "http://images.cocodataset.org/val2017/000000000139.jpg",
    "http://images.cocodataset.org/val2017/000000000285.jpg",
    "http://images.cocodataset.org/val2017/000000000632.jpg",
    "http://images.cocodataset.org/val2017/000000000724.jpg",
    "http://images.cocodataset.org/val2017/000000000776.jpg",
]


def layer_indices(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    return [int(x) for x in spec.split(",")]


def progress_path(output_dir: Path, component: str) -> Path:
    return output_dir / f"gptq_progress_{component}.json"


def load_progress(output_dir: Path, component: str) -> set[int]:
    p = progress_path(output_dir, component)
    if not p.exists():
        return set()
    return set(json.loads(p.read_text())["done_layers"])


def save_progress(output_dir: Path, component: str, done_layers: set[int]) -> None:
    progress_path(output_dir, component).write_text(json.dumps({"done_layers": sorted(done_layers)}))


def load_calibration_inputs(component: str, args, processor):
    """Returns a list of kwargs dicts, one per calibration example, ready
    to feed straight to the relevant sub-module's forward pass."""
    if component == "text":
        examples = []
        for prompt in TEXT_PROMPTS[: args.examples]:
            enc = processor(text=prompt, return_tensors="pt")
            examples.append({"input_ids": enc["input_ids"]})
        return examples

    if component == "vision":
        import requests
        from PIL import Image

        examples = []
        for url in IMAGE_URLS[: args.examples]:
            img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            # Processor already returns pre-patchified pixel_values
            # ([batch, num_patches, patch_size*patch_size*3]) plus
            # per-patch (x,y) position ids under "image_position_ids" --
            # Gemma4VisionModel.forward's own kwarg is named
            # "pixel_position_ids", so rename on the way in.
            enc = processor(images=img, return_tensors="pt")
            examples.append({
                "pixel_values": enc["pixel_values"],
                "pixel_position_ids": enc["image_position_ids"],
            })
        return examples

    if component == "audio":
        import io

        import soundfile as sf
        from datasets import Audio, load_dataset

        # decode=False + manual soundfile decode sidesteps torchcodec,
        # which failed to load its CUDA image-decoding library on this
        # pod (libnvrtc.so.13 missing) -- soundfile only needs libsndfile,
        # already present, and reads the exact same underlying FLAC bytes.
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(decode=False))
        examples = []
        for i in range(min(args.examples, len(ds))):
            raw = ds[i]["audio"]
            array, sampling_rate = sf.read(io.BytesIO(raw["bytes"]))
            enc = processor(audio=array, sampling_rate=sampling_rate, return_tensors="pt")
            # Feature-extractor output key name varies by processor version;
            # try the common ones rather than hardcoding one.
            key = next(k for k in ("input_features", "audio_values", "input_values") if k in enc)
            examples.append({key: enc[key]})
        return examples

    raise ValueError(component)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="HF repo id or local path, e.g. google/gemma-4-E4B-it")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--component", required=True, choices=["text", "vision", "audio"])
    parser.add_argument("--layers", default="all")
    parser.add_argument("--layers-per-batch", type=int, default=6)
    parser.add_argument("--max-rows-per-module", type=int, default=8192)
    parser.add_argument("--examples", type=int, default=6, help="calibration prompts/images/audio clips to use")
    parser.add_argument(
        "--bits", type=int, default=None,
        help="uniform bit-width -- mutually exclusive with --attn-bits/--ffn-bits",
    )
    parser.add_argument("--attn-bits", type=int, default=None, help="bits for attention/self_attn submodules")
    parser.add_argument("--ffn-bits", type=int, default=None, help="bits for mlp/feed_forward submodules")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    args = parser.parse_args()

    if args.bits is not None and (args.attn_bits is not None or args.ffn_bits is not None):
        parser.error("--bits is mutually exclusive with --attn-bits/--ffn-bits")
    if args.bits is None and (args.attn_bits is None or args.ffn_bits is None):
        if args.attn_bits is None and args.ffn_bits is None:
            args.bits = 8
        else:
            parser.error("--attn-bits and --ffn-bits must both be given together")
    attn_bits = args.attn_bits if args.attn_bits is not None else args.bits
    ffn_bits = args.ffn_bits if args.ffn_bits is not None else args.bits

    def bits_for(key: str) -> int:
        return attn_bits if ("attn" in key or "lconv1d" in key) else ffn_bits

    submodules = {"text": TEXT_SUBMODULES, "vision": VISION_SUBMODULES, "audio": AUDIO_SUBMODULES}[args.component]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    done_layers = load_progress(output_dir, args.component)
    if done_layers:
        print(f"Resuming {args.component}: {len(done_layers)} layer(s) already done: {sorted(done_layers)}", flush=True)

    print(f"Loading google/gemma-4-E4B-it ({args.model_dir}) for {args.component} calibration ...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16)
    model = model.to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_dir)

    tower = {
        "text": model.model.language_model,
        "vision": model.model.vision_tower,
        "audio": model.model.audio_tower,
    }[args.component]
    layers_list = tower.layers if args.component != "vision" else tower.encoder.layers

    num_layers = len(layers_list)
    requested = layer_indices(args.layers, num_layers)
    layers = [i for i in requested if i not in done_layers]
    if not layers:
        print(f"Nothing left to do for {args.component}.", flush=True)
        print(f"GEMMA4_GPTQ_CALIBRATE_DONE:{args.component}")
        return
    print(f"Calibrating {args.component} layers {layers} of {num_layers}, {args.layers_per_batch} at a time ...", flush=True)

    print(f"Loading {args.examples} real calibration example(s) for {args.component} ...", flush=True)
    calib_inputs = load_calibration_inputs(args.component, args, processor)

    batches = [layers[i : i + args.layers_per_batch] for i in range(0, len(layers), args.layers_per_batch)]

    for batch_num, batch in enumerate(batches):
        print(f"\n--- {args.component} batch {batch_num + 1}/{len(batches)}: layers {batch} ---", flush=True)
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
            layer = layers_list[i]
            for sub in submodules:
                try:
                    module = layer.get_submodule(sub)
                except AttributeError:
                    continue  # e.g. a submodule that doesn't exist on every layer
                key = f"layers.{i}.{sub}"
                target_modules[key] = module
                handles.append(module.register_forward_pre_hook(make_hook(key)))

        print(f"  Running up to {len(calib_inputs)} calibration example(s) ...", flush=True)
        with torch.no_grad():
            for i, kwargs in enumerate(calib_inputs):
                print(f"    [{i + 1}/{len(calib_inputs)}]", flush=True)
                kwargs = {k: v.to(args.device) for k, v in kwargs.items()}
                if args.component == "text":
                    model.model.language_model(**kwargs)
                else:
                    tower(**kwargs)
                if all(row_counts.get(k, 0) >= args.max_rows_per_module for k in target_modules):
                    print("    (row cap reached for all modules in this batch, skipping remaining examples)", flush=True)
                    break

        for h in handles:
            h.remove()

        print("  Running GPTQ correction for this batch ...", flush=True)
        corrected: dict[str, torch.Tensor] = {}
        key_bits: dict[str, int] = {}
        for key, module in target_modules.items():
            if key not in captured:
                print(f"    {key}: no activations captured (module never ran), skipping", flush=True)
                continue
            bits = bits_for(key)
            X = torch.cat(captured[key], dim=0)
            W = module.weight.detach().to(torch.float32).cpu()
            result = gptq_nbit(W, X, bits=bits, group_size=args.group_size, device=args.device, scheme="affine")
            corrected[key] = result["W_hat"].to(torch.bfloat16).contiguous()
            key_bits[key] = bits
            print(f"    {key}: {tuple(W.shape)}, bits={bits}, calib rows={X.shape[0]}", flush=True)

        batch_path = output_dir / f"{args.component}_batch_{'_'.join(str(i) for i in batch)}.safetensors"
        save_file(corrected, str(batch_path), metadata={"group_size": str(args.group_size), "key_bits": json.dumps(key_bits)})
        print(f"  Wrote {batch_path}", flush=True)
        done_layers.update(batch)
        save_progress(output_dir, args.component, done_layers)

        del captured, row_counts, corrected
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

    print(f"GEMMA4_GPTQ_CALIBRATE_DONE:{args.component}")


if __name__ == "__main__":
    main()
