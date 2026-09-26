"""Compares quantized FLUX.2 klein transformers (RTN, GPTQ, ...) against the
bf16 one, on prompts and edits kept out of klein_gptq.py's calibration.

Two measurements per sample:
- **Teacher-forced velocity error** (the main number): each step of the
  quantized model gets exactly the bf16 run's latents as input, and its
  prediction is compared with bf16's, ||v_q - v_bf16|| / ||v_bf16||, mean
  over the 4 steps. No trajectory drift compounds into it, so it ranks
  checkpoints far more stably than image PSNR on one seed.
- **Free run**: the image each model makes by itself, PSNR vs bf16's, and a
  grid (rows = samples, columns = models) to look at.

All models condition on the same prompt embeddings (the deployed 8-bit,
27-layer text encoder of the first checkpoint), so only the transformer
differs.

    python klein_gptq_eval.py --out <dir> --model rtn=<checkpoint> --model gptq=<checkpoint>
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image, ImageDraw

from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.flux2.model.flux2_text_encoder import prompt_encoder as pe
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

from klein_gptq import txt2img_twin

TXT = [
    ("fox", "A red fox sitting in a snowy forest, wildlife photography."),
    ("poster", "A vintage poster that says GRAND OPENING in bold letters."),
    ("portrait", "A portrait of a young man with freckles, natural window light."),
    ("sushi", "A plate of sushi on a slate board, top-down food photography."),
    ("alley", "A cyberpunk alley at night with rain and neon reflections."),
    ("sunflowers", "An oil painting of sunflowers in a blue vase."),
]
EDITS = [
    ("fox_autumn", "Make it autumn with orange leaves instead of snow.", "fox"),
    ("portrait_glasses", "Give him round glasses, keep everything else.", "portrait"),
    ("poster_text", "Change the text to SUMMER SALE.", "poster"),
]
SEED, SIZE, STEPS = 7, 1024, 4


class Recorder:
    """Patched into the scheduler's step: records each prediction, and in
    forced mode returns the reference run's next latents instead."""
    mode = "free"          # "free" | "forced"
    noises: list = []
    nexts: list = []
    forced_nexts: list = []
    original = FlowMatchEulerDiscreteScheduler.step

    @staticmethod
    def step(self, noise, timestep, latents, **kwargs):
        out = Recorder.original(self, noise=noise, timestep=timestep, latents=latents, **kwargs)
        mx.eval(noise, out)
        Recorder.noises.append(np.array(noise.astype(mx.float32)))
        Recorder.nexts.append(np.array(out.astype(mx.float32)))
        if Recorder.mode == "forced":
            return mx.array(Recorder.forced_nexts[len(Recorder.noises) - 1]).astype(out.dtype)
        return out

    @staticmethod
    def reset(mode: str, forced_nexts=None):
        Recorder.mode, Recorder.noises, Recorder.nexts = mode, [], []
        Recorder.forced_nexts = forced_nexts or []


FlowMatchEulerDiscreteScheduler.step = Recorder.step


def prepare(model, cache: dict):
    model.text_encoder = None
    model.tiling_config = TilingConfig(vae_decode_tile_size=256)
    gc.collect()
    mx.clear_cache()
    pe.Flux2PromptEncoder.encode_prompt = staticmethod(lambda prompt, **_: cache[prompt])
    return model, txt2img_twin(model)


def encode_all(checkpoint: str) -> dict:
    model = Flux2KleinEdit(model_path=checkpoint)
    model.text_encoder.layers = model.text_encoder.layers[:27]
    original = pe.Flux2PromptEncoder.encode_prompt
    cache = {}
    for prompt in [p for _, p in TXT] + [p for _, p, _ in EDITS] + [" "]:
        cache[prompt] = original(prompt=prompt, tokenizer=model.tokenizers["qwen3"], text_encoder=model.text_encoder,
                                 num_images_per_prompt=1, max_sequence_length=512, text_encoder_out_layers=(9, 18, 27))
        mx.eval(cache[prompt])
    del model
    gc.collect()
    mx.clear_cache()
    return cache


def samples(out: Path):
    for name, prompt in TXT:
        yield name, prompt, None
    for name, prompt, src in EDITS:
        yield name, prompt, str(out / f"bf16_{src}.png")


def generate(model, twin, prompt, ref):
    kwargs = dict(seed=SEED, prompt=prompt, num_inference_steps=STEPS, width=SIZE, height=SIZE)
    return (model.generate_image(**kwargs, image_paths=[ref]) if ref else twin.generate_image(**kwargs)).image


def psnr(a: Image.Image, b: Image.Image) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    mse = np.mean((x - y) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def grid(out: Path, labels: list[str]) -> None:
    names = [n for n, _, _ in samples(out)]
    side, pad = 256, 24
    canvas = Image.new("RGB", (side * len(labels), (side + pad) * len(names) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    for c, label in enumerate(labels):
        draw.text((c * side + 6, 4), label, fill="black")
    for r, name in enumerate(names):
        for c, label in enumerate(labels):
            img = Image.open(out / f"{label}_{name}.png").convert("RGB").resize((side, side))
            canvas.paste(img, (c * side, pad + r * (side + pad)))
        draw.text((6, pad + r * (side + pad) + side + 4), name, fill="black")
    canvas.save(out / "grid.jpg", quality=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint dir (repeatable)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = [m.split("=", 1) for m in args.model]

    # (mflux's compiled predict stays compiled: the patched scheduler step
    # runs outside it.)
    t0 = time.time()
    cache = encode_all(models[0][1])
    print(f"prompts encoded ({time.time() - t0:.0f}s)", flush=True)

    # bf16 reference: its trajectory and images.
    reference = {}
    model, twin = prepare(Flux2KleinEdit(quantize=None), cache)
    for name, prompt, ref in samples(out):
        t0 = time.time()
        Recorder.reset("free")
        generate(model, twin, prompt, ref).save(out / f"bf16_{name}.png")
        reference[name] = (Recorder.noises, Recorder.nexts)
        print(f"bf16 {name} {time.time() - t0:.0f}s", flush=True)
    del model, twin
    gc.collect()
    mx.clear_cache()

    results = {}
    for label, checkpoint in models:
        model, twin = prepare(Flux2KleinEdit(model_path=checkpoint), cache)
        results[label] = {}
        for name, prompt, ref in samples(out):
            t0 = time.time()
            ref_noises, ref_nexts = reference[name]
            Recorder.reset("forced", ref_nexts)
            generate(model, twin, prompt, ref)
            errors = [float(np.linalg.norm(q - r) / np.linalg.norm(r)) for q, r in zip(Recorder.noises, ref_noises)]
            Recorder.reset("free")
            image = generate(model, twin, prompt, ref)
            image.save(out / f"{label}_{name}.png")
            results[label][name] = {"velocity_rel_err": errors, "velocity_rel_err_mean": float(np.mean(errors)),
                                    "psnr_vs_bf16": psnr(image, Image.open(out / f"bf16_{name}.png").convert("RGB"))}
            print(f"{label} {name} v-err {np.mean(errors):.4f} psnr {results[label][name]['psnr_vs_bf16']:.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del model, twin
        gc.collect()
        mx.clear_cache()

    summary = {}
    for label, per in results.items():
        txt = [per[n] for n, _ in TXT]
        edit = [per[n] for n, _, _ in EDITS]
        mean = lambda rows, k: float(np.mean([r[k] for r in rows]))
        summary[label] = {"velocity_rel_err_txt": mean(txt, "velocity_rel_err_mean"),
                          "velocity_rel_err_edit": mean(edit, "velocity_rel_err_mean"),
                          "psnr_txt": mean(txt, "psnr_vs_bf16"), "psnr_edit": mean(edit, "psnr_vs_bf16")}
    json.dump({"summary": summary, "per_sample": results, "seed": SEED, "size": SIZE}, open(out / "results.json", "w"), indent=2)
    grid(out, ["bf16"] + [label for label, _ in models])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
