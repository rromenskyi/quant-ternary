"""GPTQ (Hessian-corrected) quantization of FLUX.2 klein 4B's transformer,
calibrated inside mflux itself on the real 4-step denoising loop -- text to
image and editing (reference-image tokens) both.

How it differs from zimage-quant (diffusers + a row-capped activation
dump): each target Linear is wrapped, and its input's X^T X is summed on the
GPU over every token of every step (no row cap, in^2 floats per Linear
instead of rows x in). gptq_nbit only depends on X through X^T X, so it gets
a square X with X^T X = H (a Cholesky factor). Linears reading the same
input (q/k/v, add_q/k/v) share one H. The Hessians of all 109 Linears don't
fit at once (a single block's to_out alone is 12288^2), so the blocks are
calibrated in passes, each re-running the same deterministic samples.

One-shot GPTQ: every pass calibrates against the bf16 model's own
activations. Each pass's corrected (on-grid, bf16) weights are written to
--work-dir right away with a progress file, so a killed run resumes.

    python klein_gptq.py --template <recipe checkpoint dir> --work-dir <dir> --output-dir <dir>

--template is a checkpoint from klein_make_checkpoint.py (text encoder
8-bit, tokenizer, VAE); the output is a copy of it whose transformer is the
GPTQ one, same format (loads with a plain model_path).
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # mflux's per-image step bars: the log has its own progress

import argparse
import gc
import glob
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nemotron-extreme-quant" / "poc"))
from gptq import gptq_nbit  # noqa: E402

from mflux.models.flux2.model.flux2_text_encoder import prompt_encoder as pe  # noqa: E402
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit  # noqa: E402
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein  # noqa: E402
from mflux.models.common.vae.tiling_config import TilingConfig  # noqa: E402

TXT_PROMPTS = [
    "A photo of a red apple on a wooden table, natural lighting.",
    "A futuristic city skyline at sunset, digital art.",
    "A golden retriever running on a beach, sunny day.",
    "A cozy reading nook with a cup of coffee and an open book, morning light.",
    "A mountain lake reflecting snowy peaks, landscape photography.",
    "A close-up portrait of an elderly woman smiling, soft studio light.",
    "A neon sign that says OPEN 24/7 on a brick wall at night.",
    "A spaceship flying through a colorful nebula, sci-fi concept art.",
    "A watercolor painting of a Venetian canal with gondolas.",
    "A bowl of ramen with a soft-boiled egg, overhead food photography.",
    "An isometric illustration of a tiny medieval village.",
    "A black and white street photo of people crossing in the rain.",
    "A cute cartoon robot waving, flat vector style, white background.",
    "A macro photo of a dewdrop on a green leaf.",
    "A living room interior in Scandinavian style, wide angle.",
    "A chalkboard menu with the words Soup of the Day written on it.",
]
# (instruction, index into TXT_PROMPTS whose image is the reference)
EDITS = [
    ("Make it nighttime with the moon in the sky, keep everything else.", 1),
    ("Change the dog into a black cat, same pose and beach.", 2),
    ("Turn this into a pencil sketch.", 4),
    ("Give her a red knitted hat.", 5),
    ("Replace the text on the sign with CLOSED.", 6),
    ("Add a small wooden boat on the water.", 8),
    ("Make the robot blue and add a balloon in its hand.", 12),
    ("Make it winter, with snow outside the window.", 14),
]
# Up to 768: a Linear's input statistics barely depend on the canvas, and
# 1024x768 beside the Hessians swapped (164s a sample vs ~40s at 768^2).
SIZES = [(768, 768), (768, 512), (512, 768), (512, 512)]


class Progress:
    """Timestamped log lines (stdout and <work>/log.txt) and a one-line
    <work>/status.txt with where the run is and when it should finish."""

    def __init__(self, work: Path, total_passes: int, done_passes: int):
        self.work = work
        self.total = total_passes
        self.done = done_passes
        self.pass_times: list[float] = []
        self.pass_start = time.time()
        self.current = ""

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.work / "log.txt", "a") as f:
            f.write(line + "\n")

    def start_pass(self, n: int) -> None:
        self.pass_start = time.time()
        self.current = f"pass {n + 1}/{self.total}"

    def end_pass(self) -> None:
        self.pass_times.append(time.time() - self.pass_start)
        self.done += 1

    def status(self, stage: str, fraction: float) -> None:
        """`fraction`: how far into the current pass (0...1)."""
        elapsed = time.time() - self.pass_start
        per_pass = sum(self.pass_times) / len(self.pass_times) if self.pass_times else (elapsed / fraction if fraction > 0.02 else 0)
        left = per_pass * (1 - fraction) + per_pass * max(0, self.total - self.done - 1) if per_pass else 0
        eta = time.strftime("%H:%M", time.localtime(time.time() + left)) if left else "?"
        text = f"{self.current} | {stage} | {self.done}/{self.total} passes done | ETA {eta}"
        (self.work / "status.txt").write_text(text + "\n")


def target_paths(transformer) -> list[str]:
    return [p for p, m in transformer.named_modules() if isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0]


def hessian_key(path: str) -> str:
    """Linears reading the same input share one Hessian."""
    return re.sub(r"\.(to_k|to_v)$", ".to_q", re.sub(r"\.(add_k_proj|add_v_proj)$", ".add_q_proj", path))


def passes(transformer, paths: list[str], budget_gb: float) -> list[list[str]]:
    """Whole blocks (and the loose Linears) packed into passes whose float32
    Hessians together stay within `budget_gb`."""
    def unit(p: str) -> str:
        parts = p.split(".")
        return ".".join(parts[:2]) if parts[0] in ("transformer_blocks", "single_transformer_blocks") else "rest"

    units: dict[str, list[str]] = {}
    for p in paths:
        units.setdefault(unit(p), []).append(p)

    def cost(group: list[str]) -> float:
        keys = {hessian_key(p): get_module(transformer, p).weight.shape[-1] for p in group}
        return sum(n * n * 4 for n in keys.values()) / 1e9

    out: list[list[str]] = []
    for members in units.values():
        if out and cost(out[-1] + members) <= budget_gb:
            out[-1] += members
        else:
            out.append(list(members))
    return out


class Capture(nn.Module):
    """Calls the Linear, adding its input's X^T X to a shared sum."""

    def __init__(self, inner: nn.Linear, sums: dict, key: str):
        super().__init__()
        self.inner = inner
        self._sums = sums
        self._key = key

    def __call__(self, x):
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        entry = self._sums.get(self._key)
        xtx = flat.T @ flat
        total = xtx if entry is None else entry[0] + xtx
        # Now, not at the end of the step: lazily, every call's X^T X would
        # stay alive until then -- a second full set of Hessians (+5GB,
        # swapping, measured).
        mx.eval(total)
        self._sums[self._key] = (total, (0 if entry is None else entry[1]) + flat.shape[0])
        return self.inner(x)


def set_module(root, path: str, module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    if parts[-1].isdigit():
        parent[int(parts[-1])] = module
    else:
        setattr(parent, parts[-1], module)


def get_module(root, path: str):
    for part in path.split("."):
        root = root[int(part)] if part.isdigit() else getattr(root, part)
    return root


def square_factor(H: torch.Tensor) -> torch.Tensor:
    """X with X^T X = H (H symmetric PSD; float64 for the factorization)."""
    H = H.double()
    mean = torch.diagonal(H).mean().clamp_min(1e-12)
    eye = torch.eye(H.shape[0], dtype=H.dtype)
    # A jitter far under GPTQ's own 1% damping: a rank-deficient H (fewer
    # tokens than inputs, e.g. modulation Linears) still factors, without
    # the much slower eigendecomposition.
    for rel in (1e-7, 1e-5, 1e-4):
        L, info = torch.linalg.cholesky_ex(H + rel * mean * eye)
        if info.item() == 0:
            return L.T.float().contiguous()
    vals, vecs = torch.linalg.eigh(H)
    return (vals.clamp_min(0).sqrt()[:, None] * vecs.T).float().contiguous()


def load_model(template: str):
    model = Flux2KleinEdit(model_path=None, quantize=None)
    # The recipe's text encoder (8-bit, 27 of 36 layers used) -- the one the
    # deployed checkpoint conditions with.
    nn.quantize(model.text_encoder, bits=8, group_size=64,
                class_predicate=lambda _, m: hasattr(m, "to_quantized") and m.weight.shape[-1] % 64 == 0)
    model.text_encoder.layers = model.text_encoder.layers[:27]
    model.tiling_config = TilingConfig(vae_decode_tile_size=256)
    return model


def txt2img_twin(edit):
    """A Flux2Klein on the edit model's own modules (the edit class needs a
    reference image; loading a second copy doesn't fit)."""
    twin = Flux2Klein.__new__(Flux2Klein)
    nn.Module.__init__(twin)
    for name in ("prompt_cache", "model_config", "callbacks", "tiling_config", "tokenizers", "vae", "transformer",
                 "text_encoder", "bits", "lora_paths", "lora_scales"):
        setattr(twin, name, getattr(edit, name))
    return twin


def cache_prompts(model) -> None:
    """Encodes every calibration prompt once, then frees the text encoder."""
    original = pe.Flux2PromptEncoder.encode_prompt
    cache = {}
    for prompt in TXT_PROMPTS + [e[0] for e in EDITS] + [" "]:
        out = original(prompt=prompt, tokenizer=model.tokenizers["qwen3"], text_encoder=model.text_encoder,
                       num_images_per_prompt=1, max_sequence_length=512, text_encoder_out_layers=(9, 18, 27))
        mx.eval(out)
        cache[prompt] = out
    model.text_encoder = None
    gc.collect()
    mx.clear_cache()

    def cached(prompt, **_):
        return cache[prompt]

    pe.Flux2PromptEncoder.encode_prompt = staticmethod(cached)


def samples(work: Path):
    for i, prompt in enumerate(TXT_PROMPTS):
        w, h = SIZES[i % len(SIZES)]
        yield dict(seed=100 + i, prompt=prompt, width=w, height=h, ref=None, name=f"txt_{i:02d}")
    for j, (instruction, src) in enumerate(EDITS):
        yield dict(seed=200 + j, prompt=instruction, width=768, height=768, ref=str(work / f"txt_{src:02d}.png"), name=f"edit_{j:02d}")


class Steps:
    """Registered once; `on_step` is swapped per pass."""
    on_step = staticmethod(lambda: None)

    def call_in_loop(self, *_, **__):
        Steps.on_step()


def run_samples(model, twin, work: Path, save_images: bool, on_step, limit: int, progress: Progress) -> None:
    Steps.on_step = staticmethod(on_step)
    todo = list(samples(work))[: limit or None]
    for n, s in enumerate(todo):
        # Calibration is ~2/3 of a pass, GPTQ the rest (measured on the smoke run).
        progress.status(f"calibration {n + 1}/{len(todo)}", 0.66 * n / len(todo))
        t0 = time.time()
        kwargs = dict(seed=s["seed"], prompt=s["prompt"], num_inference_steps=4, width=s["width"], height=s["height"])
        img = model.generate_image(**kwargs, image_paths=[s["ref"]]) if s["ref"] else twin.generate_image(**kwargs)
        if save_images and not s["ref"]:
            img.image.save(work / f"{s['name']}.png")
        progress.log(f"  calibration {n + 1}/{len(todo)} {s['name']} {s['width']}x{s['height']} {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--hessian-budget-gb", type=float, default=3.0,
                    help="float32 Hessians per pass; more = fewer passes, but the bf16 model (7.8GB) must fit beside them")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--gptq-device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--max-passes", type=int, default=0, help="stop after this many new passes (0 = all)")
    ap.add_argument("--limit-samples", type=int, default=0, help="calibrate on the first N samples only (0 = all; a smoke test)")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    progress_file = work / "progress.json"
    done = set(json.loads(progress_file.read_text())["done_passes"]) if progress_file.exists() else set()

    mx.disable_compile()  # the Capture wrappers' side effects must run
    # mlx keeps freed buffers for reuse; unbounded, that pool alone grew the
    # process to 17GB beside the Hessians.
    mx.set_cache_limit(1 << 30)
    model = load_model(args.template)
    cache_prompts(model)
    model.callbacks.register(Steps())
    twin = txt2img_twin(model)
    transformer = model.transformer
    paths = target_paths(transformer)
    plan = passes(transformer, paths, args.hessian_budget_gb)
    progress = Progress(work, len(plan), len(done & set(range(len(plan)))))
    progress.log(f"{len(paths)} target Linears, {len(plan)} passes, {len(done)} already done")

    # The edit references are this model's own images: made first.
    if not all((work / f"txt_{src:02d}.png").exists() for _, src in EDITS):
        progress.log("Generating the edit reference images ...")
        for i, s in enumerate(samples(work)):
            if s["ref"]:
                break
            progress.status(f"reference image {i + 1}/{len(TXT_PROMPTS)}", 0)
            img = twin.generate_image(seed=s["seed"], prompt=s["prompt"], num_inference_steps=4,
                                       width=s["width"], height=s["height"])
            img.image.save(work / f"{s['name']}.png")

    new_passes = 0
    for n, group in enumerate(plan):
        if n in done:
            continue
        if args.max_passes and new_passes >= args.max_passes:
            break
        progress.start_pass(n)
        progress.log(f"--- pass {n + 1}/{len(plan)}: {len(group)} Linears ---")
        sums: dict = {}
        originals = {p: get_module(transformer, p) for p in group}
        for p, m in originals.items():
            set_module(transformer, p, Capture(m, sums, hessian_key(p)))
        run_samples(model, twin, work, save_images=False, on_step=lambda: mx.eval([v[0] for v in sums.values()]),
                    limit=args.limit_samples, progress=progress)
        for p, m in originals.items():
            set_module(transformer, p, m)

        progress.log("  GPTQ ...")
        corrected = {}
        factors: dict[str, torch.Tensor] = {}
        for j, p in enumerate(group):
            progress.status(f"GPTQ {j + 1}/{len(group)} {p}", 0.66 + 0.34 * j / len(group))
            key = hessian_key(p)
            if key not in factors:
                # Copies (np.array copies by default): a view on an mlx
                # buffer is read after mlx frees it -- that crashed a run.
                H, rows = sums.pop(key)
                H = torch.from_numpy(np.array(H / rows))
                factors[key] = square_factor(H)
                del H
            W = torch.from_numpy(np.array(originals[p].weight.astype(mx.float32)))
            t0 = time.time()
            result = gptq_nbit(W, factors[key].to(args.gptq_device), bits=args.bits, group_size=args.group_size,
                               percdamp=args.percdamp, device=args.gptq_device, scheme="affine")
            corrected[p] = mx.array(result["W_hat"].float().cpu().numpy()).astype(mx.bfloat16)
            progress.log(f"  GPTQ {j + 1}/{len(group)} {p} {tuple(W.shape)} {time.time() - t0:.0f}s")
            if not any(hessian_key(q) == key for q in group[group.index(p) + 1 :]):
                del factors[key]
        mx.save_safetensors(str(work / f"pass_{n:02d}.safetensors"), corrected)
        done.add(n)
        progress_file.write_text(json.dumps({"done_passes": sorted(done), "passes": len(plan), "bits": args.bits,
                                             "group_size": args.group_size}))
        del sums, corrected, factors
        gc.collect()
        mx.clear_cache()
        new_passes += 1
        progress.end_pass()
        progress.log(f"pass {n + 1} done in {progress.pass_times[-1] / 60:.1f} min")

    if len(done) < len(plan):
        progress.log(f"{len(done)}/{len(plan)} passes done; re-run to continue.")
        return

    progress.log("Assembling the checkpoint ...")
    (work / "status.txt").write_text("assembling the checkpoint\n")
    for f in sorted(glob.glob(str(work / "pass_*.safetensors"))):
        for p, w in mx.load(f).items():
            get_module(transformer, p).weight = w
    # Re-quantized with the grid GPTQ corrected against: the same codes.
    nn.quantize(transformer, class_predicate=lambda p, m: p in set(paths) and {"bits": args.bits, "group_size": args.group_size})
    tensors = dict(tree_flatten(transformer.parameters()))
    mx.eval(*tensors.values())
    out = Path(args.output_dir)
    if out.exists():
        shutil.rmtree(out)
    subprocess.run(["cp", "-cR", args.template, str(out)], check=True)
    tdir = out / "transformer"
    for f in glob.glob(str(tdir / "*.safetensors")):
        os.remove(f)
    mx.save_safetensors(str(tdir / "0.safetensors"), tensors, metadata={"mflux_version": "0.20.0", "quantization_level": "8"})
    json.dump({"metadata": {}, "weight_map": {k: "0.safetensors" for k in tensors}},
              open(tdir / "model.safetensors.index.json", "w"), indent=2)
    progress.log(f"transformer {sum(v.nbytes for v in tensors.values()) / 1e9:.2f} GB -> {out}")
    (work / "status.txt").write_text(f"done -> {out}\n")
    progress.log("KLEIN_GPTQ_DONE")


if __name__ == "__main__":
    main()
