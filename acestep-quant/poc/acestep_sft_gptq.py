"""GPTQ (Hessian-corrected) 4-bit for ACE-Step 1.5 sft's DiT and condition
encoders, calibrated on the real sft sampling loop (50 steps, CFG: both the
conditional and the unconditional decoder calls) of songs outside the
evaluation set. Same modules as mlx-community's 4-bit turbo conversion
(decoder + encoder Linears; embeddings round-to-nearest, as GPTQ has no
input Hessian for a lookup). Linears reading the same input share one
Hessian (q/k/v, cross-attention k/v, gate/up).

The output is an mlx-audio ACE-Step checkpoint (config "quantization":
bits/group_size; the loader quantizes every module that has `.scales`),
loadable exactly like mlx-community/ACE-Step1.5-MLX-4bit.

    python acestep_sft_gptq.py --model <bf16 sft MLX dir> --calib <songs dir> --out <dir> [--bits 4]
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nemotron-extreme-quant" / "poc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gptq import gptq_nbit  # noqa: E402


def load_sft(path: str):
    import mlx_audio.utils as audio_utils
    from mlx_audio.tts import load
    from acestep_null_cond import install

    pick = audio_utils.get_model_class
    audio_utils.get_model_class = lambda model_type, model_name, category, model_remapping: pick(
        "ace_step", None, category, model_remapping)
    model = load(path)
    install(model)
    return model


def get_module(root, path):
    for part in path.split("."):
        root = root[int(part)] if part.isdigit() else getattr(root, part)
    return root


def set_module(root, path, module):
    parts = path.split(".")
    parent = get_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    if parts[-1].isdigit():
        parent[int(parts[-1])] = module
    else:
        setattr(parent, parts[-1], module)


def targets(model) -> list[str]:
    out = []
    for comp in ("decoder", "encoder"):
        for p, m in getattr(model, comp).named_modules():
            if isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0:
                out.append(f"{comp}.{p}")
    return out


def hessian_key(path: str) -> str:
    path = re.sub(r"\.self_attn\.(q_proj|k_proj|v_proj)$", ".self_attn.qkv", path)
    path = re.sub(r"\.cross_attn\.(k_proj|v_proj)$", ".cross_attn.kv", path)
    return re.sub(r"\.mlp\.(gate_proj|up_proj)$", ".mlp.gate_up", path)


class Capture(nn.Module):
    def __init__(self, inner, sums, key):
        super().__init__()
        self.inner, self._sums, self._key = inner, sums, key

    def __call__(self, x):
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        prev = self._sums.get(self._key)
        total = flat.T @ flat if prev is None else prev[0] + flat.T @ flat
        mx.eval(total)
        self._sums[self._key] = (total, (0 if prev is None else prev[1]) + flat.shape[0])
        return self.inner(x)


def square_factor(H: torch.Tensor) -> torch.Tensor:
    H = H.double()
    mean = torch.diagonal(H).mean().clamp_min(1e-12)
    eye = torch.eye(H.shape[0], dtype=H.dtype)
    for rel in (1e-7, 1e-5, 1e-4, 1e-3):
        L, info = torch.linalg.cholesky_ex(H + rel * mean * eye)
        if info.item() == 0:
            return L.T.float().contiguous()
    vals, vecs = torch.linalg.eigh(H)
    return (vals.clamp_min(0).sqrt()[:, None] * vecs.T).float().contiguous()


def songs(dir_: Path):
    for line in (dir_ / "songs.tsv").read_text().splitlines():
        if line.strip():
            song, caption = line.split("\t", 1)
            yield song, caption, (dir_ / f"{song}.txt").read_text()


SFT = dict(num_steps=50, guidance_scale=7.0, shift=1.0, guidance_interval=1.0, cfg_type="apg", use_lm=False, verbose=False)


def save_quantized(model, src: Path, out: Path, bits: int, group: int, paths: set[str]) -> None:
    """nn.quantize the target Linears (and the encoders' embeddings) and write
    an mlx-audio checkpoint next to a copy of the source's other files."""
    def pred(p, m):
        if not hasattr(m, "to_quantized") or m.weight.shape[-1] % group:
            return False
        return p in paths or (isinstance(m, nn.Embedding) and p.startswith("encoder."))
    dit_keys = set(mx.load(str(src / "model.safetensors")).keys())
    nn.quantize(model, group_size=group, bits=bits, class_predicate=pred)
    # The null-condition wrapper (acestep_null_cond) puts the encoder under
    # ".inner": saved under its own names again, or the checkpoint would lack
    # the encoder (mlx-audio loads it non-strict and leaves it random).
    flat = {k.replace("encoder.inner.", "encoder."): v for k, v in tree_flatten(model.parameters())}
    weights = {k: v for k, v in flat.items() if k in dit_keys or k.rsplit(".", 1)[0] + ".weight" in dit_keys}
    missing = [k for k in dit_keys if k not in weights and k.rsplit(".", 1)[0] + ".weight" not in {w.rsplit(".", 1)[0] + ".weight" for w in weights}]
    assert not missing, f"missing from the checkpoint: {missing[:5]} ({len(missing)})"
    if out.exists():
        shutil.rmtree(out)
    # Everything but the DiT's own weights (the text encoder's model.safetensors too).
    shutil.copytree(src, out, ignore=lambda d, names: ["model.safetensors"] if Path(d) == src else [])
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    config = json.loads((src / "config.json").read_text())
    config["quantization"] = {"bits": bits, "group_size": group, "quantized_components": ["decoder", "encoder"]}
    (out / "config.json").write_text(json.dumps(config, indent=4))
    print(f"saved {out}: {sum(v.nbytes for v in weights.values()) / 1e9:.2f} GB DiT", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--rtn-only", action="store_true", help="skip calibration: plain round-to-nearest")
    ap.add_argument("--hessians", help="file: calibration's Hessians saved there, and reused from there if present "
                    "(a crash in the GPTQ phase then doesn't cost the calibration)")
    args = ap.parse_args()

    mx.set_cache_limit(1 << 30)
    model = load_sft(args.model)
    paths = targets(model)
    print(f"{len(paths)} target Linears", flush=True)

    if not args.rtn_only:
        sums: dict = {}
        originals = {p: get_module(model, p) for p in paths}
        saved = Path(args.hessians) if args.hessians else None
        if saved and saved.exists():
            loaded = mx.load(str(saved))
            meta = json.loads(Path(str(saved) + ".rows.json").read_text())
            sums = {k: (loaded[k], meta[k]) for k in meta}
            print(f"{len(sums)} Hessians loaded from {saved}", flush=True)
        else:
            for p, m in originals.items():
                set_module(model, p, Capture(m, sums, hessian_key(p)))
            t0 = time.time()
            for song, caption, lyrics in songs(Path(args.calib)):
                list(model.generate(text=caption, lyrics=lyrics, duration=args.duration, seed=0, **SFT))
                print(f"calibrated on {song} ({time.time() - t0:.0f}s)", flush=True)
            for p, m in originals.items():
                set_module(model, p, m)
            print(f"{len(sums)} Hessians", flush=True)
            if saved:
                mx.save_safetensors(str(saved), {k: v[0] for k, v in sums.items()})
                Path(str(saved) + ".rows.json").write_text(json.dumps({k: v[1] for k, v in sums.items()}))
                print(f"saved to {saved}", flush=True)

        t0, factors = time.time(), {}
        for i, p in enumerate(paths):
            key = hessian_key(p)
            if key not in factors:
                H, rows = sums.pop(key)
                factors[key] = square_factor(torch.from_numpy(np.array(H / rows)))
            layer = originals[p]
            W = torch.from_numpy(np.array(layer.weight.astype(mx.float32)))
            res = gptq_nbit(W, factors[key], bits=args.bits, group_size=args.group, device="cpu", scheme="affine")
            layer.weight = mx.array(res["W_hat"].float().numpy()).astype(layer.weight.dtype)
            if not any(hessian_key(q) == key for q in paths[i + 1:]):
                del factors[key]
            if i % 40 == 0:
                print(f"GPTQ {i + 1}/{len(paths)} {p} ({time.time() - t0:.0f}s)", flush=True)
            # (No gc.collect() here: under Python 3.14 it segfaulted in
            # visit_decref walking mlx objects, mid-GPTQ.)
        print(f"GPTQ done ({time.time() - t0:.0f}s)", flush=True)
    save_quantized(model, Path(args.model), Path(args.out), args.bits, args.group, set(paths))


if __name__ == "__main__":
    main()
