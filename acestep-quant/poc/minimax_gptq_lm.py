"""GPTQ for MiniMax Music 3's 8B global LM (Qwen3), calibrated on the model's
own autoregressive generation of songs that are NOT in the evaluation set.
Forward pre-hooks sum X^T X of every Linear's input over every generated
frame; Linears reading the same input share one Hessian (q/k/v, gate/up).
gptq_nbit (nemotron-extreme-quant, MLX's exact affine grid) then corrects
each weight; the corrected, on-grid bf16 weights replace the originals in
place, and the evaluation songs are generated from them. The corrected
weights are saved for the MLX checkpoint.

    python minimax_gptq_lm.py --calib <dir> --eval <dir> --out <dir> [--bits 4] [--group 64]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from diffusers import ModularPipeline
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gptq import gptq_nbit  # noqa: E402


def shared_key(name: str) -> str:
    return re.sub(r"\.(q_proj|k_proj|v_proj)$", ".qkv_in", re.sub(r"\.(gate_proj|up_proj)$", ".mlp_in", name))


def songs(dir_: Path):
    for line in (dir_ / "songs.tsv").read_text().splitlines():
        if line.strip():
            song, caption = line.split("\t", 1)
            lyrics = "\n".join(l.strip().lower() if l.strip().startswith("[") else l.strip()
                               for l in (dir_ / f"{song}.txt").read_text().splitlines() if l.strip())
            yield song, caption, lyrics


def square_factor(H: torch.Tensor) -> torch.Tensor:
    H = H.double()
    mean = torch.diagonal(H).mean().clamp_min(1e-12)
    eye = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    for rel in (1e-7, 1e-5, 1e-4, 1e-3):
        L, info = torch.linalg.cholesky_ex(H + rel * mean * eye)
        if info.item() == 0:
            return L.T.float().contiguous()
    vals, vecs = torch.linalg.eigh(H)
    return (vals.clamp_min(0).sqrt()[:, None] * vecs.T).float().contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/workspace/mm3")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--calib-duration", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = ModularPipeline.from_pretrained(args.model)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.to("cuda")
    lm = pipe.language_model
    targets = {n: m for n, m in lm.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n
               and m.weight.shape[1] % args.group == 0}
    print(f"{len(targets)} Linears in the LM", flush=True)

    sums: dict[str, torch.Tensor] = {}
    rows: dict[str, int] = {}

    def hook(key):
        def f(module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            h = x.T @ x
            if key in sums:
                sums[key] += h
            else:
                sums[key] = h
            rows[key] = rows.get(key, 0) + x.shape[0]
        return f

    handles, hooked = [], set()
    for name, m in targets.items():
        key = shared_key(name)
        if key in hooked:
            continue   # the first of a shared group sees the same input
        hooked.add(key)
        handles.append(m.register_forward_pre_hook(hook(key)))

    t0 = time.time()
    with torch.no_grad():
        for song, caption, lyrics in songs(Path(args.calib)):
            pipe(prompt=caption, lyrics=lyrics, audio_duration=args.calib_duration,
                 generator=torch.Generator("cuda").manual_seed(0), output="audios")
            print(f"calibrated on {song} ({time.time() - t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()
    print(f"{len(sums)} Hessians, rows {min(rows.values())}..{max(rows.values())}", flush=True)

    corrected, t0 = {}, time.time()
    factors: dict[str, torch.Tensor] = {}
    names = sorted(targets)
    for i, name in enumerate(names):
        key = shared_key(name)
        if key not in factors:
            factors[key] = square_factor(sums.pop(key) / rows[key])
        m = targets[name]
        W = m.weight.detach().float()
        res = gptq_nbit(W, factors[key], bits=args.bits, group_size=args.group, device="cuda", scheme="affine")
        w_hat = res["W_hat"].to(torch.bfloat16)
        with torch.no_grad():
            m.weight.copy_(w_hat)
        corrected[name] = w_hat.cpu().contiguous()
        if not any(shared_key(n) == key for n in names[i + 1:]):
            del factors[key]
        if i % 25 == 0:
            print(f"GPTQ {i + 1}/{len(names)} {name} ({time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    save_file(corrected, str(out / "lm_gptq_corrected.safetensors"),
              metadata={"bits": str(args.bits), "group_size": str(args.group)})
    print(f"GPTQ done ({time.time() - t0:.0f}s)", flush=True)

    for song, caption, lyrics in songs(Path(args.eval)):
        for seed in args.seeds:
            wav = out / f"{song}_s{seed}.wav"
            if wav.exists():
                continue
            t = time.time()
            audio = pipe(prompt=caption, lyrics=lyrics, audio_duration=30.0,
                         generator=torch.Generator("cuda").manual_seed(seed), output="audios")[0]
            sf.write(wav, audio.T if audio.ndim == 2 else audio, pipe.sampling_rate)
            print(json.dumps({"song": song, "seed": seed, "seconds": round(time.time() - t, 1)}), flush=True)
    print("GPTQ_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
