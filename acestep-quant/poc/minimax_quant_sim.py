"""How MiniMax Music 3 holds up quantized, before any port: every Linear of
the chosen components is replaced by its MLX-affine quantize->dequantize
round trip (per-group min/max, `bits`, group 64 along the input dim --
the grid mx.quantize uses), so the audio is what a real MLX checkpoint at
those bits would give, while running in torch on the GPU.

    python minimax_quant_sim.py --songs <dir> --out <dir> --name tag --lm-bits 4 --depth-bits 8 --dit-bits 8

0 bits = left in bf16. The LM's lm_head and embeddings stay bf16 (their
rows are the music-token vocabulary; mlx-lm would quantize them too --
checked separately if the rest holds).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from diffusers import ModularPipeline


@torch.no_grad()
def fake_quant_(linear: torch.nn.Linear, bits: int, group: int) -> None:
    w = linear.weight
    out_f, in_f = w.shape
    if in_f % group:
        return
    g = w.float().reshape(out_f, in_f // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = (hi - lo).clamp_min(1e-8) / (2**bits - 1)
    q = torch.clamp(torch.round((g - lo) / scale), 0, 2**bits - 1)
    linear.weight.copy_((q * scale + lo).reshape(out_f, in_f).to(w.dtype))


def quantize_module(module: torch.nn.Module, bits: int, group: int, skip=("lm_head",)) -> tuple[int, int]:
    n = params = 0
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Linear) and not any(s in name for s in skip):
            fake_quant_(m, bits, group)
            n += 1
            params += m.weight.numel()
    return n, params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="/workspace/mm3")
    ap.add_argument("--lm-bits", type=int, default=0)
    ap.add_argument("--depth-bits", type=int, default=0)
    ap.add_argument("--dit-bits", type=int, default=0)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = ModularPipeline.from_pretrained(args.model)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.to("cuda")
    report = {}
    for label, comp, bits in (("lm", "language_model", args.lm_bits), ("depth", "rvq_depth_decoder", args.depth_bits),
                              ("dit", "transformer", args.dit_bits)):
        if bits:
            n, p = quantize_module(getattr(pipe, comp), bits, args.group)
            report[label] = {"bits": bits, "linears": n, "params": p}
            print(f"{label}: {n} Linears ({p / 1e9:.2f}B params) -> {bits}-bit g{args.group}", flush=True)
    (out / "quant.json").write_text(json.dumps(report, indent=2))

    songs_dir = Path(args.songs)
    for line in (songs_dir / "songs.tsv").read_text().splitlines():
        if not line.strip():
            continue
        song, caption = line.split("\t", 1)
        lyrics = "\n".join(l.strip().lower() if l.strip().startswith("[") else l.strip()
                           for l in (songs_dir / f"{song}.txt").read_text().splitlines() if l.strip())
        for seed in args.seeds:
            wav = out / f"{song}_s{seed}.wav"
            if wav.exists():
                continue
            t = time.time()
            audio = pipe(prompt=caption, lyrics=lyrics, audio_duration=args.duration,
                         generator=torch.Generator("cuda").manual_seed(seed), output="audios")[0]
            sf.write(wav, audio.T if audio.ndim == 2 else audio, pipe.sampling_rate)
            rec = {"name": args.name, "song": song, "seed": seed, "seconds": round(time.time() - t, 1), **report}
            with open(out / "runs.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
