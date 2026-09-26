"""MiniMax Music 3 (MiniMaxAI/MiniMax-Music3) reference runs through
diffusers' modular pipeline, on the same vocal test set as the ACE-Step
comparison (docs/vocal_eval: 3 songs x 4 seeds, 30 s), for the Whisper WER
metric (acestep_lyrics_wer.py, run on the Mac afterwards).

    python minimax_music3_ref.py --songs <vocal_eval dir> --out <dir> [--device cuda] [--duration 30]

Records per track: seconds, peak memory, and the actual duration (the LM
may end a song before the requested length).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from diffusers import ModularPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", required=True, help="dir with songs.tsv and <song>.txt lyrics")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="MiniMaxAI/MiniMax-Music3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--name", default="minimax")
    args = ap.parse_args()

    songs_dir, out = Path(args.songs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = ModularPipeline.from_pretrained(args.model)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.to(args.device)
    load_s = time.time() - t0
    print(f"loaded in {load_s:.0f}s", flush=True)

    for line in (songs_dir / "songs.tsv").read_text().splitlines():
        if not line.strip():
            continue
        song, caption = line.split("\t", 1)
        # Lowercase section tags, each on its own line (the model's input contract).
        lyrics = "\n".join(l.strip().lower() if l.strip().startswith("[") else l.strip()
                           for l in (songs_dir / f"{song}.txt").read_text().splitlines() if l.strip())
        for seed in args.seeds:
            if (out / f"{song}_s{seed}.wav").exists():
                continue   # resumable: a killed run picks up where it was
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            audio = pipe(prompt=caption, lyrics=lyrics, audio_duration=args.duration,
                         generator=torch.Generator(args.device).manual_seed(seed), output="audios")[0]
            secs = time.time() - t
            wav = out / f"{song}_s{seed}.wav"
            sf.write(wav, audio.T if audio.ndim == 2 else audio, pipe.sampling_rate)
            record = {"name": args.name, "song": song, "seed": seed, "seconds": round(secs, 1),
                      "audio_seconds": round(audio.shape[-1] / pipe.sampling_rate, 1),
                      "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if args.device == "cuda" else None,
                      "load_s": round(load_s), "caption": caption, "wav": str(wav)}
            with open(out / "minimax.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
