"""Times and measures one ACE-Step 1.5 generation through mlx-audio
(Blaizzy/mlx-audio, branch pc/add-ace): per-stage time, peak memory, the
wav and a spectrogram to look at.

    python acestep_bench.py --model mlx-community/ACE-Step1.5-MLX-4bit --out <dir> --name 4bit \
        [--lyrics-file f] [--duration 30] [--steps 8] [--lm 0.6B] [--seed 7]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import scipy.io.wavfile as wavfile


LM_TIME = {"s": 0.0, "codes": 0}
PEAKS: dict[str, float] = {}


def peak_of(name: str, fn):
    """Wraps fn: mlx's peak memory during it, in GB, into PEAKS[name]."""
    def wrapped(*a, **k):
        mx.eval()
        mx.reset_peak_memory()
        out = fn(*a, **k)
        mx.eval(out) if isinstance(out, mx.array) else None
        PEAKS[name] = max(PEAKS.get(name, 0), round(mx.get_peak_memory() / 1e9, 2))
        return out
    return wrapped


def patch_lm(cap_codes: bool) -> None:
    """Times the 5 Hz LM planner, and with `cap_codes` stops it once it has
    the duration's worth of codes (5 per second): the 0.6B LM ignores the
    duration in its prompt and writes up to max_new_tokens (3000), ~7x more
    than a 30 s track uses -- the surplus is cut off afterwards anyway."""
    from mlx_audio.tts.models.ace_step import lm as lm_module

    cls = next(v for v in vars(lm_module).values() if isinstance(v, type) and hasattr(v, "generate_audio_codes"))
    original = cls.generate_audio_codes

    def generate_audio_codes(self, caption, lyrics="", duration=30, language="en", seed=None):
        t0 = time.time()
        mx.reset_peak_memory()
        if not cap_codes:
            codes, meta = original(self, caption, lyrics=lyrics, duration=duration, language=language, seed=seed)
        else:
            import mlx_lm
            from mlx_lm.sample_utils import make_sampler

            if not self._loaded:
                self.load()
            prompt = self._apply_chat_template(self._format_prompt(caption, lyrics, duration, language), enable_thinking=True)
            if seed is not None:
                mx.random.seed(seed)
            sampler = make_sampler(temp=self.config.temperature, top_p=self.config.top_p, top_k=self.config.top_k)
            text, need = "", int(duration * 5)
            for response in mlx_lm.stream_generate(self.model, self.tokenizer, prompt=prompt,
                                                   max_tokens=self.config.max_new_tokens, sampler=sampler):
                text += response.text
                if text.count("<|audio_code_") >= need:
                    break
            meta, codes = self.parse_output(text)
        LM_TIME["s"] += time.time() - t0
        PEAKS["lm"] = round(mx.get_peak_memory() / 1e9, 2)
        LM_TIME["codes"] = codes.count("<|audio_code_")
        return codes, meta

    cls.generate_audio_codes = generate_audio_codes


def spectrogram(audio: np.ndarray, rate: int, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.specgram(mono, NFFT=2048, Fs=rate, noverlap=1024, cmap="magma", vmin=-120)
    ax.set_ylim(0, 16000)
    ax.set_title(title)
    ax.set_xlabel("s")
    ax.set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", default="upbeat synthwave with punchy drums, warm analog bass and a catchy lead melody")
    ap.add_argument("--lyrics-file")
    ap.add_argument("--language", default="en")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lm", default="0.6B")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cap-codes", action="store_true", help="stop the LM at duration*5 codes")
    ap.add_argument("--planner", default="mlx-audio", choices=["mlx-audio", "official"],
                    help="official: acestep_planner.py (the official prompt layout, CFG, forced duration)")
    ap.add_argument("--lm-path", help="LM checkpoint for --planner official (HF id or local dir)")
    ap.add_argument("--guidance", type=float, default=1.0, help="DiT CFG scale (non-turbo: official 7.0)")
    ap.add_argument("--shift", type=float, default=3.0, help="timestep shift (official sft/base: 1.0)")
    ap.add_argument("--guidance-interval", type=float, default=0.5, help="fraction of steps with CFG (official: 1.0)")
    ap.add_argument("--cfg-type", default="apg", choices=["apg", "cfg"])
    ap.add_argument("--no-lm", action="store_true", help="no 5 Hz LM planner (sft/base: CFG only; mlx-audio's LM-hint path breaks the sft DiT)")
    ap.add_argument("--null-cond", action="store_true", help="CFG's unconditional branch with the trained null_condition_emb (official)")
    ap.add_argument("--vae-chunk", type=int, default=0, help="decode in windows of N latent frames (0 = whole)")
    ap.add_argument("--no-dit-metadata", action="store_true", help="--planner official: keep mlx-audio's DiT prompt")
    args = ap.parse_args()

    from mlx_audio.tts import load
    import mlx_audio.utils as audio_utils

    # mlx-audio maps the checkpoint's model_type ("acestep") to its module
    # through the parts of the path: a local folder doesn't give it away.
    pick_module = audio_utils.get_model_class
    audio_utils.get_model_class = lambda model_type, model_name, category, model_remapping: pick_module(
        "ace_step", None, category, model_remapping)

    if args.planner == "official":
        from acestep_planner import install

        install(args.lm_path or f"ACE-Step/acestep-5Hz-lm-{args.lm}", dit_metadata=not args.no_dit_metadata)
    patch_lm(args.cap_codes and args.planner != "official")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lyrics = Path(args.lyrics_file).read_text() if args.lyrics_file else ""

    t0 = time.time()
    model = load(args.model)
    mx.eval(model.parameters())
    load_s = time.time() - t0
    mem_after_load = mx.get_active_memory() / 1e9
    if args.null_cond:
        from acestep_null_cond import install as install_null_cond

        install_null_cond(model)
    if args.vae_chunk:
        from acestep_vae_chunk import chunked

        model.vae.decode = chunked(model.vae.decode, chunk=args.vae_chunk, overlap=16)
    model.vae.decode = peak_of("vae_decode", model.vae.decode)
    mx.reset_peak_memory()

    log = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(log):
        results = list(model.generate(text=args.prompt, lyrics=lyrics, duration=args.duration, seed=args.seed,
                                      num_steps=args.steps, vocal_language=args.language, lm_model_size=args.lm,
                                      guidance_scale=args.guidance, shift=args.shift, use_lm=not args.no_lm,
                                      guidance_interval=args.guidance_interval, cfg_type=args.cfg_type,
                                      verbose=True))
    total_s = time.time() - t0
    peak = mx.get_peak_memory() / 1e9
    text = log.getvalue()
    sys.stdout.write(text)

    audio = np.array(results[-1].audio.astype(mx.float32))
    rate = results[-1].sample_rate
    wavfile.write(out / f"{args.name}.wav", rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    spectrogram(audio, rate, out / f"{args.name}.png", f"{args.name}: {args.prompt[:70]}")

    stages = {k.lower(): float(v) for k, v in re.findall(r"(Diffusion|Decode) completed in ([\d.]+)s", text)}
    record = {
        "name": args.name, "model": args.model, "lm": args.lm, "steps": args.steps, "duration_s": args.duration,
        "seed": args.seed, "vocals": bool(lyrics), "load_s": round(load_s, 1), "generate_s": round(total_s, 1),
        "rtf": round(total_s / args.duration, 2), "active_after_load_gb": round(mem_after_load, 2),
        "peak_generate_gb": round(peak, 2), "rms": float(np.sqrt(np.mean(audio**2))),
        "samples": int(audio.shape[0]), "lm_s": round(LM_TIME["s"], 1), "lm_codes": LM_TIME["codes"],
        "cap_codes": args.cap_codes, "planner": args.planner, "vae_chunk": args.vae_chunk, "guidance": args.guidance, "use_lm": not args.no_lm, "null_cond": args.null_cond, "shift": args.shift, "dit_metadata": not args.no_dit_metadata, "lm_path": args.lm_path, "prompt": args.prompt, "stages": stages, "stage_peaks_gb": PEAKS,
    }
    with open(out / "bench.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
