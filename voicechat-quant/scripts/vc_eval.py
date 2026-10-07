#!/usr/bin/env python3
"""Duplex speed + quality run of NemotronLabs VoiceChat 11B on the eval set.

For every question WAV: open a fresh duplex session, stream the speech and
then silence frame by frame (80 ms), stop once the reply has finished, and
record per-frame wall time split by component (perception, RNNT user ASR,
LLM, TTS, codec, rest). Writes <out>/<id>.wav (reply audio, 22.05 kHz),
<out>/<id>.json and <out>/run.json. Quality is scored separately by
score.py (independent Whisper ASR), so this process stays small.

  HF_HUB_OFFLINE=1 python vc_eval.py --out results/baseline
  HF_HUB_OFFLINE=1 python vc_eval.py --out results/bf16act --act-dtype bf16 --scales-dtype bf16

Variant flags: see vc_variants.py. `--model` may point at any snapshot the
fork can load (e.g. a GPTQ result synced back from the pod).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import platform
import subprocess
import sys
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vc_variants  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "~/.cache/huggingface/hub/models--mlx-community--NemotronLabs-VoiceChat-11B-4bit/snapshots/*"
PROMPT = "Be concise and answer in one sentence."


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


class Timers:
    """Wrap session components; each wrapper forces evaluation of its output
    so time is attributed to the component that produced it. The stock code
    already synchronizes at each of these boundaries (int(argmax), eval), so
    this adds no extra GPU round trips."""

    def __init__(self):
        self.t = collections.defaultdict(float)

    def wrap(self, obj, name, key, sync):
        fn = getattr(obj, name)

        def inner(*a, **k):
            t0 = time.perf_counter()
            r = fn(*a, **k)
            sync(r)
            self.t[key] += time.perf_counter() - t0
            return r

        setattr(obj, name, inner)

    def instrument_model(self, model):
        self.wrap(model.tts_model.tts_model, "generate_step", "tts", lambda r: mx.eval(r[0]))
        self.wrap(model.tts_model.audio_codec, "decode_step", "codec", lambda r: mx.eval(r))

    def instrument_session(self, session):
        self.wrap(session, "_perception_step", "perception", lambda r: mx.eval(*r))
        self.wrap(session._rnnt, "step", "rnnt", lambda r: None)
        self.wrap(
            session, "_language_step", "llm", lambda o: mx.eval(o.text_logits, o.function_logits)
        )

    def take(self):
        out = dict(self.t)
        self.t.clear()
        return out


def run_question(model, q, audio_dir, out_dir, args, timers):
    speech = read_wav(audio_dir / f"{q['id']}.wav")
    extra = {"tts_guidance": False} if args.no_tts_guidance else {}
    if args.tts_idle_frames:
        extra.update(tts_idle_frames=args.tts_idle_frames, tts_idle_rms=args.tts_idle_rms)
    session = model.create_duplex_session(system_prompt=args.prompt, seed=args.seed, **extra)
    if args.profile:
        timers.instrument_session(session)
    timers.take()  # drop time spent in the session's prompt prefill
    fs = session.frame_samples
    speech = np.concatenate([speech, np.zeros((-len(speech)) % fs, np.float32)])
    speech_frames = len(speech) // fs
    max_frames = speech_frames + int(args.max_tail_s / 0.08)
    silence = np.zeros(fs, np.float32)

    text, user, audio, frames = [], [], [], []
    first_text = first_voice = last_text = None
    loud = collections.deque(maxlen=args.quiet_frames)
    for i in range(max_frames):
        chunk = speech[i * fs : (i + 1) * fs] if i < speech_frames else silence
        t0 = time.perf_counter()
        events = session.push_audio(chunk, sample_rate=16000)
        wall = time.perf_counter() - t0
        comp = timers.take() if args.profile else {}
        comp["total"] = wall
        frames.append(comp)
        for ev in events:
            if ev.kind == "audio":
                s = np.array(ev.samples, dtype=np.float32)
                audio.append(s)
                rms = float(np.sqrt(np.mean(s**2)))
                loud.append(rms > args.voice_rms)
                if first_voice is None and rms > args.voice_rms and i >= speech_frames:
                    first_voice = i
            elif ev.kind == "assistant_text_delta":
                text.append(ev.delta)
                last_text = i
                if first_text is None:
                    first_text = i
            elif ev.kind == "user_transcript_delta":
                user.append(ev.delta)
        # Stop once the reply is over: text started, no text for idle_frames,
        # and the last quiet_frames of audio were silent.
        if (
            i >= speech_frames
            and last_text is not None
            and i - last_text >= args.idle_frames
            and len(loud) == loud.maxlen
            and not any(loud)
        ):
            break
    wav = np.concatenate(audio) if audio else np.zeros(1, np.float32)
    write_wav(out_dir / f"{q['id']}.wav", wav, session.output_sample_rate)
    rec = {
        "id": q["id"],
        "question": q["text"],
        "user_transcript": "".join(user).strip(),
        "assistant_text": "".join(text).strip(),
        "speech_frames": speech_frames,
        "frames": len(frames),
        "first_text_frame": first_text,
        "first_voiced_frame_after_speech": first_voice,
        "reply_latency_s": None if first_voice is None else round((first_voice - speech_frames) * 0.08, 2),
        "audio_s": round(len(wav) / session.output_sample_rate, 2),
        "tts_idle_skipped": getattr(getattr(session, "_stream", session), "tts_idle_skipped", 0),
        # frame 0 includes one-off graph building; keep it but report stats without it
        "frame_ms": [{k: round(v * 1000, 2) for k, v in f.items()} for f in frames],
    }
    (out_dir / f"{q['id']}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


def load_system_state():
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as e:  # pragma: no cover
            return str(e)

    gpu = []
    for _ in range(3):
        out = sh(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
        for tok in out.split(","):
            if '"Device Utilization %"' in tok:
                gpu.append(int(tok.split("=")[1].strip("} ")))
        time.sleep(0.5)
    return {
        "gpu_util_pct_idle_samples": gpu,  # other apps' GPU use; ours is not running here
        "uptime": sh(["uptime"]),
        "memory_pressure": sh(["memory_pressure"]).splitlines()[-1:] or None,
        "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "mlx": mx.__version__,
        "python": platform.python_version(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", default=str(HERE / "eval/questions.json"))
    ap.add_argument("--audio-dir", default=str(HERE / "eval/audio"))
    ap.add_argument("--ids", default=None, help="comma list, default all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tail-s", type=float, default=10.0)
    ap.add_argument("--idle-frames", type=int, default=20)
    ap.add_argument("--quiet-frames", type=int, default=10, help="reply is over after this many silent 80 ms frames")
    ap.add_argument("--voice-rms", type=float, default=0.01, help="frame RMS above which reply audio counts as voiced")
    ap.add_argument("--no-profile", dest="profile", action="store_false")
    ap.add_argument("--warmup", type=int, default=1, help="warm-up questions not recorded")
    ap.add_argument("--no-tts-guidance", action="store_true", help="TTS without classifier-free guidance (batch 1)")
    ap.add_argument("--tts-idle-frames", type=int, default=0,
                    help="pause the TTS+codec after this many quiet frames until the next token (0: off)")
    ap.add_argument("--tts-idle-rms", type=float, default=1e-3, help="decoded-speech RMS below which a frame is quiet")
    ap.add_argument("--tts-kv-chunk", type=int, default=None,
                    help="TTS backbone static K/V buffer growth in frames (0: eager caches; default: the fork's)")
    vc_variants.add_variant_args(ap)
    args = ap.parse_args()
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "run with HF_HUB_OFFLINE=1"

    model_path = glob.glob(os.path.expanduser(args.model))[0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    questions = json.loads(Path(args.questions).read_text())
    if args.ids:
        keep = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in keep]
    if args.limit:
        questions = questions[: args.limit]

    from mlx_audio.sts import load

    state_before = load_system_state()
    t0 = time.perf_counter()
    model = load(model_path)
    if args.tts_kv_chunk is not None:
        tts = model.tts_model.tts_model
        if not hasattr(type(tts), "backbone_buffer_chunk"):
            sys.exit("--tts-kv-chunk needs a fork with the static TTS backbone")
        tts.backbone_buffer_chunk = args.tts_kv_chunk
    load_s = time.perf_counter() - t0
    variant = vc_variants.apply_variant(model, args)
    vc_variants.patch_activation_dtype(args.act_dtype)
    mx.reset_peak_memory()
    timers = Timers()
    if args.profile:
        timers.instrument_model(model)
    audio_dir = Path(args.audio_dir)

    warm_dir = out / "_warmup"
    warm_dir.mkdir(exist_ok=True)
    for q in questions[: args.warmup]:
        run_question(model, q, audio_dir, warm_dir, args, timers)

    recs = []
    for q in questions:
        rec = run_question(model, q, audio_dir, out, args, timers)
        recs.append(rec)
        steady = rec["frame_ms"][1:]
        print(
            f"{q['id']} frames={rec['frames']} ms/frame={np.mean([f['total'] for f in steady]):.1f} "
            f"user={rec['user_transcript']!r} reply={rec['assistant_text']!r}",
            flush=True,
        )

    steady = [f for r in recs for f in r["frame_ms"][1:]]
    keys = sorted({k for f in steady for k in f})
    comp = {k: round(float(np.mean([f.get(k, 0.0) for f in steady])), 2) for k in keys}
    if args.profile:
        comp["rest"] = round(comp["total"] - sum(v for k, v in comp.items() if k != "total"), 2)
    totals = np.array([f["total"] for f in steady])
    summary = {
        "model": model_path.replace(os.path.expanduser("~"), "~"),  # no local user name in results
        "argv": sys.argv[1:],
        "variant": variant,
        "load_s": round(load_s, 1),
        "questions": len(recs),
        "frames": len(steady),
        "ms_per_frame": comp,
        "frame_ms_p50": round(float(np.median(totals)), 1),
        "frame_ms_p95": round(float(np.percentile(totals, 95)), 1),
        "rtf": round(float(totals.mean()) / 80.0, 3),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        "system_before": state_before,
        "system_after": load_system_state(),
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("ms_per_frame", "rtf", "frame_ms_p95", "peak_memory_gb", "variant")}, indent=1))


if __name__ == "__main__":
    main()
