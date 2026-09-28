#!/usr/bin/env python3
"""Interleaved A/B speed benchmark of variants inside ONE process.

The Mac this runs on is shared (other apps keep the GPU 25-75% busy), so
run-to-run noise between separate vc_eval.py processes is larger than most
of the effects we measure. Here all variants share one loaded model: each
variant is a structural clone of the perception / TTS modules (arrays are
shared, only the quantized copies cost memory) plus optional bf16 copies of
the LLM quantization scales, and the variants are timed in alternating
rounds on the same audio. Report = median over rounds of mean ms/frame.

  HF_HUB_OFFLINE=1 python bench_ab.py --out results/ab.json \
     --variant 'base=' --variant 'bf16=--act-dtype bf16 --scales-dtype bf16' \
     --variant 'p4=--act-dtype bf16 --scales-dtype bf16 --rtn stt_model.perception:4:64'

Only --act-dtype/--scales-dtype/--rtn/--rtn-skip/--compile are honoured.
Quality is NOT measured here -- use vc_eval.py + score.py for that.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shlex
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vc_eval  # noqa: E402
import vc_variants  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def clone(m):
    """Copy module structure, share arrays (so quantizing the clone leaves the original)."""
    if isinstance(m, nn.Module):
        new = dict.__new__(type(m))
        new.__dict__.update(m.__dict__)
        for k, v in m.items():
            dict.__setitem__(new, k, clone(v))
        return new
    if isinstance(m, list):
        return [clone(x) for x in m]
    if isinstance(m, dict):
        return {k: clone(v) for k, v in m.items()}
    return m


class Holder(nn.Module):
    pass


def build_variant(model, spec: str, llm_scales: dict):
    ap = argparse.ArgumentParser()
    vc_variants.add_variant_args(ap)
    args = ap.parse_args(shlex.split(spec))
    h = Holder()
    h.stt_model = Holder()
    h.tts_model = Holder()
    h.stt_model.perception = clone(model.stt_model.perception)
    h.tts_model.tts_model = clone(model.tts_model.tts_model)
    if args.rtn:
        vc_variants.apply_rtn(h, args.rtn, args.rtn_skip, args.rtn_mode)
    mx.eval(h.parameters())
    return {
        "perception": h.stt_model.perception,
        "tts": h.tts_model.tts_model,
        "scales": llm_scales[args.scales_dtype or "orig"],
        "act": args.act_dtype,
        "fast": (args.tts_mog_gather, args.compile_tts_codes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=vc_eval.DEFAULT_MODEL)
    ap.add_argument("--variant", action="append", required=True, metavar="NAME=FLAGS")
    ap.add_argument("--ids", default="q01,q04")
    ap.add_argument("--tail-frames", type=int, default=45)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"

    from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState
    from mlx_audio.sts import load
    from mlx_audio.sts.models.nemotron_voicechat.streaming import VoiceChatStreamingSession

    model = load(glob.glob(os.path.expanduser(args.model))[0])
    state = {"act": None}
    push, lang = ConformerStreamingState.push, VoiceChatStreamingSession._language_step
    cast = lambda x: x if state["act"] is None else x.astype(vc_variants.DTYPES[state["act"]])  # noqa: E731
    ConformerStreamingState.push = lambda self, mel, **k: push(self, cast(mel), **k)
    VoiceChatStreamingSession._language_step = lambda self, x: lang(self, cast(x))

    from mlx.utils import tree_flatten

    orig = [(k, v) for k, v in tree_flatten(model.stt_model.parameters()) if k.endswith((".scales", ".biases"))]
    llm_scales = {"orig": orig, "bf16": [(k, v.astype(mx.bfloat16)) for k, v in orig]}
    mx.eval([v for _, v in llm_scales["bf16"]])
    from mlx.utils import tree_unflatten

    variants = {}
    for item in args.variant:
        name, _, spec = item.partition("=")
        variants[name] = build_variant(model, spec, llm_scales)
        print(f"built {name}: {spec}", flush=True)

    def activate(v):
        model.stt_model.perception = v["perception"]
        model.tts_model.tts_model = v["tts"]
        model.stt_model.update(tree_unflatten(v["scales"]))
        state["act"] = v["act"]
        vc_variants.patch_tts_fast(*v["fast"])

    timers = vc_eval.Timers()
    audio_dir = HERE / "eval/audio"
    clips = [vc_eval.read_wav(audio_dir / f"{i}.wav") for i in args.ids.split(",")]

    def run_once():
        per = collections.defaultdict(list)
        timers.instrument_model(model)  # wraps the currently active TTS/codec
        for pcm in clips:
            s = model.create_duplex_session(system_prompt=vc_eval.PROMPT, seed=0)
            timers.instrument_session(s)
            timers.take()
            fs = s.frame_samples
            pcm = np.concatenate([pcm, np.zeros((-len(pcm)) % fs + fs * args.tail_frames, np.float32)])
            for i in range(0, len(pcm), fs):
                t0 = time.perf_counter()
                s.push_audio(pcm[i : i + fs], sample_rate=16000)
                comp = timers.take()
                comp["total"] = time.perf_counter() - t0
                if i:
                    for k, v in comp.items():
                        per[k].append(v * 1000)
        # unwrap model-level instrumentation
        for obj, name in ((model.tts_model.tts_model, "generate_step"), (model.tts_model.audio_codec, "decode_step")):
            try:
                delattr(obj, name)
            except AttributeError:
                pass
        return {k: float(np.mean(v)) for k, v in per.items()}

    results = collections.defaultdict(list)
    names = list(variants)
    for r in range(args.rounds):
        order = names if r % 2 == 0 else names[::-1]
        for name in order:
            activate(variants[name])
            m = run_once()
            results[name].append(m)
            print(f"round {r} {name}: total {m['total']:.1f} ms " + " ".join(f"{k}={v:.1f}" for k, v in sorted(m.items()) if k != "total"), flush=True)

    summary = {}
    for name, runs in results.items():
        keys = runs[0].keys()
        summary[name] = {k: round(float(np.median([x[k] for x in runs])), 2) for k in keys}
        summary[name]["rtf"] = round(summary[name]["total"] / 80, 3)
        summary[name]["spec"] = dict(i.partition("=")[::2] for i in args.variant)[name]
    out = {"summary": summary, "rounds": results, "system": vc_eval.load_system_state(), "argv": sys.argv[1:]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
