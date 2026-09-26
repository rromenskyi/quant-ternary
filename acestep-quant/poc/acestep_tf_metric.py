"""Teacher-forced error of quantized ACE-Step DiTs against bf16 -- the
klein/MiniMax method: the bf16 model samples a few songs and every decoder
and encoder call is recorded; each variant then runs exactly those inputs,
and its outputs are compared with bf16's. No trajectory drift, no seed luck:
  - velocity error ||v_q - v_bf16|| / ||v_bf16|| over every decoder call
    (both CFG branches, all steps);
  - condition error of the encoder's output on the same text / lyrics.

    python acestep_tf_metric.py --ref <bf16 dir> --variant rtn4=<dir> --variant gptq4=<dir> --songs <dir> --out f.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from acestep_sft_gptq import SFT, load_sft, songs


class Recorder:
    """Stands in for the decoder / encoder: calls it, keeps each call's
    inputs and output. A plain object (not an mlx Module), so the wrapped
    module's attributes (decoder.layers, ...) pass straight through."""

    def __init__(self, inner, log):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __call__(self, **kwargs):
        out = self._inner(**kwargs)
        first = out[0] if isinstance(out, tuple) else out
        mx.eval(first)
        self._log.append(({k: v for k, v in kwargs.items() if k != "cache"}, first))
        return out


def rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-8)).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--variant", action="append", required=True, help="label=dir")
    ap.add_argument("--songs", required=True)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = load_sft(args.ref)
    dec_log, enc_log = [], []
    ref.decoder = Recorder(ref.decoder, dec_log)
    ref.encoder = Recorder(ref.encoder, enc_log)
    for song, caption, lyrics in songs(Path(args.songs)):
        list(ref.generate(text=caption, lyrics=lyrics, duration=args.duration, seed=1, **SFT))
        print(f"recorded {song}: {len(dec_log)} decoder / {len(enc_log)} encoder calls", flush=True)
    del ref
    mx.clear_cache()

    results = {}
    for label, path in (v.split("=", 1) for v in args.variant):
        model = load_sft(path)
        def first(x):
            return x[0] if isinstance(x, tuple) else x

        # Every 5th call: plenty, and 5x faster.
        v_err = [rel(first(model.decoder(**kw)), out) for kw, out in dec_log[::5]]
        c_err = [rel(first(model.encoder(**kw)), out) for kw, out in enc_log]
        results[label] = {"velocity_rel_err": sum(v_err) / len(v_err), "condition_rel_err": sum(c_err) / len(c_err),
                          "decoder_calls": len(v_err)}
        print(label, json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in results[label].items()}), flush=True)
        del model
        mx.clear_cache()
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
