#!/usr/bin/env python3
"""[pod] Capture the LLM's real per-frame inputs for GPTQ calibration.

Streams every calibration clip through the unquantized (bf16) duplex session
of the mlx-audio fork -- system-prompt prefill, speech, then silence while the
model answers -- and saves the fused embedding the LLM receives at each 80 ms
frame (audio embedding + previous text/function token embeddings). GPTQ
(gptq_llm.py) then replays these sequences through the LLM in prefill mode,
which computes the same function as the frame-by-frame cached decode.

TTS and the codec do not feed back into the LLM, so they are stubbed out here.

  python calib_capture.py --model /workspace/vc-bf16 --audio calib/audio --out /workspace/calib
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vc_variants  # noqa: E402

PROMPTS = [
    "Be concise and answer in one sentence.",
    None,  # the model's default system prompt
    "You are a friendly voice assistant. Keep answers short and natural.",
]


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tail-s", type=float, default=8.0)
    ap.add_argument("--idle-frames", type=int, default=25)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th clip starting at i")
    ap.add_argument("--act-dtype", default="bf16", choices=sorted(vc_variants.DTYPES))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from mlx_audio.sts import load
    from mlx_audio.sts.models.nemotron_voicechat.streaming import VoiceChatStreamingSession

    model = load(args.model)
    tts = model.tts_model.tts_model
    ratio = model.tts_model.audio_codec.waveform_to_token_ratio

    def no_tts(current, previous_codes, cache, *, silence_codes, **_):
        return mx.broadcast_to(silence_codes, previous_codes.shape).astype(mx.int32), cache

    tts.generate_step = no_tts
    model.tts_model.audio_codec.decode_step = lambda codes, cache: mx.zeros((1, 1, ratio))

    vc_variants.patch_activation_dtype(args.act_dtype)
    captured: list[mx.array] = []
    lang = VoiceChatStreamingSession._language_step

    def recording(self, x):
        captured.append(x.astype(mx.float32))
        return lang(self, x)

    VoiceChatStreamingSession._language_step = recording

    clips = sorted(Path(args.audio).glob("c*.wav"))[: args.limit]
    shard_i, shard_n = map(int, args.shard.split("/"))
    manifest = []
    t0 = time.time()
    for i, clip in enumerate(clips):
        dst = out / f"{clip.stem}.npy"
        if i % shard_n != shard_i or dst.exists():
            continue
        captured.clear()
        prompt = PROMPTS[i % len(PROMPTS)]
        session = model.create_duplex_session(system_prompt=prompt, seed=i)
        fs = session.frame_samples
        speech = read_wav(clip)
        speech = np.concatenate([speech, np.zeros((-len(speech)) % fs, np.float32)])
        n_speech = len(speech) // fs
        text, last_text = [], None
        for f in range(n_speech + int(args.max_tail_s / 0.08)):
            chunk = speech[f * fs : (f + 1) * fs] if f < n_speech else np.zeros(fs, np.float32)
            for ev in session.push_audio(chunk, sample_rate=16000):
                if ev.kind == "assistant_text_delta":
                    text.append(ev.delta)
                    last_text = f
            if f >= n_speech and last_text is not None and f - last_text >= args.idle_frames:
                break
        seq = mx.concatenate(captured, axis=1)[0]  # (T, hidden)
        np.save(dst, np.array(seq))
        manifest.append({"id": clip.stem, "frames": int(seq.shape[0]), "prompt": prompt, "reply": "".join(text)})
        print(f"[{i + 1}/{len(clips)}] {clip.stem} T={seq.shape[0]} {time.time() - t0:.0f}s reply={''.join(text)[:80]!r}", flush=True)
    (out / f"manifest_{shard_i}of{shard_n}.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
