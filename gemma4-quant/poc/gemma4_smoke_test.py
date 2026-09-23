"""Post-splice smoke test for a Gemma 4 MLX checkpoint (E4B or 26B-A4B).

Checks what has actually broken in past releases, before anything gets
uploaded:
  - it loads with mlx-lm and answers a chat prompt with real text;
  - no control tokens leak into the decoded answer (<|channel>, <|turn>, ...);
  - no bf16 weights are left (every Linear / Embedding is quantized) -- the
    first E4B "8-bit" release silently shipped its vision tower in bf16;
  - optional --image: answers a question about a real photo through the same
    image path mlx_lm.server uses (mlx_lm.multimodal, ipsupport-llc/mlx-lm);
  - optional --audio (E4B, 16kHz mono WAV): transcribes a spoken clip through
    the audio tower (a clip made with macOS `say` is enough).

Prints SMOKE_TEST_PASSED on success, exits non-zero otherwise.

    python gemma4_smoke_test.py --model /workspace/output-26b-jang \
        --image cats.jpg --expect-in-image-answer cat
"""

import argparse
import json
import re
import sys
import wave
from pathlib import Path

LEAK_MARKERS = ("<|channel>", "<channel|>", "<|turn>", "<turn|>", "<|think|>", "<|tool_call>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="List three facts about the Roman Empire, one line each.")
    ap.add_argument("--image", default=None, help="optional photo for a vision check")
    ap.add_argument("--image-question", default="What animals are in this picture? One sentence.")
    ap.add_argument("--expect-in-image-answer", default=None, help="substring the vision answer must contain")
    ap.add_argument("--audio", default=None, help="optional 16kHz mono WAV of speech (E4B only)")
    ap.add_argument("--expect-in-audio-answer", default=None, help="substring the transcription must contain")
    ap.add_argument(
        "--keep-float", action="append", default=[], metavar="REGEX",
        help="weights allowed to stay float on purpose (same regexes the splice got), e.g. 'router\\.proj'",
    )
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load, stream_generate

    model, tok = load(args.model)
    failures = []

    # 1. No large float weights left (Linear, Embedding, SwitchLinear experts,
    # vision tower...): every big ".weight" must have a sibling ".scales".
    params = dict(tree_flatten(model.parameters()))
    float_types = (mx.bfloat16, mx.float16, mx.float32)
    keep = [re.compile(p) for p in args.keep_float]
    unquantized = [
        k for k, v in params.items()
        if k.endswith(".weight") and v.dtype in float_types and v.size >= 100_000
        and k[: -len(".weight")] + ".scales" not in params
        and not any(p.search(k) for p in keep)
    ]
    if unquantized:
        failures.append(f"{len(unquantized)} large unquantized weights: {unquantized[:5]}")
    print(f"large unquantized weights: {len(unquantized)}", flush=True)

    def answer(messages, **kw):
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
        out, n = "", 0
        for r in stream_generate(model, tok, prompt, max_tokens=args.max_tokens, **kw):
            out += r.text
            n += 1
        return out, n

    def final_answer(text):
        # Thinking-mode output: the answer follows the closing channel tag.
        return text.split("<channel|>")[-1].strip()

    # 2. Text chat.
    text, n = answer([{"role": "user", "content": args.prompt}])
    reply = final_answer(text)
    print(f"text: {n} tokens, reply={reply[:200]!r}", flush=True)
    if len(reply) < 20:
        failures.append(f"text reply too short: {reply!r}")
    leaked = [m for m in LEAK_MARKERS if m in reply]
    if leaked:
        failures.append(f"control tokens leaked into the reply: {leaked}")

    # 3. Vision.
    if args.image:
        from mlx_lm.multimodal import load_image_inputs

        ii = load_image_inputs(model, args.model)
        if ii is None:
            failures.append("model has no usable vision tower / processor_config.json")
        else:
            blob = Path(args.image).read_bytes()
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.image_question}]}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
            ids, emb, _ = ii.build(prompt, [blob])
            out = ""
            for r in stream_generate(model, tok, ids, max_tokens=args.max_tokens, input_embeddings=emb):
                out += r.text
            vreply = final_answer(out)
            print(f"vision: reply={vreply[:200]!r}", flush=True)
            if args.expect_in_image_answer and args.expect_in_image_answer.lower() not in vreply.lower():
                failures.append(f"vision reply lacks {args.expect_in_image_answer!r}: {vreply!r}")

    # 4. Audio (E4B): expand the single audio placeholder into boa + N soft
    # tokens + eoa, fuse the audio tower's features, transcribe.
    if args.audio:
        import numpy as np
        from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

        if getattr(model, "audio_tower", None) is None:
            failures.append("--audio given but the model has no audio tower")
        else:
            with wave.open(args.audio) as w:
                if w.getframerate() != 16000 or w.getnchannels() != 1:
                    raise SystemExit("--audio must be 16kHz mono WAV (afconvert -f WAVE -d LEI16@16000 -c 1)")
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            f = Gemma4AudioFeatureExtractor()([pcm], sampling_rate=16000, return_tensors="np")
            feats, fmask = mx.array(f["input_features"]), mx.array(f["input_features_mask"])
            _, valid = model.audio_tower(feats, fmask)
            n_audio = int(valid.sum().item())
            cfg = json.loads((Path(args.model) / "config.json").read_text())
            audio_id = model.args.audio_token_id
            msgs = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Transcribe this audio exactly."}]}]
            ids = []
            for t in tok.apply_chat_template(msgs, add_generation_prompt=True):
                ids += [cfg["boa_token_id"]] + [audio_id] * n_audio + [cfg["eoa_token_id"]] if t == audio_id else [t]
            fused, _ = model._fuse_multimodal_inputs(mx.array(ids)[None], None, None, feats, fmask)
            pad = model.language_model.model.config.pad_token_id
            gen_ids = [pad if t == audio_id else t for t in ids]
            out = "".join(
                r.text for r in stream_generate(model, tok, gen_ids, max_tokens=args.max_tokens, input_embeddings=fused[0])
            )
            areply = final_answer(out)
            print(f"audio: {n_audio} soft tokens, reply={areply[:200]!r}", flush=True)
            if args.expect_in_audio_answer and args.expect_in_audio_answer.lower() not in areply.lower():
                failures.append(f"audio reply lacks {args.expect_in_audio_answer!r}: {areply!r}")

    print(f"peak memory: {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    if failures:
        for f in failures:
            print("SMOKE_TEST_FAILED:", f, flush=True)
        return 1
    print("SMOKE_TEST_PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
