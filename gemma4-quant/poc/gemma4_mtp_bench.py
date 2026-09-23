"""Speed + acceptance of a Gemma 4 MTP drafter against its main model, the
way mlx_lm.server actually drives it.

Two scenarios per prompt:
  - fresh: empty prompt cache (what offline tests usually measure);
  - cached-prefix: the cache already holds the chat template's opening
    tokens and only the rest of the prompt is passed -- what the server
    does on every request that reuses its prompt cache. This is the case a
    drafter RoPE-position bug hid in (fixed in ipsupport-llc/mlx-lm#5):
    +48% fresh, but 33 vs 32.5 tok/s in LLMTray.

Prints a table and MTP_BENCH_PASSED when MTP beats plain decoding by at
least --min-speedup in EVERY scenario (exit 1 otherwise).

    python gemma4_mtp_bench.py --model roman220220/gemma-4-26B-A4B-it-gptq-mlx-jang \
        --drafter ./g4-26b-assistant-8bit
"""

import argparse
import sys

PROMPTS = [
    "Write a Python function that checks whether a number is prime, with a docstring.",
    "Explain how a hash map works, in detail.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--num-draft-tokens", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--kv-bits", type=int, default=None)
    ap.add_argument("--cached-prefix", type=int, default=4, help="prompt tokens pre-filled into the cache")
    ap.add_argument("--min-speedup", type=float, default=1.15)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(args.model)
    drafter, _ = load(args.drafter)
    kv = {"kv_bits": args.kv_bits, "quantized_kv_start": 0} if args.kv_bits else {}

    def run(prompt, draft, cached):
        cache = make_prompt_cache(model)
        if cached:
            model(mx.array(prompt[:cached])[None], cache=cache)
        rest = prompt[cached:]
        n = acc = 0
        last = None
        for r in stream_generate(
            model, tok, rest, max_tokens=args.max_tokens, draft_model=draft,
            num_draft_tokens=args.num_draft_tokens, prompt_cache=cache, **kv,
        ):
            n += 1
            acc += r.from_draft
            last = r
        return last.generation_tps, acc / max(n, 1)

    # Warm-up: first call pays Metal kernel compilation.
    warm = tok.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    run(warm, drafter, 0)

    ok = True
    print(f"{'scenario':<16}{'prompt':<8}{'plain':>9}{'mtp':>9}{'speedup':>9}{'drafted':>9}")
    for i, q in enumerate(PROMPTS):
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True)
        for name, cached in (("fresh", 0), ("cached-prefix", args.cached_prefix)):
            plain, _ = run(prompt, None, cached)
            mtp, drafted = run(prompt, drafter, cached)
            speedup = mtp / plain
            ok &= speedup >= args.min_speedup
            print(f"{name:<16}{i:<8}{plain:>8.1f} {mtp:>8.1f} {speedup:>8.2f}x {drafted:>8.0%}", flush=True)
    print("MTP_BENCH_PASSED" if ok else f"MTP_BENCH_FAILED: speedup below {args.min_speedup}x somewhere")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
