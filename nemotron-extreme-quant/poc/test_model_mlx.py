"""Smoke test for a finished MLX model (ours or a baseline to compare against):
coherence check (the degenerate-repetition heuristic from sanity_check_hf.py)
plus a basic coding-ability check.

Coding prompts are checked STATICALLY ONLY -- the generated code is parsed
(ast.parse) to confirm it's syntactically valid and defines the expected
function name, but is NEVER executed. Actually running arbitrary model-
generated code safely requires real sandboxing (a network-disabled,
throwaway Docker container -- a plain subprocess does NOT protect against
something like `os.system("rm -rf /")` running with our own permissions).
That's a separate, more involved project; this script deliberately stays on
the safe side of that line.

Usage:
    python poc/test_model_mlx.py --model ~/.lmstudio/models/local/nemotron-30b-mlx-3bit
    python poc/test_model_mlx.py --model models/gptq3bit-g64 --trust-remote-code
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import time

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

COHERENCE_PROMPTS = [
    "The capital of France is",
    "Write a one-sentence haiku about GPUs.",
    "2 + 2 =",
]

# (prompt, expected function name) -- checked via ast.parse, never executed.
CODING_PROMPTS = [
    (
        "Write a Python function `reverse_list(items)` that returns the "
        "input list reversed, without using slicing or the built-in reversed().",
        "reverse_list",
    ),
    (
        "Write a Python function `is_palindrome(s)` that returns True if the "
        "string s reads the same forwards and backwards, ignoring case.",
        "is_palindrome",
    ),
]

MIN_UNIQUE_TOKEN_RATIO = 0.35
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    return m.group(1) if m else text


def check_coherence(model, tokenizer, sampler, max_tokens: int) -> bool:
    any_degenerate = False
    for prompt in COHERENCE_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text = generate(model, tokenizer, prompt=rendered, max_tokens=max_tokens, sampler=sampler, verbose=False)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(token_ids)
        n_unique = len(set(token_ids))
        ratio = n_unique / max(n_tokens, 1)
        degenerate = ratio < MIN_UNIQUE_TOKEN_RATIO
        any_degenerate = any_degenerate or degenerate
        print(f"\n=== coherence prompt: {prompt!r} ===")
        print(text)
        print(f"--- unique_token_ratio={ratio:.2f} ({n_unique}/{n_tokens}) {'DEGENERATE' if degenerate else 'ok'} ---", flush=True)
    return not any_degenerate


def check_coding(model, tokenizer, sampler, max_tokens: int) -> bool:
    # This model "thinks" before answering (<think>...</think>), which alone
    # can burn 150-300+ tokens before any code appears -- give coding prompts
    # noticeably more headroom than the short coherence prompts, or a
    # perfectly fine answer gets cut off mid-thought and reads as a failure.
    max_tokens = max(max_tokens, 600)
    all_ok = True
    for prompt, fn_name in CODING_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text = generate(model, tokenizer, prompt=rendered, max_tokens=max_tokens, sampler=sampler, verbose=False)
        code = extract_code(text)

        print(f"\n=== coding prompt: {prompt!r} (expect def {fn_name}) ===")
        print(text)

        try:
            tree = ast.parse(code)
            fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            ok = fn_name in fn_names
        except SyntaxError as e:
            ok = False
            print(f"--- SyntaxError: {e} ---", flush=True)

        all_ok = all_ok and ok
        print(f"--- static_check={'ok' if ok else 'FAILED'} (parses + defines {fn_name}; NOT executed) ---", flush=True)
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    t0 = time.time()
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": args.trust_remote_code} if args.trust_remote_code else {})
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    sampler = make_sampler(1.0, 0.95)

    coherence_ok = check_coherence(model, tokenizer, sampler, args.max_tokens)
    coding_ok = check_coding(model, tokenizer, sampler, args.max_tokens)

    print(f"\n=== SUMMARY: coherence={'PASS' if coherence_ok else 'FAIL'} coding={'PASS' if coding_ok else 'FAIL'} ===", flush=True)
    if not (coherence_ok and coding_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
