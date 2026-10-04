"""Teacher-generated chat data: the master answers real prompts in the exact
inference format, in both thinking modes. Runs in the vLLM venv.

    python gen_teacher_data.py --master masters/gemma-4-12B-text --out gen \\
        --source HuggingFaceTB/smoltalk:all:train --tokens 16000000 --think-frac 0.5

For each conversation of --source, one assistant turn at random is the
target: the turns before it are rendered by the chat template as they are
(history keeps no thinking prefix, as at inference), then
add_generation_prompt with enable_thinking on (--think-frac of the
prompts) or off, and the master samples the reply (temperature 1, top-p
0.95, top-k 64: Gemma's defaults). A document is prompt ids + reply ids,
the stop token included.

Shards of --shard-prompts prompts, written as gen/shard_NNNNN.npz
(ids: uint32 concatenation, offs: document starts) with a JSON line in
gen/progress.json; a rerun skips finished shards (prompts and seeds are a
function of the shard index). Stops once --tokens generated tokens exist.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def prompts_for(source: str, start: int, n: int, tok, think_frac: float, seed: int, max_prompt: int):
    from datasets import load_dataset

    repo, config, split = source.rsplit(":", 2)
    ds = load_dataset(repo, config if config != "-" else None, split=split, streaming=True).skip(start)
    rng = np.random.default_rng(seed)
    out = []
    for ex in itertools.islice(ds, n):
        msgs = ex["messages"]
        turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0]
        if not turns:
            continue
        i = int(rng.choice(turns))
        think = bool(rng.random() < think_frac)
        text = tok.apply_chat_template(msgs[:i], add_generation_prompt=True, tokenize=False, enable_thinking=think)
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) <= max_prompt:
            out.append((ids, think))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="text-only view of the master (vLLM Gemma4ForCausalLM)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", required=True, help="<repo>:<config>:<split> with a 'messages' field")
    ap.add_argument("--tokens", type=int, required=True, help="generated tokens to stop at")
    ap.add_argument("--shard-prompts", type=int, default=4096)
    ap.add_argument("--think-frac", type=float, default=0.5)
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--max-new-think", type=int, default=2560)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prog_f = out / "progress.json"
    prog = json.loads(prog_f.read_text()) if prog_f.exists() else {"shards": {}}
    gen_total = lambda: sum(s["gen_tokens"] for s in prog["shards"].values())

    def save_prog():
        prog["gen_tokens"] = gen_total()
        prog["target"] = args.tokens
        tmp = prog_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(prog, indent=1))
        os.replace(tmp, prog_f)

    save_prog()
    if gen_total() >= args.tokens:
        print(f"done: {gen_total()} generated tokens")
        return

    from vllm import LLM, SamplingParams, TokensPrompt

    tok = AutoTokenizer.from_pretrained(args.master)
    llm = LLM(model=args.master, dtype="bfloat16", max_model_len=args.max_prompt + args.max_new_think,
              gpu_memory_utilization=args.gpu_mem, seed=args.seed)
    j = 0
    while gen_total() < args.tokens:
        name = f"shard_{j:05d}"
        if name in prog["shards"] and (out / f"{name}.npz").exists():
            j += 1
            continue
        t0 = time.time()
        ps = prompts_for(args.source, j * args.shard_prompts, args.shard_prompts, tok, args.think_frac,
                         args.seed + j, args.max_prompt)
        sps = [SamplingParams(temperature=1.0, top_p=0.95, top_k=64, seed=args.seed * 1_000_003 + j * 65_536 + k,
                              max_tokens=args.max_new_think if think else args.max_new)
               for k, (_, think) in enumerate(ps)]
        res = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids, _ in ps], sps, use_tqdm=False)
        docs, gen_tokens, finished = [], 0, 0
        for (ids, _), r in zip(ps, res):
            o = r.outputs[0]
            docs.append(np.asarray(list(ids) + list(o.token_ids), dtype=np.uint32))
            gen_tokens += len(o.token_ids)
            finished += o.finish_reason == "stop"
        offs = np.cumsum([0] + [len(d) for d in docs[:-1]]).astype(np.int64)
        tmp = out / f"{name}.tmp.npz"
        np.savez(tmp, ids=np.concatenate(docs), offs=offs)
        os.replace(tmp, out / f"{name}.npz")
        dt = time.time() - t0
        prog["shards"][name] = {"docs": len(docs), "gen_tokens": gen_tokens,
                                "tokens": int(sum(len(d) for d in docs)), "finished": finished,
                                "think": int(sum(t for _, t in ps)), "seconds": round(dt), "tok_s": round(gen_tokens / dt)}
        save_prog()
        print(f"{name}: {len(docs)} docs, {gen_tokens} generated ({gen_tokens / dt:.0f} tok/s), "
              f"finished {finished}, total {gen_total()}/{args.tokens}", flush=True)
        j += 1


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # vLLM / streaming threads crash interpreter teardown
