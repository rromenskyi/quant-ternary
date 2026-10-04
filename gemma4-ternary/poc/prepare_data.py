"""Distillation text: packed, tokenized sequences in .npy shards.

    python prepare_data.py --master masters/gemma-4-12B-qat-unq --out data \\
        --tokens 50000000 --seq 2048 \\
        --source HuggingFaceFW/fineweb-edu:sample-10BT:train:text:0.3 \\
        --source HuggingFaceTB/smoltalk:all:train:messages:0.7 \\
        --eval-source HuggingFaceTB/smoltalk:all:test:messages:1

A source is <repo>:<config>:<split>:<field>:<weight>; a field named
"messages" is rendered with the model's chat template (thinking closed),
any other field is plain text after BOS. Documents are concatenated and cut
into --seq token rows: --eval-seqs rows from the --eval-source streams
(eval.npy; a held-out split, and chat, which is how the model is used --
Gemma -it is very unsure on raw web text, so KL there overstates every
quantization: 8-bit RTN scored KL 0.045 on it), then train_00000.npy, ...
of --shard-seqs rows from the --source streams.

Resumable: the streams are deterministic (fixed seed, weighted pick per
document), so a rerun regenerates the same rows and writes only the files
that are missing. progress.json tracks it for the dashboard.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ternary_lib import save_npy, write_json  # noqa: E402


def doc_stream(sources, seed, tok):
    """Documents from every source, interleaved by weight with a seeded RNG
    (deterministic, so a rerun reproduces the same rows)."""
    from datasets import load_dataset

    streams, weights, fields = [], [], []
    for spec in sources:
        repo, config, split, field, weight = spec.rsplit(":", 4)
        streams.append(iter(load_dataset(repo, config if config != "-" else None, split=split, streaming=True)))
        weights.append(float(weight))
        fields.append(field)
    # The chat template renders past model turns as "<|turn>model\n<text>",
    # but a generation prompt ends "<|turn>model\n<|channel>thought\n<channel|>"
    # (thinking closed): give every model turn the inference prefix, so the
    # student trains on the context it will generate in.
    u = [{"role": "user", "content": "x"}]
    base = tok.apply_chat_template(u, tokenize=False, enable_thinking=False)
    gen_prefix = tok.apply_chat_template(u, tokenize=False, add_generation_prompt=True, enable_thinking=False)[len(base):]
    turn = tok.apply_chat_template(u + [{"role": "assistant", "content": "\x00"}], tokenize=False,
                                   enable_thinking=False)[len(base):].split("\x00")[0]
    rng = np.random.default_rng(seed)
    p = np.asarray(weights) / sum(weights)
    while True:
        i = int(rng.choice(len(streams), p=p))
        ex = next(streams[i])
        val = ex[fields[i]]
        if fields[i] == "messages":
            text = tok.apply_chat_template(val, tokenize=False, enable_thinking=False)
            if gen_prefix != turn:
                text = text.replace(turn, gen_prefix)
            ids = tok.encode(text, add_special_tokens=False)
            if not ids or ids[0] != tok.bos_token_id:
                ids = [tok.bos_token_id] + ids
        else:
            ids = [tok.bos_token_id] + tok.encode(val, add_special_tokens=False)
        yield ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="tokenizer source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, required=True, help="train tokens to prepare")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--shard-seqs", type=int, default=256)
    ap.add_argument("--eval-seqs", type=int, default=64)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--eval-source", action="append", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_shards = -(-args.tokens // (args.seq * args.shard_seqs))
    want = [out / "eval.npy"] + [out / f"train_{i:05d}.npy" for i in range(n_shards)]
    write_json(out / "meta.json", {"seq": args.seq, "shard_seqs": args.shard_seqs, "eval_seqs": args.eval_seqs,
                                   "shards": n_shards, "sources": args.source, "eval_sources": args.eval_source,
                                   "seed": args.seed})
    done = sum(p.exists() for p in want)
    write_json(out / "progress.json", {"done": done, "total": len(want)})
    if done == len(want):
        print(f"all {len(want)} files present")
        return

    tok = AutoTokenizer.from_pretrained(args.master)

    def fill(sources, seed, files, size):
        nonlocal done
        if all(f.exists() for f in files):
            return
        buf: list[int] = []
        rows: list[np.ndarray] = []
        fi = 0
        for ids in doc_stream(sources, seed, tok):
            buf.extend(ids)
            while len(buf) >= args.seq:
                rows.append(np.asarray(buf[: args.seq], dtype=np.uint32))
                del buf[: args.seq]
                if len(rows) == size:
                    if not files[fi].exists():
                        save_npy(files[fi], np.stack(rows))
                        done += 1
                        write_json(out / "progress.json", {"done": done, "total": len(want)})
                        print(f"{files[fi].name}  ({done}/{len(want)})", flush=True)
                    rows = []
                    fi += 1
                    if fi == len(files):
                        return

    fill(args.eval_source, args.seed + 1, want[:1], args.eval_seqs)
    fill(args.source, args.seed, want[1:], args.shard_seqs)

if __name__ == "__main__":
    main()
    # Streaming readers leave native threads that crash interpreter teardown
    # (core dump, nonzero exit) after every file is safely written.
    sys.stdout.flush()
    os._exit(0)
