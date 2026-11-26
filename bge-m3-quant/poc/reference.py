"""The reference bge-m3 dense embeddings: transformers, fp32, the CLS token's
last hidden state L2-normalized (BAAI's dense retrieval vector), for every
text of make_eval_set.py's set. Also the HF tokenizer's ids of a sample of
the texts, for the GGUF tokenizer check.

    python reference.py --model SNAPSHOT --eval eval.jsonl --out ref/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(args.eval, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).eval()

    ids = [tok(r["text"], truncation=True, max_length=args.max_length)["input_ids"] for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(ids[i]))
    emb = np.zeros((len(rows), model.config.hidden_size), np.float32)
    batch: list[int] = []

    def flush():
        if not batch:
            return
        width = max(len(ids[i]) for i in batch)
        x = torch.full((len(batch), width), tok.pad_token_id)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for j, i in enumerate(batch):
            x[j, : len(ids[i])] = torch.tensor(ids[i])
            mask[j, : len(ids[i])] = 1
        with torch.no_grad():
            h = model(input_ids=x, attention_mask=mask).last_hidden_state[:, 0]
        h = torch.nn.functional.normalize(h.float(), dim=-1)
        emb[batch] = h.numpy()
        batch.clear()

    for i in order:
        if batch and (len(batch) + 1) * max(len(ids[i]), max(len(ids[j]) for j in batch)) > args.batch_tokens:
            flush()
        batch.append(i)
    flush()
    np.save(out / "dense.npy", emb)
    json.dump([r["id"] for r in rows], open(out / "ids.json", "w"))
    sample = [i for i in range(len(rows)) if i % 25 == 0 or rows[i]["task"] == "long"]
    json.dump({rows[i]["id"]: ids[i] for i in sample}, open(out / "tokens.json", "w"))
    lengths = [len(x) for x in ids]
    print(f"REFERENCE_DONE {len(rows)} texts, tokens max {max(lengths)} mean {np.mean(lengths):.0f}", flush=True)


if __name__ == "__main__":
    main()
