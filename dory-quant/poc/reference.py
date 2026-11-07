"""Torch-reference (dory_reference.py) perplexity on wikitext-2 test, the
"HF bf16" baseline convert_mlx.py's check compares the MLX JANG build
against. There is no `transformers` AutoModel for Dory to use instead (see
dory_reference.py's module docstring), so this project's own reference
implementation, already cross-checked against the mlx_lm fork's
mlx_lm.models.dory (crosscheck_reference.py), stands in for it.

Writes --out/ref.json ({"ppl_bf16": ..., "ppl_bf16_code": ...}); keeps what
is already there instead of recomputing.

    python reference.py --model SNAPSHOT --out /workspace/ref --wikitext /workspace/data/wiki.test.raw
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dory_reference import load_dory
from gptq_dory import load_tokenizer  # the AutoConfig layer_types-validator fallback


def perplexity(model, ids: torch.Tensor, device: str, chunks: int, n: int) -> tuple[float, int]:
    stride = max(n, (len(ids) - n) // chunks)
    nll, count = 0.0, 0
    with torch.no_grad():
        for k in range(chunks):
            chunk = ids[k * stride: k * stride + n].unsqueeze(0).to(device)
            if chunk.shape[1] < n:
                break
            logits = model(input_ids=chunk).float()
            nll += torch.nn.functional.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="sum").item()
            count += n - 1
    return math.exp(nll / count), count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--code", help="held-out code text: a second perplexity, ppl_bf16_code")
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = json.load(open(out / "ref.json")) if (out / "ref.json").exists() else {}

    sets = []
    if "ppl_bf16" not in result:
        sets.append(("", args.wikitext, "ppl_bf16"))
    if args.code and "ppl_bf16_code" not in result:
        sets.append(("_code", args.code, "ppl_bf16_code"))
    if not sets:
        print("ref.json already has every requested perplexity, skipping", flush=True)
        return

    tok = load_tokenizer(args.model)
    model, _ = load_dory(args.model, device=args.device, dtype=torch.bfloat16)
    model.eval()
    for suffix, path, key in sets:
        ids = torch.tensor(tok.encode(open(path, encoding="utf-8", errors="ignore").read()))
        ppl, count = perplexity(model, ids, args.device, args.chunks, args.tokens)
        result[key] = ppl
        print(f"PPL{suffix} {ppl:.4f} ({count} tokens)", flush=True)
    json.dump(result, open(out / "ref.json", "w"), indent=2)
    print("REFERENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
