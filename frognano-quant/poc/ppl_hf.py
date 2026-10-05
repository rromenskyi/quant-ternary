"""Wikitext-2 test perplexity of a Qwen3.5 checkpoint's text decoder (HF,
bf16): the source model, or a gptq_qwen35.py output (its decoder weights
are on the quantization grid, so this measures the decoder's quantization
only -- embeddings/vision stay bf16 here; the MLX numbers come from
mlx_lm on the converted model).

    python ppl_hf.py --model /root/frognano-jang --wikitext /root/wikitext-2-raw/wiki.test.raw
"""

from __future__ import annotations

import argparse
import math

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--chunk-tokens", type=int, default=512)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    ids = tok(open(args.wikitext, encoding="utf-8").read(), return_tensors="pt")["input_ids"][0]
    n = args.chunk_tokens
    stride = max(n, (len(ids) - n) // args.chunks)
    nll, count = 0.0, 0
    with torch.no_grad():
        for k in range(args.chunks):
            chunk = ids[k * stride : k * stride + n].unsqueeze(0).cuda()
            if chunk.shape[1] < n:
                break
            hidden = model.model.language_model(input_ids=chunk, use_cache=False).last_hidden_state
            logits = model.lm_head(hidden).float()
            nll += torch.nn.functional.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="sum").item()
            count += n - 1
    print(f"PPL {math.exp(nll / count):.4f} ({count} tokens, {args.chunks}x{n})")


if __name__ == "__main__":
    main()
