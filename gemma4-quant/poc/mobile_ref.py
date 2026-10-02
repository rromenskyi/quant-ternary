"""Google's mobile QAT checkpoint in transformers (the reference) against its
MLX conversion: logits on the same windows, KL both ways of the activation
quantization (the static int8 "a8o8" on, as on a phone NPU; off, as the MLX
build runs).

  python mobile_ref.py --hf e2b-mobile --mlx E2B-mobile-mlx --text wiki.test.raw --windows 8
"""
import argparse, json, math
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True)
ap.add_argument("--mlx", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--windows", type=int, default=8)
ap.add_argument("--json")
args = ap.parse_args()

from qat_text import windows
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(args.hf)
wins = windows(tok, args.text, 512, args.windows, chat=True)

def logprobs_torch(model):
    out = []
    with torch.no_grad():
        for w, s in wins:
            lg = model(torch.tensor([w], device="cuda")).logits[0, s:-1].float()
            out.append(torch.log_softmax(lg, -1).cpu().numpy())
    return out

ref = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.bfloat16).to("cuda").eval()
with_srq = logprobs_torch(ref)
# Off: an activation scale of 0 is "uncalibrated" -- apply_srq passes through.
n_off = 0
for name, p in ref.named_parameters():
    if name.endswith(("input_activation_scale", "output_activation_scale", "k_cache_scale", "v_cache_scale")):
        p.data.zero_(); n_off += 1
no_srq = logprobs_torch(ref)
del ref; torch.cuda.empty_cache()

import mlx.core as mx
from mlx_lm import load
m, _ = load(args.mlx)
ours = []
for w, s in wins:
    lg = m(mx.array(w)[None])[0, s:-1].astype(mx.float32)
    ours.append(np.array(lg - mx.logsumexp(lg, axis=-1, keepdims=True)))

def kl(p_list, q_list):
    tot = n = agree = 0
    for p, q in zip(p_list, q_list):
        tot += (np.exp(p) * (p - q)).sum(); n += p.shape[0]; agree += (p.argmax(-1) == q.argmax(-1)).sum()
    return tot / n, agree / n
res = {}
for name, a, b in (("mlx vs transformers, activations bf16", no_srq, ours),
                   ("mlx vs transformers, int8 activations (NPU)", with_srq, ours),
                   ("transformers: int8 vs bf16 activations", with_srq, no_srq)):
    k, t1 = kl(a, b)
    res[name] = {"kl": float(k), "top1": float(t1)}
    print(f"{name}: KL {k:.5f}, top-1 agree {t1:.2%}", flush=True)
print(f"({n_off} activation/KV scales zeroed for the bf16 run; {len(wins)} windows)")
if args.json:
    json.dump(res, open(args.json, "w"), indent=1)
