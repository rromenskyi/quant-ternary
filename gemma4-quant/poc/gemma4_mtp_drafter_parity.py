"""Parity of mlx-lm's gemma4_assistant (MTP drafter) against transformers'
Gemma4AssistantForCausalLM, on the real drafter weights, fp32, with synthetic
main-model inputs (random hidden state + random KV for the full and the
sliding layer, the sliding one at exactly `sliding_window` keys).

Two halves because the two frameworks usually live in different venvs:

    python gemma4_mtp_drafter_parity.py dump  --drafter google/gemma-4-26B-A4B-it-assistant --out ref.npz   # torch
    python gemma4_mtp_drafter_parity.py check --drafter google/gemma-4-26B-A4B-it-assistant --ref ref.npz   # mlx

`check` prints DRAFTER_PARITY_PASSED when argmax matches and the logits /
next-hidden max abs diff stay under --tol (measured on the 26B drafter:
0.012 on a logit scale of 33).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def dump(args) -> int:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import Gemma4AssistantForCausalLM

    path = Path(args.drafter) if Path(args.drafter).exists() else Path(snapshot_download(args.drafter))
    model = Gemma4AssistantForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    # Shapes from the raw config.json: transformers' config object doesn't
    # expose every field (e.g. num_global_key_value_heads).
    raw = json.loads((path / "config.json").read_text())
    tc = raw["text_config"]
    B = raw["backbone_hidden_size"]
    rng = np.random.default_rng(0)
    L, W = args.context, tc["sliding_window"]
    full_kv_heads = tc.get("num_global_key_value_heads") or tc["num_key_value_heads"]
    arrays = {
        "emb": rng.standard_normal((1, 1, 2 * B)),
        "kf": rng.standard_normal((1, full_kv_heads, L, tc["global_head_dim"])),
        "vf": rng.standard_normal((1, full_kv_heads, L, tc["global_head_dim"])),
        "ks": rng.standard_normal((1, tc["num_key_value_heads"], W, tc["head_dim"])),
        "vs": rng.standard_normal((1, tc["num_key_value_heads"], W, tc["head_dim"])),
    }
    arrays = {k: v.astype(np.float32) for k, v in arrays.items()}
    t = torch.from_numpy
    with torch.no_grad():
        o = model(
            inputs_embeds=t(arrays["emb"]),
            position_ids=torch.tensor([[L]]),
            shared_kv_states={
                "full_attention": (t(arrays["kf"]), t(arrays["vf"])),
                "sliding_attention": (t(arrays["ks"]), t(arrays["vs"])),
            },
        )
    np.savez(args.out, pos=L, logits=o.logits.numpy(), hid=o.last_hidden_state.numpy(), **arrays)
    print(f"wrote {args.out}: logits {tuple(o.logits.shape)}, argmax {int(o.logits[0, 0].argmax())}")
    return 0


def check(args) -> int:
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_model

    path = Path(args.drafter) if Path(args.drafter).exists() else Path(snapshot_download(args.drafter))
    model, _ = load_model(path)
    model.set_dtype(mx.float32)
    ref = np.load(args.ref)
    a = lambda k: mx.array(ref[k])
    logits, hid = model(
        a("emb"),
        {"full_attention": (a("kf"), a("vf")), "sliding_attention": (a("ks"), a("vs"))},
        int(ref["pos"]),
    )
    logits, hid = np.array(logits), np.array(hid)
    d_logit = float(np.abs(logits - ref["logits"]).max())
    d_hid = float(np.abs(hid - ref["hid"]).max())
    same_argmax = int(logits[0, 0].argmax()) == int(ref["logits"][0, 0].argmax())
    print(f"logits max|diff| {d_logit:.4f} (scale {np.abs(ref['logits']).max():.1f}), "
          f"hidden max|diff| {d_hid:.4f}, argmax equal: {same_argmax}")
    if same_argmax and d_logit < args.tol and d_hid < args.tol:
        print("DRAFTER_PARITY_PASSED")
        return 0
    print("DRAFTER_PARITY_FAILED")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--drafter", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--context", type=int, default=1500, help="length of the synthetic full-attention KV")
    c = sub.add_parser("check")
    c.add_argument("--drafter", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--tol", type=float, default=0.1)
    args = ap.parse_args()
    return dump(args) if args.cmd == "dump" else check(args)


if __name__ == "__main__":
    sys.exit(main())
