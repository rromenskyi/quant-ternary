"""A tiny random Inkling 2 checkpoint in the published layout (model.llm.*,
model.visual.*, NVFP4 experts as w13/w2 + scale / amax /
outer_scale_denominator), from a real snapshot's config and tokenizer:
the dry run of the whole pipeline on a CPU (smoke_test.sh).

    python make_tiny_checkpoint.py --like SNAPSHOT --out /tmp/tiny-mini
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForImageTextToText

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def nvfp4(w: torch.Tensor) -> dict[str, torch.Tensor]:
    """[E, R, C] float -> packed E2M1 codes [E, R, C/2] (even element in the
    low nibble), an e4m3 scale per 16 values and a per-expert
    amax / outer_scale_denominator, as transformers' Inkling2LoopDequantizeExperts reads them."""
    E, R, C = w.shape
    w = w.float()
    amax = w.abs().amax(dim=(1, 2))
    denominator = torch.full((E,), 6.0 * 448.0)
    factor = (amax / denominator)[:, None, None, None]
    blocks = w.reshape(E, R, C // 16, 16)
    scale = (blocks.abs().amax(-1, keepdim=True) / 6.0 / factor).to(torch.float8_e4m3fn)
    x = blocks / (scale.float() * factor).clamp_min(1e-30)
    idx = (x.abs()[..., None] - E2M1).abs().argmin(-1)
    code = (idx | ((x < 0) & (idx > 0)).long() << 3).reshape(E, R, C).to(torch.uint8)
    return {"": code[..., 0::2] | (code[..., 1::2] << 4), "_scale": scale[..., 0],
            "_amax": amax, "_outer_scale_denominator": denominator}


def mtp_tensors(m: dict, vocab: int) -> dict[str, torch.Tensor]:
    """A random DSpark draft under mtp.safetensors' names."""
    H, D, R, I = m["hidden_size"], m["head_dim"], m["markov_rank"], m["intermediate_size"]
    q, kv = m["num_attention_heads"] * D, m["num_key_value_heads"] * D
    shapes = {"fuse_weight": (m["num_hidden_layers"] + 1, len(m["target_layer_ids"])), "mask_embed": (H,),
              "norm.weight": (H,), "markov_head.dflare_norm.weight": (H,),
              "markov_head.hidden_out.weight": (H, R), "markov_head.hidden_out.bias": (H,),
              "markov_head.markov_w1.weight": (vocab, R),
              "markov_head.rnn_proj.weight": (3 * R, 2 * R + 2 * H), "markov_head.rnn_proj.bias": (3 * R,)}
    for i in range(m["num_hidden_layers"]):
        p = f"layers.{i}."
        shapes.update({p + "input_layernorm.weight": (H,), p + "post_attention_layernorm.weight": (H,),
                       p + "dflare_norm.weight": (H,), p + "self_attn.q_proj.weight": (q, H),
                       p + "self_attn.k_proj.weight": (kv, H), p + "self_attn.v_proj.weight": (kv, H),
                       p + "self_attn.k_proj_ctx.weight": (kv, H), p + "self_attn.v_proj_ctx.weight": (kv, H),
                       p + "self_attn.o_proj.weight": (H, q), p + "self_attn.q_norm.weight": (D,),
                       p + "self_attn.k_norm.weight": (D,), p + "self_attn.k_norm_ctx.weight": (D,),
                       p + "mlp.gate_proj.weight": (I, H), p + "mlp.up_proj.weight": (I, H),
                       p + "mlp.down_proj.weight": (H, I)})
    out = {}
    for k, s in shapes.items():
        if "norm" in k:
            out[k] = 1 + 0.1 * torch.randn(s)
        else:
            out[k] = torch.randn(s) * (s[-1] ** -0.5 if len(s) > 1 else 0.5)
    return {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True, help="a real snapshot: config and tokenizer files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.like), Path(args.out)
    raw = json.load(open(src / "config.json"))
    t = raw["text_config"]
    t.update(hidden_size=64, num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
             swa_num_attention_heads=2, swa_num_key_value_heads=1, swa_head_dim=32, n_routed_experts=8,
             num_experts_per_tok=2, intermediate_size=64, n_layers_per_attn_res_block=2,
             local_layer_ids=[0, 2, 3, 4], sliding_window_size=64, rel_extent=64)
    # transformers takes the layer types from layer_types (default: every 6th full), mlx-lm from local_layer_ids.
    t["layer_types"] = ["hybrid_sliding" if i in t["local_layer_ids"] else "hybrid" for i in range(6)]
    if t.get("loop_config"):
        t["loop_config"].update(core_start=2, core_end=4, gate_rank=16)
    if raw.get("vision_config"):
        raw["vision_config"].update(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                    projection_hidden_dim=64, text_hidden_size=64)
    if raw.get("mtp_config"):
        # 6 layers, 2-3 looped 3 times: 10 layer visits, 10 is the output.
        raw["mtp_config"].update(hidden_size=64, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                                 intermediate_size=128, markov_rank=64, sliding_window_size=16,
                                 target_layer_ids=[1, 4, 7, 10])
    quant = raw.pop("quantization_config", None)
    torch.manual_seed(args.seed)
    config = AutoConfig.for_model(**raw)
    model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            # Zero-initialized queries, scales and conv weights would hide layout mistakes.
            if p.dtype.is_floating_point and "norm" not in name and "layernorm" not in name:
                p.normal_(0.0, 0.5 if p.ndim == 1 or "query" in name else 0.05)
    tmp = out.with_name(out.name + ".save")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    tensors = {}
    for f in sorted(tmp.glob("*.safetensors")):
        tensors.update(load_file(f))
    if not any(k.startswith("model.llm.") for k in tensors):
        raise SystemExit(f"save_pretrained did not write the published names: {sorted(tensors)[:5]}")
    if quant:
        for k in [k for k in tensors if k.endswith(("experts.w13_weight", "experts.w2_weight"))]:
            for suffix, v in nvfp4(tensors.pop(k)).items():
                tensors[k + suffix] = v.contiguous()
        raw["quantization_config"] = quant
    out.mkdir(parents=True, exist_ok=True)
    if raw.get("mtp_config"):
        save_file(mtp_tensors(raw["mtp_config"], t["vocab_size"]), out / "mtp.safetensors", metadata={"format": "pt"})
    save_file(tensors, out / "model-00001-of-00001.safetensors", metadata={"format": "pt"})
    json.dump({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}},
              open(out / "model.safetensors.index.json", "w"), indent=2)
    json.dump(raw, open(out / "config.json", "w"), indent=2)
    for f in src.iterdir():
        if f.is_file() and f.name not in ("config.json", "model.safetensors.index.json", "README.md") \
                and not f.name.endswith(".safetensors"):
            shutil.copyfile(f, out / f.name)
    shutil.rmtree(tmp)
    print(f"TINY_DONE {out} ({len(tensors)} tensors)", flush=True)


if __name__ == "__main__":
    main()
