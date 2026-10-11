"""GPTQ calibration of an LFM2 / LFM2.5 checkpoint (Liquid AI) -- dense
(`lfm2`, e.g. LFM2.5-1.2B, 2.6B) or mixture-of-experts (`lfm2_moe`,
LFM2.5-8B-A1B) -- with a per-component bit recipe ("JANG"), on MLX's
affine grid. Same method as qwen35-quant/poc/gptq_qwen35.py (whose helpers
it uses): layer by layer, the model in CPU memory, one layer on the GPU,
activations carried from the quantized layers before.

Components (module suffix within a decoder layer -> recipe key):
  self_attn.{q,k,v,out}_proj                attn     (6 of 24 layers)
  conv.{in,out}_proj                        conv     (gated short convolution)
  feed_forward.{w1,w3,w2}                   mlp      (dense layers)
  feed_forward.experts.{gate_up,down}_proj  experts_gate_up / experts_down
Not quantized: the depthwise conv kernel (conv.conv, 3 taps), the norms,
the MoE router (feed_forward.gate) and its expert_bias.

Writes, under --work: layers/NNN.safetensors (HF module names; fused
experts as feed_forward.experts.{gate_up,down}_proj, which
assemble_lfm2.py splits into the checkpoint's per-expert w1/w3/w2),
hidden.pt + progress.json (resume), quant_recipe.json.

    python gptq_lfm2.py --model SNAPSHOT --work /workspace/gptq-lfm2-8b \\
        --recipe attn=8,conv=6,mlp=6,experts=3 \\
        --calib wikitext:/workspace/data/wiki.train.raw --calib code:/workspace/data/code.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
import transformers
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen35-quant" / "poc"))
import gptq_qwen35 as q  # noqa: E402
from gptq_qwen35 import batched_gptq, bits_for, codes, expert_xp, load_chunks, parse_recipe  # noqa: E402
from gptq import gptq_nbit  # noqa: E402  (on sys.path through gptq_qwen35)

COMPONENTS = {
    "self_attn.q_proj": "attn", "self_attn.k_proj": "attn", "self_attn.v_proj": "attn", "self_attn.out_proj": "attn",
    "conv.in_proj": "conv", "conv.out_proj": "conv",
    "feed_forward.w1": "mlp", "feed_forward.w3": "mlp", "feed_forward.w2": "mlp",
    "feed_forward.experts.gate_up_proj": "experts_gate_up", "feed_forward.experts.down_proj": "experts_down",
}
EXPECTED = {
    "self_attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"],
    "conv": ["conv.in_proj", "conv.out_proj"],
}


def calibrate_experts(experts, rows, recipe, args) -> dict[str, torch.Tensor]:
    """gptq_qwen35's expert calibration, under this model's names."""
    out = q.calibrate_experts(experts, rows, recipe, args)
    return {k.replace("mlp.experts.", "feed_forward.experts."): v for k, v in out.items()}


def calibrate_layer(layer, hidden, kwargs, recipe, args) -> dict[str, torch.Tensor]:
    targets = {n: m for n, m in layer.named_modules() if n in COMPONENTS and isinstance(m, torch.nn.Linear)}
    names = {n for n, _ in layer.named_modules()}
    # A renamed module would otherwise stay bf16 without a word.
    if not {"self_attn", "conv"} & names:
        raise SystemExit(f"no self_attn or conv block in the layer: {sorted(names)[:12]}")
    for block, expected in EXPECTED.items():
        if block in names and set(expected) - set(targets):
            raise SystemExit(f"{block}: no {sorted(set(expected) - set(targets))}")
    experts = getattr(layer.feed_forward, "experts", None)
    if experts is None and not {"feed_forward.w1", "feed_forward.w2", "feed_forward.w3"} <= set(targets):
        raise SystemExit(f"feed_forward is neither experts nor w1/w2/w3: {sorted(names)}")

    inputs: dict[str, list[torch.Tensor]] = {n: [] for n in targets}
    hooks = [m.register_forward_hook(lambda m, a, o, n=n: inputs[n].append(a[0].detach().reshape(-1, a[0].shape[-1])))
             for n, m in targets.items()]
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    if experts is not None:
        hooks.append(experts.register_forward_pre_hook(lambda _m, a: seen.append((a[0].detach(), a[1].detach()))))
    t0 = time.time()
    with torch.no_grad():
        for h in hidden:
            layer(h, **kwargs)
    for h in hooks:
        h.remove()
    rows: dict[int, torch.Tensor] = {}
    if seen:
        x, top = torch.cat([a for a, _ in seen]), torch.cat([b for _, b in seen])
        seen.clear()
        for e in torch.unique(top).tolist():
            rows[e] = x[(top == e).any(dim=-1)][: args.max_rows_per_expert]
        del x, top
    t1 = time.time()

    out: dict[str, torch.Tensor] = {}
    for n, m in targets.items():
        bits = bits_for(COMPONENTS[n], recipe)
        res = gptq_nbit(m.weight.data.float(), torch.cat(inputs.pop(n)), bits=bits, group_size=args.group_size,
                        percdamp=args.percdamp, device=q.DEVICE, scheme="affine")
        m.weight.data.copy_(res["W_hat"].to(m.weight.dtype).to(m.weight.device))
        out[f"{n}.weight"] = res["W_hat"].to(torch.bfloat16).cpu()
        out[f"{n}.weight.gptq_scales"] = res["scale"].float().cpu()
        out[f"{n}.weight.gptq_biases"] = res["bias"].float().cpu()
        out[f"{n}.weight.gptq_codes"] = codes(res["W_hat"], res["scale"], res["bias"], args.group_size, bits).cpu()
    t2 = time.time()
    if experts is not None:
        out.update(calibrate_experts(experts, rows, recipe, args))
    print(f"  capture {t1 - t0:.0f}s, linears {t2 - t1:.0f}s, experts {time.time() - t2:.0f}s "
          f"({len(rows)} experts hit, fewest rows {min((r.shape[0] for r in rows.values()), default=0)})", flush=True)
    return out


def layer_kwargs(lm, chunk: torch.Tensor) -> dict[int, dict]:
    """Each layer's call kwargs (mask by layer type, rope), from one pass
    with every layer swapped for gptq_qwen35's _Catch."""
    store: dict[int, dict] = {}
    real = lm.layers
    lm.layers = torch.nn.ModuleList([q._Catch(store, i) for i in range(len(real))])
    rope = getattr(lm, "pos_emb", None) or lm.rotary_emb  # lfm2_moe / lfm2
    small = (lm.embed_tokens, rope, lm.embedding_norm)
    for m in small:
        m.to(q.DEVICE)
    try:
        with torch.no_grad():
            lm(input_ids=chunk.to(q.DEVICE), use_cache=False)
    finally:
        lm.layers = real
        for m in small:
            m.cpu()
    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="bf16 checkpoint (HF snapshot dir)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--recipe", required=True, help="e.g. attn=8,conv=6,mlp=6,experts=3")
    ap.add_argument("--calib", action="append", required=True, help="name:path of a text file; repeat to mix")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--calib-chunks", type=int, default=128)
    ap.add_argument("--calib-chunk-tokens", type=int, default=512)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--expert-batch", type=int, default=32)
    ap.add_argument("--max-rows-per-expert", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--auto-class", default="AutoModelForCausalLM",
                    help="transformers class to load with (AutoModelForImageTextToText for lfm2_vl)")
    ap.add_argument("--text-module", default="model",
                    help="attribute path of the decoder (model.language_model for lfm2_vl)")
    ap.add_argument("--prefix", default="model.layers",
                    help="checkpoint name prefix of the decoder layers (model.language_model.layers for lfm2_vl)")
    args = ap.parse_args()
    q.DEVICE = args.device
    recipe = parse_recipe(args.recipe)
    work = Path(args.work)
    (work / "layers").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    auto = getattr(transformers, args.auto_class)
    model = auto.from_pretrained(args.model, dtype=torch.bfloat16).eval()  # CPU memory
    lm = model
    for name in args.text_module.split("."):
        lm = getattr(lm, name)
    layers = lm.layers
    present = {COMPONENTS[n] for layer in layers for n, _ in layer.named_modules() if n in COMPONENTS}
    if any(getattr(layer.feed_forward, "experts", None) is not None for layer in layers):
        present |= {"experts_gate_up", "experts_down"}
    missing = {c for c in present if bits_for(c, recipe) is None}
    if missing:
        raise SystemExit(f"recipe has no bits for: {sorted(missing)} (model has {sorted(present)})")

    chunks = load_chunks(args.calib, tok, args.calib_chunks, args.calib_chunk_tokens)
    if not chunks:
        raise SystemExit(f"no calibration chunk of {args.calib_chunk_tokens} tokens in {args.calib}")
    print(f"{len(chunks)} chunks x {args.calib_chunk_tokens} tokens; recipe {recipe}; {len(layers)} layers", flush=True)
    kwargs = layer_kwargs(lm, chunks[0])

    progress = work / "progress.json"
    made_with = {"recipe": args.recipe, "group_size": args.group_size, "calib": args.calib,
                 "calib_chunks": args.calib_chunks, "calib_chunk_tokens": args.calib_chunk_tokens,
                 "max_rows_per_expert": args.max_rows_per_expert, "prefix": args.prefix}
    start = 0
    if progress.exists():
        old = json.load(open(progress))
        if old.get("made_with") != made_with:
            raise SystemExit(f"{work} holds layers made with {old.get('made_with')}, not {made_with}: "
                             "use another --work or delete it")
        start = old["done"] + 1
    json.dump({"recipe": {k: bits_for(k, recipe) for k in sorted(present)}, "group_size": args.group_size,
               "components": COMPONENTS, "keep_float": [],
               "calibration": {"chunks": args.calib_chunks, "tokens": args.calib_chunk_tokens, "sources": args.calib}},
              open(work / "quant_recipe.json", "w"), indent=2)
    if start:
        hidden = [h.to(q.DEVICE) for h in torch.load(work / "hidden.pt")]
        print(f"resuming at layer {start}", flush=True)
    else:
        lm.embed_tokens.to(q.DEVICE)
        with torch.no_grad():
            hidden = [lm.embed_tokens(c.to(q.DEVICE)) for c in chunks]
        lm.embed_tokens.cpu()

    started = time.time()
    for i in range(start, len(layers)):
        layer = layers[i].to(q.DEVICE)
        out = calibrate_layer(layer, hidden, kwargs[i], recipe, args)
        with torch.no_grad():
            hidden = [layer(h, **kwargs[i]) for h in hidden]
        save_file({f"{args.prefix}.{i}.{k}": v.contiguous() for k, v in out.items()}, work / "layers" / f"{i:03d}.safetensors")
        torch.save([h.cpu() for h in hidden], work / "hidden.pt")
        json.dump({"done": i, "layers": len(layers), "made_with": made_with}, open(progress, "w"))
        layers[i] = layer.cpu()
        if q.DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"layer {i:2d}/{len(layers)} {time.time() - started:.0f}s", flush=True)
    print("GPTQ_DONE", flush=True)


if __name__ == "__main__":
    main()
