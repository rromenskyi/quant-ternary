"""GPTQ calibration of Meta's Muse Glimmer 30B text decoder (dense,
`muse_glimmer`) with a per-component bit recipe ("JANG": bits by component
type, uniform across layers), on MLX's affine grid.

Layer by layer: the model stays in CPU memory and only the layer being
calibrated is on the GPU (gptq_qwen35.py's approach), so this fits a modest
single GPU despite the 30B size -- the per-layer working set (hidden 6656,
intermediate 19968) is small.

Glimmer is dense and single-tower for text (unlike qwen3.5's dense-or-MoE,
vision-capable checkpoints): no MoE experts, no linear (delta-rule)
attention, no MTP head. Its one twist is the attention block's extra output
gate (`self_attn.gate_proj`, sigmoid-gated before `o_proj`), folded into the
"attn" component below -- it's attention-sized, not MLP-sized, and there is
no reason to budget it separately.

Components (module suffix within a decoder layer -> recipe key):
  self_attn.{q,k,v,o,gate}_proj   attn
  mlp.{gate,up,down}_proj         mlp

Writes, under --work:
  layers/NNN.safetensors  layer NNN's quantized weights, bf16 on the grid,
                          under their checkpoint names (resume unit)
  hidden.pt, progress.json  the activations after the last finished layer
  quant_recipe.json       the recipe, for assemble_checkpoint.py and
                          convert_muse_glimmer.py

    python gptq_muse_glimmer.py --model SNAPSHOT --work /workspace/glimmer-gptq \\
        --recipe attn=8,mlp=4 \\
        --calib wikitext:/workspace/data/wiki.train.raw --calib code:/workspace/data/code.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nemotron-extreme-quant" / "poc"))
from gptq import gptq_nbit  # noqa: E402  (MLX-exact affine grid)

COMPONENTS = {
    "self_attn.q_proj": "attn", "self_attn.k_proj": "attn", "self_attn.v_proj": "attn",
    "self_attn.o_proj": "attn", "self_attn.gate_proj": "attn",
    "mlp.gate_proj": "mlp", "mlp.up_proj": "mlp", "mlp.down_proj": "mlp",
}
EXPECTED = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "self_attn.gate_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
PREFIX = "model.language_model.layers"
DEVICE = "cuda"


def parse_recipe(text: str) -> dict[str, int]:
    return {k.strip(): int(v) for k, v in (p.split("=") for p in text.split(","))}


def codes(W: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """GPTQ's integer codes from its float32 result (uint8): convert_muse_glimmer.py
    packs these, not codes re-derived from the bf16 weights (at 8 bits a
    value near a group's edge can round across a code in bf16)."""
    W = W.float().reshape(*W.shape[:-1], -1, group_size)
    s = scale.float()[..., None]
    q = torch.round((W - bias.float()[..., None]) / torch.where(s == 0, torch.ones_like(s), s))
    return q.clamp(0, 2**bits - 1).to(torch.uint8).reshape(*W.shape[:-2], -1)


def load_chunks(sources: list[str], tok, n: int, tokens: int) -> list[torch.Tensor]:
    """n chunks of exactly `tokens` tokens, spread evenly over each source
    (name:path), the sources taking turns."""
    texts = [open(s.split(":", 1)[1], encoding="utf-8", errors="ignore").read() for s in sources]
    per = -(-n // len(texts))
    chunks = []
    for text in texts:
        stride = max(1, len(text) // per)
        for i in range(per):
            ids = tok(text[i * stride: i * stride + tokens * 8], return_tensors="pt",
                      truncation=True, max_length=tokens)["input_ids"]
            if ids.shape[1] == tokens:
                chunks.append(ids)
    return chunks[:n]


class _Catch(torch.nn.Module):
    """Stands in for a decoder layer: records the kwargs the text model
    passes it (attention_mask differs local/global, position_embeddings is
    None on NoPE/global layers) and changes nothing."""

    def __init__(self, store: dict, index: int):
        super().__init__()
        self.store, self.index = store, index

    def forward(self, hidden_states, **kwargs):
        self.store.setdefault(self.index, kwargs)
        return hidden_states


def layer_kwargs(lm, chunk: torch.Tensor) -> dict[int, dict]:
    """Each layer's call kwargs, from one pass with every layer swapped for
    a _Catch (chunks are all one length, so one pass serves them all)."""
    store: dict[int, dict] = {}
    real = lm.layers
    lm.layers = torch.nn.ModuleList([_Catch(store, i) for i in range(len(real))])
    for m in (lm.embed_tokens, lm.rotary_emb, lm.norm):
        m.to(DEVICE)
    try:
        with torch.no_grad():
            lm(input_ids=chunk.to(DEVICE), use_cache=False)
    finally:
        lm.layers = real
        for m in (lm.embed_tokens, lm.rotary_emb, lm.norm):
            m.cpu()
    return store


def calibrate_layer(layer, hidden: list[torch.Tensor], kwargs: dict, recipe: dict, args) -> dict[str, torch.Tensor]:
    """GPTQ of one decoder layer on its inputs `hidden`."""
    targets = {n: m for n, m in layer.named_modules() if n in COMPONENTS and isinstance(m, torch.nn.Linear)}
    missing = {e for e in EXPECTED if e not in targets}
    if missing:
        has = sorted(n for n, _ in layer.named_modules())
        raise SystemExit(f"missing {sorted(missing)} (layer has {has[:20]})")
    inputs: dict[str, list[torch.Tensor]] = {n: [] for n in targets}
    hooks = [m.register_forward_hook(lambda m, a, o, n=n: inputs[n].append(a[0].detach().reshape(-1, a[0].shape[-1])))
             for n, m in targets.items()]
    t0 = time.time()
    with torch.no_grad():
        for h in hidden:
            layer(h, **kwargs)
    for h in hooks:
        h.remove()
    t1 = time.time()

    out: dict[str, torch.Tensor] = {}
    for n, m in targets.items():
        bits = recipe[COMPONENTS[n]]
        res = gptq_nbit(m.weight.data.float(), torch.cat(inputs.pop(n)), bits=bits, group_size=args.group_size,
                        percdamp=args.percdamp, device=DEVICE, scheme="affine")
        m.weight.data.copy_(res["W_hat"].to(m.weight.dtype).to(m.weight.device))
        out[f"{n}.weight"] = res["W_hat"].to(torch.bfloat16).cpu()
        out[f"{n}.weight.gptq_scales"] = res["scale"].float().cpu()
        out[f"{n}.weight.gptq_biases"] = res["bias"].float().cpu()
        out[f"{n}.weight.gptq_codes"] = codes(res["W_hat"], res["scale"], res["bias"], args.group_size, bits).cpu()
    print(f"  capture {t1 - t0:.0f}s, linears {time.time() - t1:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="bf16 checkpoint (HF snapshot dir)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--recipe", required=True, help="e.g. attn=8,mlp=4")
    ap.add_argument("--calib", action="append", required=True, help="name:path of a text file; repeat to mix")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--calib-chunks", type=int, default=128)
    ap.add_argument("--calib-chunk-tokens", type=int, default=512)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--device", default="cuda", help="cpu: a dry run on a tiny model")
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device
    recipe = parse_recipe(args.recipe)
    missing = {c for c in ("attn", "mlp") if c not in recipe}
    if missing:
        raise SystemExit(f"recipe has no bits for: {sorted(missing)}")
    work = Path(args.work)
    (work / "layers").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).eval()  # CPU memory
    lm = model.model.language_model
    layers = lm.layers

    chunks = load_chunks(args.calib, tok, args.calib_chunks, args.calib_chunk_tokens)
    if not chunks:
        raise SystemExit(f"no calibration chunk of {args.calib_chunk_tokens} tokens in {args.calib}")
    print(f"{len(chunks)} chunks x {args.calib_chunk_tokens} tokens; recipe {recipe}; {len(layers)} layers", flush=True)
    kwargs = layer_kwargs(lm, chunks[0])

    progress = work / "progress.json"
    made_with = {"recipe": args.recipe, "group_size": args.group_size, "calib": args.calib,
                 "calib_chunks": args.calib_chunks, "calib_chunk_tokens": args.calib_chunk_tokens}
    start = 0
    if progress.exists():
        old = json.load(open(progress))
        if old.get("made_with") != made_with:
            raise SystemExit(f"{work} holds layers made with {old.get('made_with')}, not {made_with}: "
                             "use another --work or delete it")
        start = old["done"] + 1
    json.dump({"recipe": recipe, "group_size": args.group_size, "components": COMPONENTS,
               "calibration": {"chunks": args.calib_chunks, "tokens": args.calib_chunk_tokens, "sources": args.calib}},
              open(work / "quant_recipe.json", "w"), indent=2)
    if start:
        hidden = [h.to(DEVICE) for h in torch.load(work / "hidden.pt")]
        print(f"resuming at layer {start}", flush=True)
    else:
        lm.embed_tokens.to(DEVICE)
        with torch.no_grad():
            hidden = [lm.embed_tokens(c.to(DEVICE)) for c in chunks]
        lm.embed_tokens.cpu()

    started = time.time()
    for i in range(start, len(layers)):
        layer = layers[i].to(DEVICE)
        out = calibrate_layer(layer, hidden, kwargs[i], recipe, args)
        with torch.no_grad():
            hidden = [layer(h, **kwargs[i]) for h in hidden]
        save_file({f"{PREFIX}.{i}.{k}": v.contiguous() for k, v in out.items()}, work / "layers" / f"{i:03d}.safetensors")
        torch.save([h.cpu() for h in hidden], work / "hidden.pt")
        json.dump({"done": i, "layers": len(layers), "made_with": made_with}, open(progress, "w"))
        layers[i] = layer.cpu()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        n = sum(1 for k in out if ".gptq_" not in k)
        print(f"layer {i:2d}/{len(layers)} ({n} tensors) {time.time() - started:.0f}s", flush=True)
    print("GPTQ_DONE", flush=True)


if __name__ == "__main__":
    main()
