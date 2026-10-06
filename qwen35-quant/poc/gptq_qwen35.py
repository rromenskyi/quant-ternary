"""GPTQ calibration of a Qwen3.5 checkpoint's text decoder -- dense
(`qwen3_5`, e.g. FrogNano-4B) or mixture-of-experts (`qwen3_5_moe`, e.g.
Ornith-1.5-35B-A3B) -- with a per-component bit recipe ("JANG": bits by
component type, uniform across layers), on MLX's affine grid.

Layer by layer: the model stays in CPU memory and only the layer being
calibrated is on the GPU, so a 72 GB bf16 MoE fits an 80 GB card. The
calibration activations go from layer to layer (layer i is calibrated on
the outputs of the already-quantized layers 0..i-1) instead of re-running
the model from the start for every layer.

Writes, under --work:
  layers/NNN.safetensors  layer NNN's quantized weights, bf16 on the grid,
                          under their checkpoint names (resume unit)
  hidden.pt, progress.json  the activations after the last finished layer
  quant_recipe.json       the recipe, for assemble_checkpoint.py and
                          convert_mlx.py

Components (module suffix within a decoder layer -> recipe key):
  self_attn.{q,k,v,o}_proj                        attn
  linear_attn.{in_proj_qkv,in_proj_z,out_proj}    linear
  mlp.{gate,up,down}_proj                         mlp      (dense models)
  mlp.shared_expert.{gate,up,down}_proj           shared   (MoE)
  mlp.experts.{gate_up_proj,down_proj}            experts  (MoE, fused [E, out, in])
Anything matching --keep-float stays bf16 (defaults: the delta rule's
in_proj_a / in_proj_b gates, the MoE router and the shared-expert gate).

Experts: a pre-hook on `mlp.experts` sees (hidden, top_k_index, ...)
whatever experts implementation runs; each expert is calibrated on the
tokens routed to it (batched GPTQ, --expert-batch experts at a time).
down_proj's inputs are recomputed from the quantized gate_up_proj, as at
inference.

    python gptq_qwen35.py --model SNAPSHOT --work /workspace/ornith-gptq \\
        --recipe attn=8,linear=6,shared=6,experts=3 \\
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
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nemotron-extreme-quant" / "poc"))
from gptq import gptq_nbit, gptq_nbit_batched  # noqa: E402  (MLX-exact affine grid)

COMPONENTS = {
    "self_attn.q_proj": "attn", "self_attn.k_proj": "attn", "self_attn.v_proj": "attn", "self_attn.o_proj": "attn",
    "linear_attn.in_proj_qkv": "linear", "linear_attn.in_proj_z": "linear", "linear_attn.out_proj": "linear",
    "mlp.gate_proj": "mlp", "mlp.up_proj": "mlp", "mlp.down_proj": "mlp",
    "mlp.shared_expert.gate_proj": "shared", "mlp.shared_expert.up_proj": "shared", "mlp.shared_expert.down_proj": "shared",
    "mlp.experts.gate_up_proj": "experts_gate_up", "mlp.experts.down_proj": "experts_down",
}
DEFAULT_KEEP_FLOAT = [r"linear_attn\.in_proj_[ab]$", r"mlp\.gate$", r"mlp\.shared_expert_gate$"]
PREFIX = "model.language_model.layers"
DEVICE = "cuda"


def parse_recipe(text: str) -> dict[str, int]:
    return {k.strip(): int(v) for k, v in (p.split("=") for p in text.split(","))}


def bits_for(component: str, recipe: dict[str, int]) -> int | None:
    """A component's bits; "experts_down" etc. fall back to "experts"."""
    return recipe.get(component, recipe.get(component.split("_")[0]))


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
    passes it (masks and rope differ by layer type) and changes nothing."""

    def __init__(self, store: dict, index: int):
        super().__init__()
        self.store, self.index = store, index

    def forward(self, hidden_states, **kwargs):
        self.store.setdefault(self.index, kwargs)
        return hidden_states


def layer_kwargs(lm, chunk: torch.Tensor) -> tuple[torch.Tensor, dict[int, dict]]:
    """The embeddings of `chunk` and each layer's call kwargs, from one pass
    with every layer swapped for a _Catch (chunks are all one length, so
    one pass serves them all)."""
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


def keep_float(name: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(name) for p in patterns)


def batched_gptq(W: torch.Tensor, Xp: torch.Tensor, bits: int, group_size: int, percdamp: float) -> dict:
    """gptq_nbit_batched with more damping when a near-empty expert's
    Hessian won't factor."""
    for attempt in range(6):
        try:
            return gptq_nbit_batched(W=W, Xp=Xp, bits=bits, group_size=group_size, percdamp=percdamp * 10**attempt,
                                     device=DEVICE, scheme="affine")
        except torch._C._LinAlgError:
            if attempt == 5:
                raise


def expert_xp(rows: dict[int, torch.Tensor], experts: list[int], dim: int) -> torch.Tensor:
    n = max([rows[e].shape[0] for e in experts if e in rows] + [1])
    Xp = torch.zeros(len(experts), n, dim, dtype=torch.float32, device=DEVICE)
    for j, e in enumerate(experts):
        if e in rows:
            Xp[j, : rows[e].shape[0]] = rows[e]
    return Xp


def calibrate_experts(experts, rows: dict[int, torch.Tensor], recipe: dict[str, int], args) -> dict[str, torch.Tensor]:
    E, inter = experts.num_experts, experts.intermediate_dim
    gate_up = experts.gate_up_proj.data
    down = experts.down_proj.data
    gu_hat = torch.empty_like(gate_up, device="cpu")
    dn_hat = torch.empty_like(down, device="cpu")
    groups = lambda W: (W.shape[0], W.shape[1], W.shape[2] // args.group_size)  # noqa: E731
    gu_s, gu_b = torch.empty(groups(gate_up)), torch.empty(groups(gate_up))
    dn_s, dn_b = torch.empty(groups(down)), torch.empty(groups(down))
    for start in range(0, E, args.expert_batch):
        ids = list(range(start, min(E, start + args.expert_batch)))
        Xp = expert_xp(rows, ids, gate_up.shape[-1])
        res = batched_gptq(gate_up[ids].float(), Xp, bits_for("experts_gate_up", recipe), args.group_size, args.percdamp)
        W = res["W_hat"]
        gu_hat[ids] = W.to(torch.bfloat16).cpu()
        gu_s[ids], gu_b[ids] = res["scale"].float().cpu(), res["bias"].float().cpu()
        # down_proj sees act(gate) * up of the QUANTIZED gate_up, as at inference.
        down_rows = {}
        for j, e in enumerate(ids):
            if e in rows:
                h = rows[e].float() @ W[j].to(DEVICE).t()
                down_rows[e] = experts.act_fn(h[:, :inter]) * h[:, inter:]
        res = batched_gptq(down[ids].float(), expert_xp(down_rows, ids, inter), bits_for("experts_down", recipe),
                           args.group_size, args.percdamp)
        dn_hat[ids] = res["W_hat"].to(torch.bfloat16).cpu()
        dn_s[ids], dn_b[ids] = res["scale"].float().cpu(), res["bias"].float().cpu()
    experts.gate_up_proj.data.copy_(gu_hat.to(gate_up.device))
    experts.down_proj.data.copy_(dn_hat.to(down.device))
    # The GPTQ grid itself (scale per code, bias): convert_mlx.py writes these
    # codes exactly instead of letting MLX re-derive the grid from min / max.
    return {"mlp.experts.gate_up_proj": gu_hat, "mlp.experts.down_proj": dn_hat,
            "mlp.experts.gate_up_proj.gptq_scales": gu_s, "mlp.experts.gate_up_proj.gptq_biases": gu_b,
            "mlp.experts.down_proj.gptq_scales": dn_s, "mlp.experts.down_proj.gptq_biases": dn_b}


def calibrate_layer(layer, hidden: list[torch.Tensor], kwargs: dict, recipe: dict, floats, args) -> dict[str, torch.Tensor]:
    targets = {n: m for n, m in layer.named_modules()
               if n in COMPONENTS and isinstance(m, torch.nn.Linear) and not keep_float(n, floats)}
    # Inputs stay on the GPU (bf16): per-chunk copies to the CPU and per-expert
    # gathers in the hook made a layer CPU-bound (21 min on an A100).
    inputs: dict[str, list[torch.Tensor]] = {n: [] for n in targets}
    hooks = [m.register_forward_hook(lambda m, a, o, n=n: inputs[n].append(a[0].detach().reshape(-1, a[0].shape[-1])))
             for n, m in targets.items()]
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    if experts is not None and not keep_float("mlp.experts", floats):
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
        # (X moved to the GPU and cast inside gptq_nbit)
        bits = bits_for(COMPONENTS[n], recipe)
        res = gptq_nbit(m.weight.data.float(), torch.cat(inputs.pop(n)), bits=bits, group_size=args.group_size,
                        percdamp=args.percdamp, device=DEVICE, scheme="affine")
        m.weight.data.copy_(res["W_hat"].to(m.weight.dtype).to(m.weight.device))
        out[f"{n}.weight"] = res["W_hat"].to(torch.bfloat16).cpu()
        out[f"{n}.weight.gptq_scales"] = res["scale"].float().cpu()
        out[f"{n}.weight.gptq_biases"] = res["bias"].float().cpu()
    t2 = time.time()
    if rows:
        out.update(calibrate_experts(experts, rows, recipe, args))
    print(f"  capture {t1 - t0:.0f}s, linears {t2 - t1:.0f}s, experts {time.time() - t2:.0f}s "
          f"({len(rows)} experts hit)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="bf16 checkpoint (HF snapshot dir)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--recipe", required=True, help="e.g. attn=8,linear=6,shared=6,experts=3 (or mlp=4 for dense)")
    ap.add_argument("--calib", action="append", required=True, help="name:path of a text file; repeat to mix")
    ap.add_argument("--keep-float", action="append", default=None, help="regex on module paths within a layer")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--calib-chunks", type=int, default=128)
    ap.add_argument("--calib-chunk-tokens", type=int, default=512)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--expert-batch", type=int, default=32)
    ap.add_argument("--max-rows-per-expert", type=int, default=4096)
    ap.add_argument("--device", default="cuda", help="cpu: a dry run on a tiny model")
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device
    recipe = parse_recipe(args.recipe)
    floats = [re.compile(p) for p in (args.keep_float or DEFAULT_KEEP_FLOAT)]
    work = Path(args.work)
    (work / "layers").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).eval()  # CPU memory
    lm = model.model.language_model
    layers = lm.layers
    present = {COMPONENTS[n] for layer in layers for n, _ in layer.named_modules() if n in COMPONENTS}
    if any(getattr(getattr(layer, "mlp", None), "experts", None) is not None for layer in layers):
        present |= {"experts_gate_up", "experts_down"}
    missing = {c for c in present if bits_for(c, recipe) is None}
    if missing:
        raise SystemExit(f"recipe has no bits for: {sorted(missing)} (model has {sorted(present)})")
    json.dump({"recipe": {k: bits_for(k, recipe) for k in sorted(present)}, "group_size": args.group_size,
               "components": COMPONENTS, "keep_float": [p.pattern for p in floats],
               "calibration": {"chunks": args.calib_chunks, "tokens": args.calib_chunk_tokens, "sources": args.calib}},
              open(work / "quant_recipe.json", "w"), indent=2)

    chunks = load_chunks(args.calib, tok, args.calib_chunks, args.calib_chunk_tokens)
    print(f"{len(chunks)} chunks x {args.calib_chunk_tokens} tokens; recipe {recipe}; {len(layers)} layers", flush=True)
    kwargs = layer_kwargs(lm, chunks[0])

    progress = work / "progress.json"
    start = json.load(open(progress))["done"] + 1 if progress.exists() else 0
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
        out = calibrate_layer(layer, hidden, kwargs[i], recipe, floats, args)
        with torch.no_grad():
            hidden = [layer(h, **kwargs[i]) for h in hidden]
        save_file({f"{PREFIX}.{i}.{k}": v.contiguous() for k, v in out.items()}, work / "layers" / f"{i:03d}.safetensors")
        torch.save([h.cpu() for h in hidden], work / "hidden.pt")
        json.dump({"done": i, "layers": len(layers)}, open(progress, "w"))
        layers[i] = layer.cpu()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        n = sum(1 for k in out if not k.endswith(("gptq_scales", "gptq_biases")))
        print(f"layer {i:2d}/{len(layers)} ({n} tensors) {time.time() - started:.0f}s", flush=True)
    print("GPTQ_DONE", flush=True)


if __name__ == "__main__":
    main()
