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
  layers/mtp.safetensors  the MTP head, when the recipe has "mtp" bits
                          (its own recipe in the file's metadata)

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
from safetensors import safe_open
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
# Per block a decoder layer may have, the projections it must show.
EXPECTED = {
    "self_attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    "linear_attn": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"],
    "mlp.shared_expert": ["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"],
}
DEFAULT_KEEP_FLOAT = [r"linear_attn\.in_proj_[ab]$", r"mlp\.gate$", r"mlp\.shared_expert_gate$"]
PREFIX = "model.language_model.layers"
DEVICE = "cuda"


def parse_recipe(text: str) -> dict[str, int]:
    return {k.strip(): int(v) for k, v in (p.split("=") for p in text.split(","))}


def codes(W: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """GPTQ's integer codes from its float32 result (uint8): convert_mlx.py
    packs these, not codes re-derived from the bf16 weights (at 8 bits a
    value near a group's edge can round across a code in bf16)."""
    W = W.float().reshape(*W.shape[:-1], -1, group_size)
    s = scale.float()[..., None]
    q = torch.round((W - bias.float()[..., None]) / torch.where(s == 0, torch.ones_like(s), s))
    return q.clamp(0, 2**bits - 1).to(torch.uint8).reshape(*W.shape[:-2], -1)


def bits_for(component: str, recipe: dict[str, int]) -> int | None:
    """A component's bits, else its shorter prefixes' bits: "experts_down" falls
    back to "experts", "mtp_experts_down" to "mtp_experts", then "mtp"."""
    parts = component.split("_")
    for n in range(len(parts), 0, -1):
        if "_".join(parts[:n]) in recipe:
            return recipe["_".join(parts[:n])]
    return None


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


def calibrate_experts(experts, rows: dict[int, torch.Tensor], recipe: dict[str, int], args,
                      prefix: str = "") -> dict[str, torch.Tensor]:
    E, inter = experts.num_experts, experts.intermediate_dim
    gate_up = experts.gate_up_proj.data
    down = experts.down_proj.data
    gu_hat = torch.empty_like(gate_up, device="cpu")
    dn_hat = torch.empty_like(down, device="cpu")
    groups = lambda W: (W.shape[0], W.shape[1], W.shape[2] // args.group_size)  # noqa: E731
    gu_s, gu_b = torch.empty(groups(gate_up)), torch.empty(groups(gate_up))
    dn_s, dn_b = torch.empty(groups(down)), torch.empty(groups(down))
    gu_c = torch.empty(gate_up.shape, dtype=torch.uint8)
    dn_c = torch.empty(down.shape, dtype=torch.uint8)
    for start in range(0, E, args.expert_batch):
        ids = list(range(start, min(E, start + args.expert_batch)))
        Xp = expert_xp(rows, ids, gate_up.shape[-1])
        res = batched_gptq(gate_up[ids].float(), Xp, bits_for(prefix + "experts_gate_up", recipe), args.group_size, args.percdamp)
        W = res["W_hat"]
        gu_hat[ids] = W.to(torch.bfloat16).cpu()
        gu_s[ids], gu_b[ids] = res["scale"].float().cpu(), res["bias"].float().cpu()
        gu_c[ids] = codes(W.cpu(), gu_s[ids], gu_b[ids], args.group_size, bits_for(prefix + "experts_gate_up", recipe))
        # down_proj sees act(gate) * up of the QUANTIZED gate_up, as at inference.
        down_rows = {}
        for j, e in enumerate(ids):
            if e in rows:
                h = rows[e].float() @ W[j].to(DEVICE).t()
                down_rows[e] = experts.act_fn(h[:, :inter]) * h[:, inter:]
        res = batched_gptq(down[ids].float(), expert_xp(down_rows, ids, inter), bits_for(prefix + "experts_down", recipe),
                           args.group_size, args.percdamp)
        dn_hat[ids] = res["W_hat"].to(torch.bfloat16).cpu()
        dn_s[ids], dn_b[ids] = res["scale"].float().cpu(), res["bias"].float().cpu()
        dn_c[ids] = codes(res["W_hat"].cpu(), dn_s[ids], dn_b[ids], args.group_size, bits_for(prefix + "experts_down", recipe))
    experts.gate_up_proj.data.copy_(gu_hat.to(gate_up.device))
    experts.down_proj.data.copy_(dn_hat.to(down.device))
    # The GPTQ grid itself (scale per code, bias): convert_mlx.py writes these
    # codes exactly instead of letting MLX re-derive the grid from min / max.
    return {"mlp.experts.gate_up_proj": gu_hat, "mlp.experts.down_proj": dn_hat,
            "mlp.experts.gate_up_proj.gptq_scales": gu_s, "mlp.experts.gate_up_proj.gptq_biases": gu_b,
            "mlp.experts.gate_up_proj.gptq_codes": gu_c,
            "mlp.experts.down_proj.gptq_scales": dn_s, "mlp.experts.down_proj.gptq_biases": dn_b,
            "mlp.experts.down_proj.gptq_codes": dn_c}


def calibrate_layer(layer, hidden: list[torch.Tensor], kwargs: dict, recipe: dict, floats, args,
                    prefix: str = "") -> dict[str, torch.Tensor]:
    """GPTQ of one decoder layer on its inputs `hidden`; the recipe key of a
    component is `prefix` + its name (the MTP head's: "mtp_attn", ...)."""
    targets = {n: m for n, m in layer.named_modules()
               if n in COMPONENTS and isinstance(m, torch.nn.Linear) and not keep_float(n, floats)}
    # A renamed module would otherwise stay bf16 without a word: every block
    # the layer has must show all its projections.
    names = {n for n, _ in layer.named_modules()}
    if not {"self_attn", "linear_attn"} & names:
        raise SystemExit(f"no self_attn or linear_attn block in the layer: {sorted(names)[:12]}")
    for block, expected in EXPECTED.items():
        if block in names:
            missing = {e for e in expected if e not in targets and not keep_float(e, floats)}
            if missing:
                has = sorted(n for n in names if n.startswith(block + "."))
                raise SystemExit(f"{block}: no {sorted(missing)} (has {has})")
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
        bits = bits_for(prefix + COMPONENTS[n], recipe)
        res = gptq_nbit(m.weight.data.float(), torch.cat(inputs.pop(n)), bits=bits, group_size=args.group_size,
                        percdamp=args.percdamp, device=DEVICE, scheme="affine")
        m.weight.data.copy_(res["W_hat"].to(m.weight.dtype).to(m.weight.device))
        out[f"{n}.weight"] = res["W_hat"].to(torch.bfloat16).cpu()
        out[f"{n}.weight.gptq_scales"] = res["scale"].float().cpu()
        out[f"{n}.weight.gptq_biases"] = res["bias"].float().cpu()
        out[f"{n}.weight.gptq_codes"] = codes(res["W_hat"], res["scale"], res["bias"], args.group_size, bits).cpu()
    t2 = time.time()
    if rows:
        out.update(calibrate_experts(experts, rows, recipe, args, prefix))
    print(f"  capture {t1 - t0:.0f}s, linears {t2 - t1:.0f}s, experts {time.time() - t2:.0f}s "
          f"({len(rows)} experts hit)", flush=True)
    return out


def mtp_tensors(src: Path) -> dict[str, torch.Tensor]:
    """The checkpoint's MTP head tensors (mtp.*), from the shards that hold
    them; empty when it has none."""
    index = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for shard in sorted({f for k, f in index.items() if k.startswith("mtp.")}):
        with safe_open(src / shard, "pt") as s:
            out.update({k: s.get_tensor(k) for k in s.keys() if k.startswith("mtp.")})
    return out


def build_mtp(lm, weights: dict[str, torch.Tensor]) -> tuple[torch.nn.Module, int]:
    """The MTP head from its tensors, made of the text model's own classes:
    fc over [embedding, hidden] (each normed), then decoder layers like the
    backbone's full-attention ones, then a norm. Per-expert tensors are
    fused into the experts module's [E, 2I, H] / [E, H, I] layout.
    Returns the head and the index of the backbone layer its layers copy."""
    cfg = lm.config
    full = cfg.layer_types.index("full_attention")
    n = 1 + max(int(k.split(".")[2]) for k in weights if k.startswith("mtp.layers."))
    norm = lambda: type(lm.norm)(cfg.hidden_size, eps=cfg.rms_norm_eps)  # noqa: E731
    head = torch.nn.Module()
    head.pre_fc_norm_embedding, head.pre_fc_norm_hidden, head.norm = norm(), norm(), norm()
    head.fc = torch.nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
    head.layers = torch.nn.ModuleList([type(lm.layers[full])(cfg, full) for _ in range(n)])
    state, experts = {}, {}
    for k, v in weights.items():
        m = re.match(r"mtp\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$", k)
        if m:
            experts.setdefault(int(m[1]), {}).setdefault(m[3], {})[int(m[2])] = v
        else:
            state[k[len("mtp."):]] = v
    for i, parts in experts.items():
        E = len(parts["gate"])
        state[f"layers.{i}.mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([parts["gate"][e], parts["up"][e]]) for e in range(E)])
        state[f"layers.{i}.mlp.experts.down_proj"] = torch.stack([parts["down"][e] for e in range(E)])
    head.load_state_dict(state, strict=True)
    return head.to(torch.bfloat16).eval(), full


def split_experts(out: dict[str, torch.Tensor], layer: str) -> dict[str, torch.Tensor]:
    """calibrate_experts' fused results as the checkpoint's per-expert MTP
    tensors (mtp.layers.i.mlp.experts.e.{gate,up,down}_proj.weight[.gptq_*])."""
    res = {}
    for fused, names in (("gate_up_proj", ("gate_proj", "up_proj")), ("down_proj", ("down_proj",))):
        for suffix in ("", ".gptq_scales", ".gptq_biases", ".gptq_codes"):
            t = out.pop(f"mlp.experts.{fused}{suffix}")
            rows = t.shape[1] // len(names)
            for e in range(t.shape[0]):
                for j, name in enumerate(names):
                    key = f"{layer}.mlp.experts.{e}.{name}.weight{suffix}"
                    res[key] = t[e, j * rows:(j + 1) * rows].contiguous()
    return res


def calibrate_mtp(lm, head, full: int, hidden: list[torch.Tensor], chunks: list[torch.Tensor], recipe: dict,
                  floats, args) -> dict[str, torch.Tensor]:
    """GPTQ of the MTP head on what it reads at inference: the backbone's
    final (normed) hidden state at t, from the calibrated layers, and the
    embedding of token t+1."""
    for m in (lm.norm, lm.embed_tokens, head):
        m.to(DEVICE)
    with torch.no_grad():
        xs = [torch.cat([head.pre_fc_norm_embedding(lm.embed_tokens(c[:, 1:].to(DEVICE))),
                         head.pre_fc_norm_hidden(lm.norm(h)[:, :-1])], dim=-1) for h, c in zip(hidden, chunks)]
    lm.norm.cpu()
    lm.embed_tokens.cpu()
    out: dict[str, torch.Tensor] = {}
    bits = bits_for("mtp_fc", recipe)
    res = gptq_nbit(head.fc.weight.data.float(), torch.cat([x.reshape(-1, x.shape[-1]) for x in xs]), bits=bits,
                    group_size=args.group_size, percdamp=args.percdamp, device=DEVICE, scheme="affine")
    head.fc.weight.data.copy_(res["W_hat"].to(head.fc.weight.dtype).to(DEVICE))
    out["mtp.fc.weight"] = res["W_hat"].to(torch.bfloat16).cpu()
    out["mtp.fc.weight.gptq_scales"] = res["scale"].float().cpu()
    out["mtp.fc.weight.gptq_biases"] = res["bias"].float().cpu()
    out["mtp.fc.weight.gptq_codes"] = codes(res["W_hat"], res["scale"], res["bias"], args.group_size, bits).cpu()
    with torch.no_grad():
        xs = [head.fc(x) for x in xs]
    # The pairs are one token shorter than the chunks: their masks and rope.
    kwargs = layer_kwargs(lm, chunks[0][:, :-1])[full]
    for i, layer in enumerate(head.layers):
        lo = calibrate_layer(layer, xs, kwargs, recipe, floats, args, prefix="mtp_")
        with torch.no_grad():
            xs = [layer(x, **kwargs) for x in xs]
        if "mlp.experts.gate_up_proj" in lo:
            out.update(split_experts(lo, f"mtp.layers.{i}"))
        out.update({f"mtp.layers.{i}.{k}": v for k, v in lo.items()})
        print(f"MTP layer {i} ({sum(1 for k in lo if '.gptq_' not in k)} tensors)", flush=True)
    head.cpu()
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
    # The MTP head, when the recipe gives it bits ("mtp=4" or per component).
    if any(k.startswith("mtp") for k in recipe) and bits_for("mtp_fc", recipe) is None:
        raise SystemExit("the recipe gives the MTP head some bits but none for mtp_fc: use mtp=<bits> or add mtp_fc")
    mtp = mtp_tensors(Path(args.model)) if bits_for("mtp_fc", recipe) is not None else {}
    if bits_for("mtp_fc", recipe) is not None and not mtp:
        raise SystemExit("the recipe has bits for the MTP head, the checkpoint has no mtp.* tensors")
    if mtp:
        moe = any(".mlp.experts." in k for k in mtp)
        present |= {"mtp_fc", "mtp_attn"} | ({"mtp_shared", "mtp_experts_gate_up", "mtp_experts_down"}
                                             if moe else {"mtp_mlp"})
    missing = {c for c in present if bits_for(c, recipe) is None}
    if missing:
        raise SystemExit(f"recipe has no bits for: {sorted(missing)} (model has {sorted(present)})")

    chunks = load_chunks(args.calib, tok, args.calib_chunks, args.calib_chunk_tokens)
    if not chunks:
        raise SystemExit(f"no calibration chunk of {args.calib_chunk_tokens} tokens in {args.calib}")
    print(f"{len(chunks)} chunks x {args.calib_chunk_tokens} tokens; recipe {recipe}; {len(layers)} layers", flush=True)
    kwargs = layer_kwargs(lm, chunks[0])

    progress = work / "progress.json"
    # What the finished layers were made with: a resume must not mix recipes.
    # The MTP head's bits ("mtp...") aren't the layers': another head recipe
    # recalibrates only the head (layers/mtp.safetensors keeps its own).
    layer_recipe = ",".join(p for p in args.recipe.split(",") if not p.strip().startswith("mtp"))
    head_recipe = ",".join(p for p in args.recipe.split(",") if p.strip().startswith("mtp"))
    made_with = {"recipe": layer_recipe, "group_size": args.group_size, "calib": args.calib,
                 "calib_chunks": args.calib_chunks, "calib_chunk_tokens": args.calib_chunk_tokens,
                 "keep_float": [p.pattern for p in floats]}
    start = 0
    if progress.exists():
        old = json.load(open(progress))
        if old.get("made_with") != made_with:
            raise SystemExit(f"{work} holds layers made with {old.get('made_with')}, not {made_with}: "
                             "use another --work or delete it")
        start = old["done"] + 1
    # Written only once the work directory is known to be this recipe's.
    json.dump({"recipe": {k: bits_for(k, recipe) for k in sorted(present)}, "group_size": args.group_size,
               "components": COMPONENTS, "keep_float": [p.pattern for p in floats],
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
        out = calibrate_layer(layer, hidden, kwargs[i], recipe, floats, args)
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
    mtp_file = work / "layers" / "mtp.safetensors"
    head_made_with = json.dumps({"recipe": head_recipe, **{k: v for k, v in made_with.items() if k != "recipe"}},
                                sort_keys=True)
    if mtp_file.exists():
        try:
            with safe_open(mtp_file, "pt") as f:
                stale = (f.metadata() or {}).get("made_with") != head_made_with
        except Exception:
            stale = True   # cut off while it was written
        if stale:
            mtp_file.unlink()
    if mtp and not mtp_file.exists():
        head, full = build_mtp(lm, mtp)
        out = calibrate_mtp(lm, head, full, hidden, chunks, recipe, floats, args)
        save_file({k: v.contiguous() for k, v in out.items()}, mtp_file, metadata={"made_with": head_made_with})
        print(f"MTP head calibrated ({head_recipe})", flush=True)
    print("GPTQ_DONE", flush=True)


if __name__ == "__main__":
    main()
