"""GPTQ calibration of an Inkling 2 checkpoint's text decoder -- Inkling 2
Mini (`inkling2_loop`: layers loop_start..loop_end-1 run loop_num_passes
times with shared weights) or Inkling 2 Small (`inkling2`, no loop) -- with
a per-component bit recipe ("JANG"), on MLX's affine grid. Same grid,
Hessian math and output format as qwen35-quant/poc/gptq_qwen35.py, whose
expert calibration this reuses.

The model loads through transformers (Mini's NVFP4 experts are dequantized
to bf16 on load) and stays in CPU memory; only the layer being calibrated
is on the GPU. Each chunk carries (hidden, block_states): the block
attention residual reads every earlier block's output, not just the last
hidden state.

The loop (Mini), as for Dory (dory-quant/docs/FINDINGS.md section 2): one
weight per looped layer, so the looped layers are chain-quantized once, on
the first pass's inputs. Then the later passes run for real on the
quantized layers (with the message gates reading the previous pass), and
the layers after the loop are calibrated on that output.

Components (transformers module within a decoder layer -> recipe key):
  self_attn.{q,k,v}_proj.linear, self_attn.relative_position_bias.linear,
  self_attn.o_proj.{linear,gate_proj}                      attn
  mlp.experts.gate_up_proj / down_proj (fused [E, out, in]) experts_gate_up / experts_down
Kept bf16 / fp32 (--keep-float): the attention sink projection, the
dynamic convolutions' weight projections, the loop's message gates and the
router.

Writes, under --work:
  layers/NNN.safetensors  layer NNN's quantized weights under their
                          transformers names, plus .gptq_scales / .gptq_biases
                          / .gptq_codes (resume unit)
  hidden.pt, progress.json  the chunk states after the last finished layer
  quant_recipe.json       the recipe, for assemble_checkpoint.py and convert_mlx.py

    python gptq_inkling2.py --model SNAPSHOT --work /workspace/gptq-mini \\
        --recipe attn=8,experts=3 \\
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
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nemotron-extreme-quant" / "poc"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen35-quant" / "poc"))
import gptq_qwen35 as q35  # noqa: E402
from gptq import gptq_nbit  # noqa: E402  (MLX-exact affine grid)

COMPONENTS = {
    "self_attn.q_proj.linear": "attn", "self_attn.k_proj.linear": "attn", "self_attn.v_proj.linear": "attn",
    "self_attn.relative_position_bias.linear": "attn",
    "self_attn.o_proj.linear": "attn", "self_attn.o_proj.gate_proj": "attn",
    "mlp.experts.gate_up_proj": "experts_gate_up", "mlp.experts.down_proj": "experts_down",
}
DEFAULT_KEEP_FLOAT = [r"\.sink_proj$", r"\.sconv\.weight_proj$", r"_message_gate\.", r"mlp\.gate$"]
PREFIX = "model.language_model.layers"
DEVICE = "cuda"


class _Catch(torch.nn.Module):
    """Stands in for a decoder layer: records the kwargs the text model
    passes it (masks and relative distances differ by layer type)."""

    def __init__(self, store: dict, index: int, layer):
        super().__init__()
        self.store, self.index = store, index
        self.attention_type, self.is_block_boundary = layer.attention_type, layer.is_block_boundary

    def forward(self, hidden_states, block_states, messages=None, **kwargs):
        self.store.setdefault(self.index, {k: v for k, v in kwargs.items() if k not in ("cache", "cache_offset")})
        return hidden_states


def layer_kwargs(lm, chunk: torch.Tensor) -> dict[int, dict]:
    store: dict[int, dict] = {}
    real = lm.layers
    lm.layers = torch.nn.ModuleList([_Catch(store, i, layer) for i, layer in enumerate(real)])
    lm.embed_tokens.to(DEVICE)
    try:
        with torch.no_grad():
            lm(input_ids=chunk.to(DEVICE), use_cache=False)
    finally:
        lm.layers = real
        lm.embed_tokens.cpu()
    return store


def is_loop_layer(layer) -> bool:
    return hasattr(layer, "q_message_gate")


def step(layer, state, kwargs, messages=None):
    """One decoder layer on one chunk's (hidden, block_states), as the text
    model runs it: a block boundary adds the layer's input to the blocks."""
    h, bs = state
    extra = {"messages": messages} if is_loop_layer(layer) else {}
    out = layer(h, bs, **extra, **kwargs)
    if layer.is_block_boundary:
        bs = torch.cat([bs, h.unsqueeze(-2)], dim=-2)
    return out, bs


def calibrate_layer(layer, states, kwargs, recipe, floats, args) -> dict[str, torch.Tensor]:
    targets = {n: m for n, m in layer.named_modules()
               if n in COMPONENTS and isinstance(m, torch.nn.Linear) and not q35.keep_float(n, floats)}
    # A renamed module would otherwise stay bf16 without a word.
    missing = {n for n, c in COMPONENTS.items() if not c.startswith("experts")} - set(targets)
    missing = {n for n in missing if not q35.keep_float(n, floats)}
    if missing:
        raise SystemExit(f"no {sorted(missing)} in the layer: {sorted(n for n, _ in layer.named_modules())[:20]}")
    inputs: dict[str, list[torch.Tensor]] = {n: [] for n in targets}
    hooks = [m.register_forward_hook(lambda m, a, o, n=n: inputs[n].append(a[0].detach().reshape(-1, a[0].shape[-1])))
             for n, m in targets.items()]
    experts = layer.mlp.experts
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    if not q35.keep_float("mlp.experts", floats):
        hooks.append(experts.register_forward_pre_hook(lambda _m, a: seen.append((a[0].detach(), a[1].detach()))))
    t0 = time.time()
    with torch.no_grad():
        for s in states:
            step(layer, s, kwargs)
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
        bits = q35.bits_for(COMPONENTS[n], recipe)
        res = gptq_nbit(m.weight.data.float(), torch.cat(inputs.pop(n)), bits=bits, group_size=args.group_size,
                        percdamp=args.percdamp, device=DEVICE, scheme="affine")
        m.weight.data.copy_(res["W_hat"].to(m.weight.dtype).to(m.weight.device))
        out[f"{n}.weight"] = res["W_hat"].to(torch.bfloat16).cpu()
        out[f"{n}.weight.gptq_scales"] = res["scale"].float().cpu()
        out[f"{n}.weight.gptq_biases"] = res["bias"].float().cpu()
        out[f"{n}.weight.gptq_codes"] = q35.codes(res["W_hat"], res["scale"], res["bias"], args.group_size, bits).cpu()
    t2 = time.time()
    if rows:
        out.update(q35.calibrate_experts(experts, rows, recipe, args))
    print(f"  capture {t1 - t0:.0f}s, linears {t2 - t1:.0f}s, experts {time.time() - t2:.0f}s "
          f"({len(rows)} experts hit)", flush=True)
    return out


def later_passes(lm, states, pre, kwargs, cfg):
    """Loop passes 2.. on the (quantized) looped layers: each restarts from
    the pre-loop states and reads messages from the previous pass's blocks."""
    start, end = cfg.loop_layer_index_start, cfg.loop_layer_index_end
    n_pre = pre[0][1].shape[-2]
    layers = lm.layers[start:end]
    for m in [lm.message_input, *layers]:
        m.to(DEVICE)
    with torch.no_grad():
        for _ in range(1, cfg.loop_num_passes):
            new = []
            for (h, bs), p in zip(states, pre):
                prev_bs, prev_h = bs[..., n_pre + 1:, :], h
                s = (p[0].to(DEVICE), p[1].to(DEVICE))
                for j, layer in enumerate(layers):
                    messages = lm.message_input(prev_bs, prev_h, j)
                    s = step(layer, s, kwargs[start + j], messages)
                new.append(s)
            states = new
    for m in [lm.message_input, *layers]:
        m.cpu()
    return states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF snapshot dir (NVFP4 experts are dequantized on load)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--recipe", required=True, help="e.g. attn=8,experts=3")
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
    DEVICE = q35.DEVICE = args.device
    recipe = q35.parse_recipe(args.recipe)
    floats = [re.compile(p) for p in (args.keep_float or DEFAULT_KEEP_FLOAT)]
    work = Path(args.work)
    (work / "layers").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).eval()  # CPU memory
    lm = model.model.language_model
    cfg = lm.config
    layers = lm.layers
    loop = (cfg.loop_layer_index_start, cfg.loop_layer_index_end) if hasattr(cfg, "loop_layer_index_start") else None
    present = set(COMPONENTS.values())
    missing = {c for c in present if q35.bits_for(c, recipe) is None}
    if missing:
        raise SystemExit(f"recipe has no bits for: {sorted(missing)} (model has {sorted(present)})")

    chunks = q35.load_chunks(args.calib, tok, args.calib_chunks, args.calib_chunk_tokens)
    if not chunks:
        raise SystemExit(f"no calibration chunk of {args.calib_chunk_tokens} tokens in {args.calib}")
    print(f"{len(chunks)} chunks x {args.calib_chunk_tokens} tokens; recipe {recipe}; {len(layers)} layers; "
          f"loop {loop and (*loop, cfg.loop_num_passes)}", flush=True)
    kwargs = layer_kwargs(lm, chunks[0])

    progress = work / "progress.json"
    made_with = {"recipe": args.recipe, "group_size": args.group_size, "calib": args.calib,
                 "calib_chunks": args.calib_chunks, "calib_chunk_tokens": args.calib_chunk_tokens,
                 "keep_float": [p.pattern for p in floats]}
    start = 0
    if progress.exists():
        old = json.load(open(progress))
        if old.get("made_with") != made_with:
            raise SystemExit(f"{work} holds layers made with {old.get('made_with')}, not {made_with}: "
                             "use another --work or delete it")
        start = old["done"] + 1
    json.dump({"recipe": {k: q35.bits_for(k, recipe) for k in sorted(present)}, "group_size": args.group_size,
               "components": COMPONENTS, "keep_float": [p.pattern for p in floats],
               "loop": loop and {"start": loop[0], "end": loop[1], "passes": cfg.loop_num_passes,
                                 "calibrated_on": "first pass"},
               "calibration": {"chunks": args.calib_chunks, "tokens": args.calib_chunk_tokens, "sources": args.calib}},
              open(work / "quant_recipe.json", "w"), indent=2)
    pre = None
    if start:
        # The later loop passes run the looped layers again: quantized, as calibrated.
        for i in range(start):
            for k, v in load_file(work / "layers" / f"{i:03d}.safetensors").items():
                if ".gptq_" not in k:
                    layers[i].get_parameter(k[len(f"{PREFIX}.{i}."):]).data.copy_(v)
        saved = torch.load(work / "hidden.pt")
        states = [(h.to(DEVICE), b.to(DEVICE)) for h, b in saved["states"]]
        pre = saved.get("pre")
        print(f"resuming at layer {start}", flush=True)
    else:
        lm.embed_tokens.to(DEVICE)
        with torch.no_grad():
            states = []
            for c in chunks:
                h = lm.embed_tokens(c.to(DEVICE))
                states.append((h, h.new_zeros(*h.shape[:-1], 0, h.shape[-1])))
        lm.embed_tokens.cpu()

    started = time.time()
    for i in range(start, len(layers)):
        if loop and i == loop[0]:
            pre = [(h.cpu(), b.cpu()) for h, b in states]
        layer = layers[i].to(DEVICE)
        out = calibrate_layer(layer, states, kwargs[i], recipe, floats, args)
        with torch.no_grad():
            states = [step(layer, s, kwargs[i]) for s in states]
        layers[i] = layer.cpu()
        if loop and i == loop[1] - 1:
            states = later_passes(lm, states, pre, kwargs, cfg)
            pre = None
        save_file({f"{PREFIX}.{i}.{k}": v.contiguous() for k, v in out.items()}, work / "layers" / f"{i:03d}.safetensors")
        torch.save({"states": [(h.cpu(), b.cpu()) for h, b in states], "pre": pre}, work / "hidden.pt")
        json.dump({"done": i, "layers": len(layers), "made_with": made_with}, open(progress, "w"))
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        n = sum(1 for k in out if ".gptq_" not in k)
        print(f"layer {i:2d}/{len(layers)} ({n} tensors) {time.time() - started:.0f}s", flush=True)
    print("GPTQ_DONE", flush=True)


if __name__ == "__main__":
    main()
