"""GPTQ calibration of an NVIDIA Dory 1.0 checkpoint (looped nGPT-style,
dense; `n_input_layers` -> a shared `n_recurrent_layers` block run
`n_recurrent_loops` times -> `n_output_layers`) with a per-component bit
recipe ("JANG": bits by component type, uniform across layers), on MLX's
affine grid. Same grid, Hessian math and output format (layers/NNN.safetensors
with .gptq_scales/.gptq_biases/.gptq_codes) as gptq_qwen35.py / gptq.py; see
dory_reference.py for why there is no `transformers` AutoModel to load here.

Why the recurrent block isn't just "layer i on layer i-1's quantized
output" the way the input and output stacks are: see docs/FINDINGS.md §2.
In short -- the 40 recurrent layers are the *same* modules run twice
(n_recurrent_loops=2 by default) with `loop_inject`/`justnorm` in between,
so there is only one quantization of each of them, not one per loop. This
driver chain-quantizes them once (exactly like a normal dense stack) off
the real loop-1 forward, then re-runs the (now fully quantized) block for
the remaining loops to get the real hidden state the output stack sees.

Components (module suffix within a layer -> recipe key). Every Dory layer
is either all-attention or all-MLP (never both), unlike a normal block:
    mixer.{q,k,v,g,o}_proj              attn
    mixer.{gate,up,down}_proj           mlp
The nGPT scale vectors (sqk, s_gate, suv_gate, suv_up, attn_alpha, mlp_alpha,
logit_scale) and the embedding are not nn.Linear and are never targeted --
they stay bf16 unconditionally, the same way every other model here keeps
its gate/router scalars out of the recipe.

    python gptq_dory.py --model SNAPSHOT --work /workspace/dory-gptq \\
        --recipe attn=8,mlp=6 --calib wikitext:/workspace/data/wiki.train.raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub.errors import StrictDataclassError
from safetensors.torch import save_file
from transformers import AutoTokenizer, PretrainedConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nemotron-extreme-quant" / "poc"))
from gptq import gptq_nbit  # noqa: E402  (MLX-exact affine grid)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dory_reference import DoryModel, load_dory, loop_inject, justnorm  # noqa: E402


def load_tokenizer(model_path: str):
    """AutoTokenizer.from_pretrained, with the same fallback
    mlx_lm.tokenizer_utils.load uses: transformers' generic `layer_types`
    validator rejects Dory's config.json (it uses "mlp" as a layer type,
    not one of the fixed attention-pattern names transformers expects
    there), so AutoConfig construction raises before the tokenizer even
    loads. Retried with a stub config carrying only the handful of fields
    a tokenizer actually needs."""
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except (AttributeError, ValueError, KeyError, StrictDataclassError):
        raw = json.loads((Path(model_path) / "config.json").read_text())
        stub = {k: raw[k] for k in ("model_type", "max_position_embeddings", "vocab_size",
                                    "bos_token_id", "eos_token_id", "pad_token_id") if k in raw}
        return AutoTokenizer.from_pretrained(model_path, config=PretrainedConfig(**stub))

COMPONENTS = {
    "mixer.q_proj": "attn", "mixer.k_proj": "attn", "mixer.v_proj": "attn",
    "mixer.g_proj": "attn", "mixer.o_proj": "attn",
    "mixer.gate_proj": "mlp", "mixer.up_proj": "mlp", "mixer.down_proj": "mlp",
}
PREFIX = "backbone.layers"
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


def mixer_args(layer, positions, causal, sliding):
    if not layer.is_attn:
        return ()
    return (positions, sliding if layer.mixer.sliding else causal)


def calibrate_layer(layer, hidden: list[torch.Tensor], margs, recipe: dict, args) -> dict[str, torch.Tensor]:
    """GPTQ of one Dory layer (all-attention or all-MLP) on its inputs."""
    targets = {n: m for n, m in layer.named_modules() if n in COMPONENTS and isinstance(m, nn.Linear)}
    inputs: dict[str, list[torch.Tensor]] = {n: [] for n in targets}
    hooks = [m.register_forward_hook(lambda m, a, o, n=n: inputs[n].append(a[0].detach().reshape(-1, a[0].shape[-1])))
             for n, m in targets.items()]
    t0 = time.time()
    with torch.no_grad():
        for h in hidden:
            layer(h, *margs)
    for h in hooks:
        h.remove()
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
    print(f"  capture+quantize {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="bf16 checkpoint (HF snapshot dir)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--recipe", required=True, help="e.g. attn=8,mlp=6")
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
    missing = {"attn", "mlp"} - set(recipe)
    if missing:
        raise SystemExit(f"recipe has no bits for: {sorted(missing)}")
    work = Path(args.work)
    (work / "layers").mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(args.model)
    model, cfg = load_dory(args.model, device=DEVICE, dtype=torch.bfloat16)
    model.eval()
    layers: list[nn.Module] = list(model.layers)

    chunks = load_chunks(args.calib, tok, args.calib_chunks, args.calib_chunk_tokens)
    if not chunks:
        raise SystemExit(f"no calibration chunk of {args.calib_chunk_tokens} tokens in {args.calib}")
    print(f"{len(chunks)} chunks x {args.calib_chunk_tokens} tokens; recipe {recipe}; "
          f"{len(layers)} layers ({cfg.n_input_layers}/{cfg.n_recurrent_layers}x{cfg.n_recurrent_loops}"
          f"/{cfg.n_output_layers})", flush=True)

    T = args.calib_chunk_tokens
    positions = torch.arange(T, device=DEVICE)
    causal, sliding = model._masks(T, DEVICE, torch.bfloat16)

    made_with = {"recipe": args.recipe, "group_size": args.group_size, "calib": args.calib,
                 "calib_chunks": args.calib_chunks, "calib_chunk_tokens": args.calib_chunk_tokens}
    progress = work / "progress.json"
    start = 0
    if progress.exists():
        old = json.load(open(progress))
        if old.get("made_with") != made_with:
            raise SystemExit(f"{work} holds layers made with {old.get('made_with')}, not {made_with}: "
                             "use another --work or delete it")
        start = old["done"] + 1
    json.dump({"recipe": recipe, "group_size": args.group_size, "components": COMPONENTS,
               "n_input_layers": cfg.n_input_layers, "n_recurrent_layers": cfg.n_recurrent_layers,
               "n_output_layers": cfg.n_output_layers, "n_recurrent_loops": cfg.n_recurrent_loops},
              open(work / "quant_recipe.json", "w"), indent=2)

    lo, hi = cfg.n_input_layers, cfg.n_input_layers + cfg.n_recurrent_layers
    if start:
        state = torch.load(work / "state.pt")
        hidden = [h.to(DEVICE) for h in state["hidden"]]
        input_h = [h.to(DEVICE) for h in state["input_h"]] if state["input_h"] is not None else None
        print(f"resuming at layer {start}", flush=True)
    else:
        with torch.no_grad():
            hidden = [model.embeddings(c.to(DEVICE)) for c in chunks]
        input_h = None

    started = time.time()

    def checkpoint(i, hidden, input_h):
        torch.save({"hidden": [h.cpu() for h in hidden],
                    "input_h": [h.cpu() for h in input_h] if input_h is not None else None},
                   work / "state.pt")
        json.dump({"done": i, "layers": len(layers), "made_with": made_with}, open(progress, "w"))

    def run_and_save(i, hidden):
        layer = layers[i].to(DEVICE)
        margs = mixer_args(layer, positions, causal, sliding)
        out = calibrate_layer(layer, hidden, margs, recipe, args)
        with torch.no_grad():
            hidden = [layer(h, *margs) for h in hidden]
        save_file({f"{PREFIX}.{i}.{k}": v.contiguous() for k, v in out.items()}, work / "layers" / f"{i:03d}.safetensors")
        n = sum(1 for k in out if ".gptq_" not in k)
        print(f"layer {i:3d}/{len(layers)} ({n} tensors) {time.time() - started:.0f}s", flush=True)
        return hidden

    i = start
    while i < lo:
        hidden = run_and_save(i, hidden)
        i += 1
        checkpoint(i - 1, hidden, input_h)
    if i == lo:
        input_h = [h.clone() for h in hidden]
    while i < hi:
        if i == lo:
            # First loop's start: state == input_h (Model.forward's `state = input_h`
            # before the first loop_inject), real loop-1 activations.
            with torch.no_grad():
                hidden = [loop_inject(h, h) for h in input_h]
        hidden = run_and_save(i, hidden)
        i += 1
        checkpoint(i - 1, hidden, input_h)
    if i == hi:
        # Loop 1 is now fully quantized (every recurrent layer above ran
        # through run_and_save once, so re-deriving this on a resume costs
        # nothing but forward passes -- no requantization happens here).
        # Re-run it, as-is, for the remaining configured loops to get the
        # real hidden state the output stack sees.
        with torch.no_grad():
            state = [justnorm(h) for h in hidden]
            for _ in range(cfg.n_recurrent_loops - 1):
                hidden = [loop_inject(s, ih) for s, ih in zip(state, input_h)]
                for j in range(lo, hi):
                    hidden = [layers[j](h) if not layers[j].is_attn
                             else layers[j](h, positions, sliding if layers[j].mixer.sliding else causal)
                             for h in hidden]
                state = [justnorm(h) for h in hidden]
            hidden = state
        print(f"recurrent block done, {cfg.n_recurrent_loops} loops re-run with quantized weights", flush=True)
    while i < cfg.num_hidden_layers:
        hidden = run_and_save(i, hidden)
        i += 1
        checkpoint(i - 1, hidden, input_h)

    print("GPTQ_DONE", flush=True)


if __name__ == "__main__":
    main()
