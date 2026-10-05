"""GPTQ-calibrate a Qwen3.5 (qwen3_5) checkpoint's text decoder with a
per-component bit recipe ("JANG": bits by component type, uniform across
layers), and write a copy of the ORIGINAL checkpoint whose decoder weights
already sit on MLX's affine grid -- so stock `mlx_lm.convert` with the same
recipe (quant_recipe.json, read by mlx_convert_recipe.py) reproduces these
Hessian-calibrated codes instead of re-deriving them by round-to-nearest.

Everything the decoder doesn't own -- the vision tower (model.visual.*),
the MTP head (mtp.*, which the HF model class doesn't even load), the
embeddings and norms -- is copied byte for byte from the source shards.

Sequential: layer i is calibrated on the outputs of the already-quantized
layers 0..i-1 (each calibration pass stops right after layer i).

Usage (on the GPU pod):
    python gptq_qwen35.py --model /root/frognano-bf16 --output /root/frognano-jang \
        --wikitext /root/wikitext-2-raw/wiki.train.raw \
        --recipe attn=8,linear=6,mlp=3 --group-size 64 \
        --calib-chunks 64 --calib-chunk-tokens 512
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nemotron-extreme-quant" / "poc"))
from gptq import gptq_nbit  # noqa: E402  (MLX-exact affine grid)

# Decoder projection -> recipe component. in_proj_a / in_proj_b (32 rows:
# the delta rule's per-head gates) stay bf16: tiny, and the gates are
# sensitive.
COMPONENTS = {
    "self_attn.q_proj": "attn", "self_attn.k_proj": "attn", "self_attn.v_proj": "attn", "self_attn.o_proj": "attn",
    "linear_attn.in_proj_qkv": "linear", "linear_attn.in_proj_z": "linear", "linear_attn.out_proj": "linear",
    "mlp.gate_proj": "mlp", "mlp.up_proj": "mlp", "mlp.down_proj": "mlp",
}


class _Stop(Exception):
    pass


def parse_recipe(text: str) -> dict[str, int]:
    recipe = {}
    for part in text.split(","):
        key, bits = part.split("=")
        recipe[key.strip()] = int(bits)
    missing = set(COMPONENTS.values()) - recipe.keys()
    if missing:
        raise SystemExit(f"recipe has no bits for: {sorted(missing)}")
    return recipe


def load_chunks(path: str, tokenizer, n: int, tokens: int) -> list[torch.Tensor]:
    text = open(path, encoding="utf-8").read()
    stride = max(1, len(text) // n)
    chunks = []
    for i in range(n):
        ids = tokenizer(text[i * stride : i * stride + tokens * 6], return_tensors="pt",
                        truncation=True, max_length=tokens)["input_ids"]
        if ids.shape[1] == tokens:
            chunks.append(ids)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--recipe", required=True, help="e.g. attn=8,linear=6,mlp=3")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--calib-chunks", type=int, default=64)
    ap.add_argument("--calib-chunk-tokens", type=int, default=512)
    ap.add_argument("--percdamp", type=float, default=0.01)
    args = ap.parse_args()
    recipe = parse_recipe(args.recipe)
    src, out = Path(args.model), Path(args.output)

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda").eval()
    layers = model.model.language_model.layers
    chunks = load_chunks(args.wikitext, tok, args.calib_chunks, args.calib_chunk_tokens)
    print(f"{len(chunks)} calibration chunks x {args.calib_chunk_tokens} tokens; recipe {recipe}", flush=True)

    quantized: dict[str, torch.Tensor] = {}  # checkpoint key -> on-grid bf16 weight
    started = time.time()
    for i, layer in enumerate(layers):
        targets = {name: mod for name, mod in layer.named_modules() if name in COMPONENTS}
        inputs: dict[str, list[torch.Tensor]] = {name: [] for name in targets}
        hooks = [mod.register_forward_hook(lambda m, a, o, n=name: inputs[n].append(a[0].detach().reshape(-1, a[0].shape[-1]).float().cpu()))
                 for name, mod in targets.items()]

        def stop(_m, _a, _o):
            raise _Stop

        hooks.append(layer.register_forward_hook(stop))
        with torch.no_grad():
            for ids in chunks:
                try:
                    model.model.language_model(input_ids=ids.cuda(), use_cache=False)
                except _Stop:
                    pass
        for h in hooks:
            h.remove()

        for name, mod in targets.items():
            bits = recipe[COMPONENTS[name]]
            X = torch.cat(inputs.pop(name))
            res = gptq_nbit(mod.weight.data.float(), X, bits=bits, group_size=args.group_size,
                            percdamp=args.percdamp, device="cuda", scheme="affine")
            W = res["W_hat"].to(torch.bfloat16)
            mod.weight.data.copy_(W.to(mod.weight.device))
            quantized[f"model.language_model.layers.{i}.{name}.weight"] = W.cpu()
        print(f"layer {i:2d} ({'attn' if hasattr(layer, 'self_attn') else 'linear'}) done, "
              f"{time.time() - started:.0f}s", flush=True)

    # The source checkpoint with the calibrated decoder weights swapped in.
    out.mkdir(parents=True, exist_ok=True)
    index = json.load(open(src / "model.safetensors.index.json"))
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(src / shard)
        for key in tensors:
            if key in quantized:
                assert tensors[key].shape == quantized[key].shape, key
                tensors[key] = quantized.pop(key)
        save_file(tensors, out / shard, metadata={"format": "pt"})
    assert not quantized, f"not written: {list(quantized)[:3]}"
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy(f, out / f.name)
    json.dump({"recipe": recipe, "group_size": args.group_size, "components": COMPONENTS,
               "bf16": ["linear_attn.in_proj_a", "linear_attn.in_proj_b"],
               "calibration": {"chunks": len(chunks), "tokens": args.calib_chunk_tokens, "source": "wikitext-2-raw train"}},
              open(out / "quant_recipe.json", "w"), indent=2)
    print(f"wrote {out} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
