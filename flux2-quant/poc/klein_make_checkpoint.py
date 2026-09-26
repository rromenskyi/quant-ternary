"""Builds the recipe checkpoint for LLMTray from an `mflux-save --quantize 8`
of FLUX.2 klein 4B: the transformer is re-quantized to 4-bit (RTN, group 64)
from the bf16 weights, the text encoder and the rest stay as saved (8-bit).
mflux reads each module's bits from its tensors' shapes, so the mixed
checkpoint loads with a plain model_path. The text encoder is truncated to
27 layers at load time (LLMTray's runner), not on disk.

    python klein_make_checkpoint.py <mflux-save-8bit dir>
"""
import json, sys, glob, os, mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
out = sys.argv[1]
m = Flux2Klein(quantize=None)
def pred(path, mod):
    if not hasattr(mod, "to_quantized") or not hasattr(mod, "weight") or mod.weight.shape[-1] % 64:
        return False
    return {"bits": 4, "group_size": 64}
nn.quantize(m.transformer, class_predicate=pred)
tensors = dict(tree_flatten(m.transformer.parameters()))
mx.eval(*tensors.values())
tdir = f"{out}/transformer"
old = sorted(glob.glob(f"{tdir}/*.safetensors"))
meta = {"mflux_version": "0.20.0", "quantization_level": "8"}
for f in old: os.remove(f)
mx.save_safetensors(f"{tdir}/0.safetensors", tensors, metadata=meta)
json.dump({"metadata": {}, "weight_map": {k: "0.safetensors" for k in tensors}}, open(f"{tdir}/model.safetensors.index.json", "w"), indent=2)
print("transformer tensors", len(tensors), "size", round(sum(v.nbytes for v in tensors.values()) / 1e9, 2), "GB")
