"""Compare per-component quantization mixes of FLUX.2 klein 4B against bf16:
PSNR over fixed prompts/seed (fox, neon sign text, portrait), per-component
weight sizes. Text encoder truncated to 27 layers (bit-identical output, see
FINDINGS). Writes <out>/<config>_<prompt>.png and <out>/results.json.

    python klein_quant_mix.py <out_dir>
"""
import json, re, sys, time, numpy as np, mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
from PIL import Image
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.common.vae.tiling_config import TilingConfig
OUT = sys.argv[1]
PROMPTS = [
    ("fox", "A red fox sitting in a snowy birch forest at dawn, soft golden light, photorealistic"),
    ("sign", "A neon sign on a brick wall that says \"OPEN 24/7\" in bright pink letters, night, rain"),
    ("portrait", "Close-up portrait of an elderly fisherman with a grey beard, wool cap, harbour background, 85mm photo"),
]
SENSITIVE = re.compile(r"(modulation|embedder|embed|norm_out|proj_out|transformer_blocks\.\d+\.attn\.)")
def q_all(bits):
    return lambda comp, path: bits
CONFIGS = {
    "bf16": None,
    "A_all8": {"te": q_all(8), "tr": q_all(8)},
    "B_all4": {"te": q_all(4), "tr": q_all(4)},
    "C_te8_tr4": {"te": q_all(8), "tr": q_all(4)},
    "D_te4_tr8": {"te": q_all(4), "tr": q_all(8)},
    "E_te8_tr4_sens8": {"te": q_all(8), "tr": lambda c, p: 8 if SENSITIVE.search(p) and not p.startswith("single_") else 4},
}
def quantize(comp, rule, name):
    def pred(path, m):
        if not hasattr(m, "to_quantized") or not hasattr(m, "weight") or m.weight.shape[-1] % 64:
            return False
        return {"bits": rule(name, path), "group_size": 64}
    nn.quantize(comp, class_predicate=pred)
def nbytes(mod): return sum(v.nbytes for _, v in tree_flatten(mod.parameters()))
res = {}
for cname, cfg in CONFIGS.items():
    m = Flux2Klein(quantize=None)
    m.tiling_config = TilingConfig(vae_decode_tile_size=256)
    if cfg:
        quantize(m.text_encoder, cfg["te"], "te"); quantize(m.transformer, cfg["tr"], "tr")
    m.text_encoder.layers = m.text_encoder.layers[:27]   # dead layers (FINDINGS): output unchanged
    mx.eval(m.parameters())
    size = {"te": nbytes(m.text_encoder)/1e9, "tr": nbytes(m.transformer)/1e9}
    mx.reset_peak_memory(); t = time.time()
    for pname, p in PROMPTS:
        m.generate_image(seed=7, prompt=p, num_inference_steps=4, height=1024, width=1024).image.save(f"{OUT}/{cname}_{pname}.png")
    res[cname] = {"sec_per_img": (time.time()-t)/len(PROMPTS), "peak_gb": mx.get_peak_memory()/1e9, **{k+"_gb": v for k, v in size.items()}}
    print(cname, json.dumps(res[cname]), flush=True)
    del m; mx.clear_cache()
def load(n): return np.asarray(Image.open(n).convert("RGB")).astype(float)
for cname in CONFIGS:
    if cname == "bf16": continue
    ps = []
    for pname, _ in PROMPTS:
        mse = ((load(f"{OUT}/bf16_{pname}.png") - load(f"{OUT}/{cname}_{pname}.png"))**2).mean()
        ps.append(10*np.log10(255**2/mse))
    res[cname]["psnr_vs_bf16"] = [round(x, 1) for x in ps]
    print(f"{cname:18s} PSNR vs bf16 {ps[0]:5.1f} {ps[1]:5.1f} {ps[2]:5.1f}  | TE {res[cname]['te_gb']:.2f}GB TR {res[cname]['tr_gb']:.2f}GB peak {res[cname]['peak_gb']:.1f}GB {res[cname]['sec_per_img']:.1f}s/img")
json.dump(res, open(f"{OUT}/results.json", "w"), indent=2)
