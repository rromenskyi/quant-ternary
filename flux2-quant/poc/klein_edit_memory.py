"""FLUX.2 klein 4B editing (Flux2KleinEdit) with the recipe from FINDINGS
(text encoder 8-bit truncated to 27 layers, transformer 4-bit, VAE tiled
256): peak memory and time for txt2img vs editing with one and two
reference images. Saves the images for a visual check.

    python klein_edit_memory.py <out_dir>
"""
import json, sys, time, mlx.core as mx, mlx.nn as nn
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.common.vae.tiling_config import TilingConfig
from mlx.utils import tree_flatten
OUT = sys.argv[1]

def recipe(m):
    def pred(bits):
        def p(path, mod):
            if not hasattr(mod, "to_quantized") or not hasattr(mod, "weight") or mod.weight.shape[-1] % 64:
                return False
            return {"bits": bits, "group_size": 64}
        return p
    nn.quantize(m.text_encoder, class_predicate=pred(8))
    nn.quantize(m.transformer, class_predicate=pred(4))
    m.text_encoder.layers = m.text_encoder.layers[:27]
    m.tiling_config = TilingConfig(vae_decode_tile_size=256)
    mx.eval(m.parameters()); mx.clear_cache()
    return m

def gb(x): return round(x / 1e9, 2)
def weights(m): return gb(sum(v.nbytes for _, v in tree_flatten(m.parameters())))
res = {}

def run(name, fn):
    mx.eval(); mx.clear_cache(); mx.reset_peak_memory(); t = time.time()
    img = fn()
    res[name] = {"sec": round(time.time() - t, 1), "peak_gb": gb(mx.get_peak_memory())}
    img.image.save(f"{OUT}/{name}.png")
    print(name, res[name], flush=True)

gen = recipe(Flux2Klein(quantize=None))
res["weights_gb"] = weights(gen)
print("weights", res["weights_gb"], "GB", flush=True)
P = "A red fox sitting in a snowy birch forest at dawn, soft golden light, photorealistic"
gen.generate_image(seed=1, prompt=P, num_inference_steps=4, height=512, width=512)  # warm-up
run("txt2img_1024", lambda: gen.generate_image(seed=42, prompt=P, num_inference_steps=4, height=1024, width=1024))
run("txt2img_cat_1024", lambda: gen.generate_image(seed=5, prompt="A ginger cat sitting on a windowsill, photorealistic", num_inference_steps=4, height=1024, width=1024))
del gen; mx.clear_cache()

ed = recipe(Flux2KleinEdit(quantize=None))
fox, cat = f"{OUT}/txt2img_1024.png", f"{OUT}/txt2img_cat_1024.png"
run("edit_night_1ref", lambda: ed.generate_image(seed=42, prompt="Make it night, with a full moon and falling snow", num_inference_steps=4, height=1024, width=1024, image_paths=[fox]))
run("edit_hat_1ref", lambda: ed.generate_image(seed=42, prompt="Put a small red knitted hat on the fox", num_inference_steps=4, height=1024, width=1024, image_paths=[fox]))
run("edit_2ref", lambda: ed.generate_image(seed=42, prompt="The fox and the cat sitting together in the snowy forest", num_inference_steps=4, height=1024, width=1024, image_paths=[fox, cat]))
json.dump(res, open(f"{OUT}/edit_results.json", "w"), indent=2)
