"""Per-stage peak memory + time of FLUX.2 klein 4B in mflux: prompt encode,
each transformer step, VAE decode. Used for docs/FINDINGS.md "Measurements".

    python klein_stage_memory.py <bits|bf16> [size=1024] [vae_tile_px=0]
"""
import sys, time, mlx.core as mx
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
import mflux.models.flux2.variants.txt2img.flux2_klein as fk
q = None if sys.argv[1] == "bf16" else int(sys.argv[1]); size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
m = Flux2Klein(quantize=q); mx.eval(m.parameters())
tile = int(sys.argv[3]) if len(sys.argv) > 3 else 0
if tile:
    from mflux.models.common.vae.tiling_config import TilingConfig
    m.tiling_config = TilingConfig(vae_decode_tile_size=tile)
base = mx.get_active_memory()/1e9
stats = {}
def wrap(name, fn):
    def inner(*a, **k):
        mx.eval(); mx.reset_peak_memory(); t = time.time()
        out = fn(*a, **k)
        mx.eval(out if isinstance(out, mx.array) else [x for x in (out if isinstance(out, tuple) else [out]) if isinstance(x, mx.array)])
        s = stats.setdefault(name, [0, 0.0, 0])
        s[0] = max(s[0], mx.get_peak_memory()); s[1] += time.time() - t; s[2] += 1
        return out
    return inner
enc = fk.Flux2PromptEncoder.encode_prompt
fk.Flux2PromptEncoder.encode_prompt = staticmethod(wrap("encode", enc))
m.vae.decode_packed_latents = wrap("vae_decode", m.vae.decode_packed_latents)
pred = m._predict
m._predict = lambda tr: wrap("transformer_step", pred(tr))
P = "A red fox sitting in a snowy birch forest at dawn, soft golden light, photorealistic"
m.generate_image(seed=1, prompt=P, num_inference_steps=4, height=size, width=size)  # warm-up
stats.clear()
t = time.time(); img = m.generate_image(seed=42, prompt=P, num_inference_steps=4, height=size, width=size); total = time.time() - t
img.image.save(f"klein_q{q}_{size}_tile{tile}.png")
print(f"q={q} {size}x{size}: weights resident {base:.2f} GB, total {total:.1f}s, tiling={m.tiling_config}")
for k, (peak, sec, n) in stats.items():
    print(f"  {k:16s} peak {peak/1e9:6.2f} GB   {sec:6.2f}s over {n} call(s)")
