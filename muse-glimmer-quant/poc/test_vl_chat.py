"""End-to-end functional test: load the built VL MLX model, ask it about a
real photo, print the answer."""
import sys
import mlx.core as mx
import numpy as np
from PIL import Image
from transformers.models.muse_glimmer.image_processing_muse_glimmer import MuseGlimmerImageProcessor

from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

model_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/Muse-Glimmer-30B-VL-gptq-mlx-jang"
model, tokenizer = load(model_path)
print("model loaded, vision_tower:", model.vision_tower is not None)

image = Image.open("/workspace/cats.jpg").convert("RGB")
inputs = MuseGlimmerImageProcessor()(images=[image], return_tensors="np")
pixel_values = mx.array(inputs["pixel_values"].astype(np.float32))
grid_thw = inputs["image_grid_thw"]

n_merged = int((grid_thw[0, 1] // model.vision_tower.merge_size) * (grid_thw[0, 2] // model.vision_tower.merge_size))
image_token_id = model.args.image_token_id
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What animals are in this picture? Answer in one sentence."},
]}]
prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
# The template emits ONE <|patch|> placeholder per image; expand it to the
# real merged-patch count (the HF processor does this expansion itself).
n_before = prompt_ids.count(image_token_id)
if n_before != 1:
    raise SystemExit(f"expected exactly one image placeholder token, found {n_before}: {prompt_ids}")
i = prompt_ids.index(image_token_id)
prompt_ids = prompt_ids[:i] + [image_token_id] * n_merged + prompt_ids[i + 1:]
input_ids = mx.array([prompt_ids])

embeds = model.embed_with_images(input_ids, pixel_values, grid_thw)
mx.eval(embeds)
print("embeds:", embeds.shape)

cache = model.make_cache()
logits = model(input_ids, cache=cache, input_embeddings=embeds)
sampler = make_sampler(0.0)
tok = sampler(logits[:, -1, :])
mx.eval(tok)
answer_ids = [int(tok.item())]
for _ in range(80):
    logits = model(mx.array([[answer_ids[-1]]]), cache=cache)
    tok = sampler(logits[:, -1, :])
    mx.eval(tok)
    tid = int(tok.item())
    if tid in tokenizer.eos_token_ids:
        break
    answer_ids.append(tid)

print("ANSWER:", tokenizer.decode(answer_ids))
