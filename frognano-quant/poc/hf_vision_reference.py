"""HF transformers reference for frognano-quant/poc/check_mlx.py, run on the GPU pod:
compares the Qwen35ImageInputs preprocessing / mRoPE positions (copied in
below from the mlx-lm fork, numpy only) with HF's processor and
get_rope_index on one synthetic 640 x 480 image, and saves the bf16 vision
tower's features (ref_vision_feats.npy), the image and the prompt.

    python hf_vision_reference.py   # expects /root/frognano-bf16
"""
from typing import Tuple, List
import math, numpy as np, json, torch
from PIL import Image, ImageDraw

class Qwen35ImageInputs:
    """Qwen3.5 (qwen3_5) image preprocessing, prompt expansion, fusion and
    mRoPE positions. Preprocessing follows HF's Qwen2VLImageProcessor (no
    torch): resize so each side is a multiple of patch * merge (32) with the
    pixel count in [min_pixels, max_pixels], normalize, and cut into
    temporal x patch x patch patches (a still image is its own 2 frames).
    Each <|image_pad|> placeholder becomes h * w / merge^2 image tokens; their
    positions are (t, h, w) on the merged grid, starting after the text
    before them, and the text after an image continues from max + 1 (HF's
    get_rope_index)."""

    # The checkpoint allows up to 16.7 MP (16k tokens an image); a chat
    # doesn't need that, and every token is prefill time.
    MAX_PIXELS = 1_048_576

    def __init__(self, model, processor_config: dict):
        cfg = processor_config.get("image_processor", processor_config)
        self.model = model
        self.patch = int(cfg.get("patch_size", 16))
        self.merge = int(cfg.get("merge_size", 2))
        self.temporal = int(cfg.get("temporal_patch_size", 2))
        size = cfg.get("size", {})
        self.min_pixels = int(size.get("shortest_edge", cfg.get("min_pixels", 65536)))
        self.max_pixels = min(int(size.get("longest_edge", cfg.get("max_pixels", self.MAX_PIXELS))), self.MAX_PIXELS)
        self.mean = np.array(cfg.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.std = np.array(cfg.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.image_token_id = model.args.image_token_id

    def _resize(self, h: int, w: int) -> Tuple[int, int]:
        factor = self.patch * self.merge
        h_bar = max(factor, round(h / factor) * factor)
        w_bar = max(factor, round(w / factor) * factor)
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((h * w) / self.max_pixels)
            h_bar = max(factor, math.floor(h / beta / factor) * factor)
            w_bar = max(factor, math.floor(w / beta / factor) * factor)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (h * w))
            h_bar = math.ceil(h * beta / factor) * factor
            w_bar = math.ceil(w * beta / factor) * factor
        return h_bar, w_bar

    def _patches(self, image) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        from PIL import Image

        h, w = self._resize(image.height, image.width)
        image = image.convert("RGB").resize((w, h), Image.BICUBIC)
        x = (np.asarray(image, dtype=np.float32) / 255.0 - self.mean) / self.std  # [H, W, C]
        x = np.repeat(x.transpose(2, 0, 1)[None], self.temporal, axis=0)  # [T, C, H, W]
        gt, gh, gw = 1, h // self.patch, w // self.patch
        m, p, t, c = self.merge, self.patch, self.temporal, x.shape[1]
        x = x.reshape(gt, t, c, gh // m, m, p, gw // m, m, p)
        x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(gt * gh * gw, c * t * p * p), (gt, gh, gw)



def _mine(img, cfg):
    o = Qwen35ImageInputs.__new__(Qwen35ImageInputs)
    c = cfg
    o.patch=16; o.merge=2; o.temporal=2
    o.min_pixels=c["size"]["shortest_edge"]; o.max_pixels=min(c["size"]["longest_edge"], o.MAX_PIXELS)
    o.mean=np.array(c["image_mean"],dtype=np.float32); o.std=np.array(c["image_std"],dtype=np.float32)
    return o._patches(img)

img = Image.new("RGB", (640, 480), (30, 120, 200)); d = ImageDraw.Draw(img); d.rectangle([100,100,400,300], fill=(250,200,40)); d.ellipse([350,200,600,460], fill=(200,30,60))
cfg = json.load(open("/root/frognano-bf16/preprocessor_config.json"))
mine_pv, grid = _mine(img, cfg)
from transformers import AutoProcessor, AutoModelForImageTextToText
proc = AutoProcessor.from_pretrained("/root/frognano-bf16")
msgs=[{"role":"user","content":[{"type":"text","text":"Describe:"},{"type":"image"},{"type":"text","text":"Short."}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inp = proc(text=[text], images=[img], return_tensors="pt")
print("grid hf", inp["image_grid_thw"].tolist(), "mine", grid)
print("pv shapes", tuple(inp["pixel_values"].shape), mine_pv.shape, "max abs diff", float(np.abs(inp["pixel_values"].float().numpy()-mine_pv).max()))
m = AutoModelForImageTextToText.from_pretrained("/root/frognano-bf16", dtype=torch.bfloat16, device_map="cuda")
tt = (inp["input_ids"] == m.config.image_token_id).int().cuda()
pos, delta = m.model.get_rope_index(inp["input_ids"].cuda(), tt, inp["image_grid_thw"].cuda(), None, attention_mask=inp["attention_mask"].cuda())
ids = inp["input_ids"][0].tolist()
# mine: rebuild positions from the unexpanded prompt (one image_pad per image)
img_id = m.config.image_token_id
lh, lw = grid[1]//2, grid[2]//2
axes=[[],[],[]]; nxt=0; i=0
while i < len(ids):
    if ids[i]==img_id:
        for a in range(lh):
            for b in range(lw):
                axes[0].append(nxt); axes[1].append(nxt+a); axes[2].append(nxt+b)
        nxt += max(lh,lw); i += lh*lw; continue
    for ax in axes: ax.append(nxt)
    nxt += 1; i += 1
mine_pos = torch.tensor(axes)
print("positions equal:", torch.equal(pos[:,0].cpu(), mine_pos), "delta hf", delta.tolist(), "mine", nxt-len(ids))

