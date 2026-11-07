"""Chatterbox Turbo's distilled S3Gen ("meanflow": 2 steps, no CFG) as a
drop-in decoder for Chatterbox Multilingual V3's speech tokens (both use
S3 v2 tokens, 25 Hz). Takes mlx-community/chatterbox-turbo-fp16's
model.safetensors, keeps its s3gen.* weights without the speech tokenizer
(the multilingual model's own S3Gen still conditions on a voice clip),
renames them to the module's names (sanitize), and writes one file:

    python extract_meanflow.py <chatterbox-turbo-fp16 dir> <out.safetensors>

Load it with mlx-audio's chatterbox_turbo S3Gen(meanflow=True) and
load_weights(..., strict=False): only computed buffers are left out.
Ship it in a subfolder of the model: mlx-audio loads every top-level
*.safetensors of a model folder.
"""
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_audio.tts.models.chatterbox_turbo.models.s3gen import S3Gen

BUFFERS = {"encoder.embed.pos_enc.pe", "encoder.up_embed.pos_enc.pe", "mel2wav.stft_window", "trim_fade"}

if len(sys.argv) != 3:
    sys.exit(__doc__)
src, out = sys.argv[1], sys.argv[2]
weights = mx.load(src + "/model.safetensors")
raw = {k[len("s3gen."):]: v for k, v in weights.items()
       if k.startswith("s3gen.") and not k.startswith("s3gen.tokenizer.")}
decoder = S3Gen(meanflow=True)
named = decoder.sanitize(raw)
params = set(dict(tree_flatten(decoder.parameters())))
missing = params - set(named)
assert missing == BUFFERS, sorted(missing - BUFFERS)
mx.save_safetensors(out, {k: v for k, v in named.items() if k in params}, metadata={"format": "mlx"})
print("wrote", out)
