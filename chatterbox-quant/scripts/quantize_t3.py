"""Chatterbox Multilingual V3 for MLX with its T3 (the 0.5B Llama that turns
text into speech tokens) quantized to 8 bits; everything else stays as
mlx-community converted it (fp32). Also copies in the built-in voice
(conds.safetensors), which the multilingual MLX repo lacks.

    python quantize_t3.py <mlx-community/chatterbox-multilingual-v3 dir> \
        <conds.safetensors from mlx-community/chatterbox-fp16> <S3TokenizerV2 dir> <out dir>

Run with an mlx-audio that has the chatterbox multilingual v3 model (the
ipsupport-llc/mlx-audio fork, commit 67f89de, was used). The output loads
with mlx-audio's own loader: config.json carries "quantization" and the
loader quantizes T3 where it finds scales.
"""
import json
import os
import shutil
import sys

sys.modules["russian_text_stresser"] = None   # AGPL; never needed here

if len(sys.argv) != 5:
    sys.exit(__doc__)
src, conds, s3tok, out = sys.argv[1:5]
NEEDED = ["model.safetensors", "config.json", "tokenizer.json", "Cangjie5_TC.json"]
missing = [os.path.join(src, n) for n in NEEDED if not os.path.exists(os.path.join(src, n))]
missing += [p for p in (conds, os.path.join(s3tok, "model.safetensors")) if not os.path.exists(p)]
if missing:
    sys.exit("missing: " + ", ".join(missing))

import huggingface_hub   # noqa: E402

real = huggingface_hub.snapshot_download


def local(repo_id=None, *a, **k):
    # mlx-audio fetches the speech tokenizer by name: a local folder here.
    return s3tok if repo_id == "mlx-community/S3TokenizerV2" else real(repo_id, *a, **k)


huggingface_hub.snapshot_download = local

import mlx.core as mx   # noqa: E402
import mlx.nn as nn   # noqa: E402
from mlx.utils import tree_flatten   # noqa: E402
from mlx_audio.tts.utils import load_model   # noqa: E402

GROUP, BITS = 64, 8

os.makedirs(out, exist_ok=True)
model = load_model(src)
nn.quantize(model.t3.tfmr, group_size=GROUP, bits=BITS)
weights = {}
for prefix, module in (("ve.", model.ve), ("t3.", model.t3), ("s3gen.", model.s3gen)):
    for k, v in tree_flatten(module.parameters()):
        weights[prefix + k] = v
mx.save_safetensors(os.path.join(out, "model.safetensors"), weights, metadata={"format": "mlx"})
with open(os.path.join(src, "config.json")) as f:
    config = json.load(f)
# One global entry, as mlx-audio's chatterbox loader expects: it quantizes a
# module only where the weights carry scales (only T3's transformer here).
config["quantization"] = {"group_size": GROUP, "bits": BITS}
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
for name in ("tokenizer.json", "Cangjie5_TC.json"):
    shutil.copy(os.path.join(src, name), os.path.join(out, name))
shutil.copy(conds, os.path.join(out, "conds.safetensors"))
print("wrote %s: %.2f GB" % (out, os.path.getsize(os.path.join(out, "model.safetensors")) / 1e9))
