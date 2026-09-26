"""Drops the text encoder layers FLUX.2 klein never reads from a checkpoint.

klein conditions on the Qwen3 hidden states 9, 18 and 27 only; index 0 is
the embedding, so layers 27 and up (0-based) never affect the output. mflux
still builds all 36 layers and leaves the missing ones uninitialized; the
loader (LLMTray's runner) cuts the list to 27 right after loading, so they
are never evaluated. Output embeddings and the final image are bit-identical
(checked, same seed); the 8-bit text encoder goes 4.3 -> 3.3 GB.

    python klein_truncate_te.py <checkpoint dir> [--keep 27]

Rewrites the shards in place, through new files: mx.load maps the file
lazily, so saving over the one being read corrupts it (it did).
"""
import argparse, glob, json, os, re, struct

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--keep", type=int, default=27)
args = ap.parse_args()
te = os.path.join(args.checkpoint, "text_encoder")


def metadata(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__")


def dropped(key):
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
    return m is not None and int(m.group(1)) >= args.keep


weight_map = {}
for shard in sorted(glob.glob(os.path.join(te, "*.safetensors"))):
    kept = {k: v for k, v in mx.load(shard).items() if not dropped(k)}
    mx.eval(*kept.values())
    tmp = shard[: -len(".safetensors")] + ".tmp"
    mx.save_safetensors(tmp, kept, metadata=metadata(shard))  # mlx appends .safetensors
    weight_map.update({k: os.path.basename(shard) for k in kept})
    del kept
    os.replace(tmp + ".safetensors", shard)
index_path = os.path.join(te, "model.safetensors.index.json")
index = json.load(open(index_path))
index["weight_map"] = weight_map
json.dump(index, open(index_path, "w"), indent=2)
print(f"{len(weight_map)} text encoder tensors kept (layers 0..{args.keep - 1})")
