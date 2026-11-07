"""Smoke test for convert_mlx.py's Dory-specific fuse logic (gptq_tensors /
write_gptq_codes), on the tiny checkpoint smoke_test_gptq.py already
exercises gptq_dory.py against. Skips mlx_lm.convert.convert()'s own
load()/tokenizer plumbing (generic, already proven by every other model in
this pipeline) and calls mlx_lm.utils.quantize_model directly on an
in-memory dory.Model, so this only has to prove the part that's new here:
that q/k/v/g_proj's separately-calibrated (W, scale, bias, codes) fuse into
qkvg_proj's RTN-quantized shape correctly, and the GPTQ codes round-trip
through packing without producing NaNs or shape mismatches.

    python smoke_test_convert.py
    # MLX_LM_FORK: a local ipsupport-llc/mlx-lm checkout, if mlx_lm isn't
    # already pip-installed from it in the current environment.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten

if fork := os.environ.get("MLX_LM_FORK"):
    sys.path.insert(0, fork)
from mlx_lm.models import dory as mdory
from mlx_lm.utils import quantize_model

sys.path.insert(0, ".")
import gptq_dory
import smoke_test_gptq as st
from convert_mlx import gptq_tensors, write_gptq_codes

gptq_dory.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: st._FakeTok())


def main():
    work_root = Path(tempfile.mkdtemp(prefix="dory_convert_smoke_"))
    snap, calib, work, assembled, out = (
        work_root / "snap", work_root / "calib.txt", work_root / "work",
        work_root / "assembled", work_root / "mlx_out",
    )
    st.build_snapshot(snap)
    calib.write_text("the quick brown fox jumps over the lazy dog " * 50)
    sys.argv = ["gptq_dory.py", "--model", str(snap), "--work", str(work), "--recipe", "attn=8,mlp=6",
                "--calib", f"text:{calib}", "--device", "cpu", "--calib-chunks", "2",
                "--calib-chunk-tokens", "16", "--group-size", "32"]
    gptq_dory.main()

    sys.argv = ["assemble_checkpoint.py", "--model", str(snap), "--work", str(work), "--out", str(assembled)]
    import assemble_checkpoint
    assemble_checkpoint.main()

    args = mdory.ModelArgs(model_type="dory", **st.CFG)
    model = mdory.Model(args)
    weights = model.sanitize({k: mx.array(v.float().numpy()) for k, v in
                              __import__("safetensors.torch", fromlist=["load_file"]).load_file(
                                  assembled / "model.safetensors").items()})
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    recipe = json.load(open(assembled / "quant_recipe.json"))
    bits, gs = recipe["recipe"], recipe["group_size"]
    from convert_mlx import MLX_COMPONENTS, bits_for

    def predicate(path, module):
        if path.endswith("backbone.embeddings"):
            return {"group_size": gs, "bits": 8}
        if path.endswith("lm_head"):
            return {"group_size": gs, "bits": 8}
        for suffix, component in MLX_COMPONENTS.items():
            if path.endswith("." + suffix):
                return {"group_size": gs, "bits": bits_for(component, bits)}
        return False

    config = {"model_type": "dory"}
    qmodel, qconfig = quantize_model(model, config, gs, min(bits.values()), quant_predicate=predicate)
    mx.eval(qmodel.parameters())
    out.mkdir(parents=True)
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(qmodel.parameters())),
                        metadata={"format": "mlx"})
    shutil.copy(assembled / "quant_recipe.json", out / "quant_recipe.json")

    # Sanity: the fuse map covers every attention/mlp layer, nothing left over.
    exact = gptq_tensors(work, recipe)
    n_attn = sum(1 for t in args.layer_types if t != "mlp")
    n_mlp = sum(1 for t in args.layer_types if t == "mlp")
    assert len(exact) == 2 * n_attn + 2 * n_mlp, (len(exact), n_attn, n_mlp)
    print(f"gptq_tensors: {len(exact)} fused/passthrough tensors ({n_attn} attn + {n_mlp} mlp layers)")

    write_gptq_codes(out, work, recipe, gs)

    # Reload the GPTQ-exact MLX model, run it, check for NaN/Inf and the
    # right output shape -- the thing that would blow up if a fuse order or
    # a packed-code shape were wrong.
    qmodel2, _ = quantize_model(mdory.Model(args), config, gs, min(bits.values()), quant_predicate=predicate)
    qmodel2.load_weights(str(out / "model.safetensors"))
    mx.eval(qmodel2.parameters())
    x = mx.random.randint(0, st.CFG["vocab_size"], (2, 9))
    logits = qmodel2(x)
    arr = np.array(logits.astype(mx.float32))
    assert arr.shape == (2, 9, st.CFG["vocab_size"]), arr.shape
    assert np.isfinite(arr).all(), "non-finite logits after GPTQ-exact packing"
    print("forward pass OK, shape", arr.shape, "finite:", bool(np.isfinite(arr).all()))

    shutil.rmtree(work_root)
    print("CONVERT_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
