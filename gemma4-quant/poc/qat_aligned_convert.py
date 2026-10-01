"""Google's Gemma 4 QAT release to an MLX checkpoint on the QAT grid.

Google's `-qat-q4_0-gguf` is llama.cpp's q4_0 of the `-qat-q4_0-unquantized`
master weights, bit for bit (checked: every block's d and codes match). q4_0
(ggml-quants.c, quantize_row_q4_0_ref): blocks of 32; d = (the block's
value with the largest magnitude) / -8, stored as fp16; q = min(15,
(int)(x/d + 8.5)); x' = d * (q - 8). MLX's affine 4-bit with group_size 32
stores x' = scale * q + bias, so scale = d, bias = -8 d is the same grid.
The one difference: MLX keeps scales in the model's dtype, bf16, so d is
rounded from fp16 to bf16 (under 1/64 of a step per weight; fp16 scales
would make every quantized matmul return float32). A plain
`mlx_lm.convert -q` instead fits each group's own min/max, and
mlx-community's qat builds use group 64 with some layers at 8-bit: off the
grid the QAT trained for.

  - tensors matching --q4-0 (default: the text decoder's Linears, exactly
    the GGUF's Q4_0 tensors): q4_0 as above;
  - embeddings (--embed-pattern): RTN at --embed-bits (the GGUF has Q6_K);
  - every other module mlx-lm quantizes (vision/audio towers, never
    QAT-trained): RTN at --other-bits (0 = keep float);
  - the rest (norms, scalars) copied.
With --gguf, every q4_0 tensor's codes and scales are compared with the
GGUF's and the run fails on any difference.

  python qat_aligned_convert.py --hf-checkpoint-dir E2B-unq --output-dir E2B-qat-mlx \\
      --gguf E2B-gguf/gemma-4-E2B_q4_0-it.gguf
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from splice_common import is_dead_kv_shared, module_path_of, quantizable_module_paths

GROUP = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_LINEARS = (r"^model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj"
                r"|per_layer_input_gate|per_layer_projection)\.weight$")
EMBEDDINGS = r"^model\.language_model\.embed_tokens(_per_layer)?\.weight$"


def q4_0(w: torch.Tensor):
    """llama.cpp's q4_0 of `w` (rows, cols): (fp16 d (rows, cols/32),
    codes uint8 (rows, cols))."""
    rows, cols = w.shape
    x = w.to(DEVICE).float().reshape(rows, cols // GROUP, GROUP)
    idx = x.abs().argmax(dim=-1, keepdim=True)          # the first of the largest, as ggml's strict '<'
    d = torch.gather(x, -1, idx) / -8.0
    inv = torch.where(d != 0, 1.0 / torch.where(d == 0, torch.ones_like(d), d), torch.zeros_like(d))
    q = torch.clamp((x * inv + 8.5).to(torch.int32), max=15)   # C's (int8_t) cast truncates; x*id+8.5 >= 0.5
    return d.squeeze(-1).to(torch.float16), q.reshape(rows, cols).to(torch.uint8)


def to_mlx_affine(d16: torch.Tensor, q: torch.Tensor):
    """q4_0 (d, codes) -> MLX affine 4-bit (packed uint32, scales, biases)."""
    rows, cols = q.shape
    qi = q.to(torch.int64).reshape(rows, cols // 8, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=qi.device)
    packed = mx.array((qi << shifts).sum(dim=-1).cpu().numpy().astype(np.uint32))
    d = mx.array(d16.float().cpu().numpy()).astype(mx.bfloat16)
    return packed, d, d * -8.0


class GGUFCheck:
    """The GGUF's Q4_0 tensors by their HF name, to compare against."""

    NAMES = {r"self_attn\.q_proj": "attn_q", r"self_attn\.k_proj": "attn_k", r"self_attn\.v_proj": "attn_v",
             r"self_attn\.o_proj": "attn_output", r"mlp\.gate_proj": "ffn_gate", r"mlp\.up_proj": "ffn_up",
             r"mlp\.down_proj": "ffn_down", r"per_layer_input_gate": "inp_gate", r"per_layer_projection": "proj"}

    def __init__(self, path: str):
        import gguf
        self.tensors = {t.name: t for t in gguf.GGUFReader(path).tensors}
        self.q4_0 = {n for n, t in self.tensors.items() if t.tensor_type.name == "Q4_0"}
        self.checked = set()

    def gguf_name(self, hf: str) -> str | None:
        m = re.match(r"^model\.language_model\.layers\.(\d+)\.(.+)\.weight$", hf)
        if not m:
            return None
        for pat, short in self.NAMES.items():
            if re.fullmatch(pat, m.group(2)):
                return f"blk.{m.group(1)}.{short}.weight"
        return None

    def check(self, hf: str, d16: torch.Tensor, q: torch.Tensor) -> None:
        name = self.gguf_name(hf)
        if name not in self.q4_0:
            raise SystemExit(f"{hf}: no Q4_0 tensor {name} in the GGUF")
        raw = np.array(self.tensors[name].data).reshape(-1, 18)
        gd = raw[:, :2].copy().view(np.float16).reshape(-1)
        qs = raw[:, 2:]
        gq = np.concatenate([qs & 0x0F, qs >> 4], axis=1).reshape(-1)
        if not (np.array_equal(gd, d16.cpu().numpy().reshape(-1)) and np.array_equal(gq, q.cpu().numpy().reshape(-1))):
            raise SystemExit(f"{hf}: q4_0 differs from the GGUF's {name}")
        self.checked.add(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--q4-0", default=TEXT_LINEARS, metavar="REGEX", help="raw keys that get q4_0 (the QAT-trained Linears)")
    ap.add_argument("--embed-pattern", default=EMBEDDINGS, metavar="REGEX")
    ap.add_argument("--embed-bits", type=int, default=6)
    ap.add_argument("--other-bits", type=int, default=8, help="RTN bits for every other quantizable module (0 = keep float)")
    ap.add_argument("--other-group-size", type=int, default=64)
    ap.add_argument("--gguf", help="Google's q4_0 GGUF: every q4_0 tensor must match it")
    ap.add_argument("--raise-json", help="qat_sensitivity.py's JSON: modules of the --raise-budget-mb curve point go to --raise-bits")
    ap.add_argument("--raise-budget-mb", type=float)
    ap.add_argument("--raise-bits", type=int, default=8)
    ap.add_argument("--drop-kv-shared-dead", action="store_true")
    ap.add_argument("--shard-size-gb", type=float, default=4.0)
    args = ap.parse_args()

    src, out = Path(args.hf_checkpoint_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    files = sorted(src.glob("*.safetensors"))
    key_file, shapes = {}, {}
    for p in files:
        with safe_open(str(p), "pt") as f:
            for k in f.keys():
                key_file[k] = p
                sl = f.get_slice(k)
                shapes[k] = (list(sl.get_shape()), sl.get_dtype())
    keys = [k for k in key_file if not (args.drop_kv_shared_dead and is_dead_kv_shared(k, config))]
    quantizable = quantizable_module_paths(config)
    q4_re, embed_re = re.compile(args.q4_0), re.compile(args.embed_pattern)
    gguf = GGUFCheck(args.gguf) if args.gguf else None
    raised: set[str] = set()
    if args.raise_json:
        curve = json.loads(Path(args.raise_json).read_text())["curve"]
        point = next(c for c in curve if c["budget_mb"] == args.raise_budget_mb)
        raised = set(point["raised"])
        print(f"raising {len(raised)} modules to {args.raise_bits}-bit (the +{args.raise_budget_mb:g} MB point)", flush=True)

    def kind(k: str) -> str:
        shape, dtype = shapes[k]
        if not (k.endswith(".weight") and len(shape) == 2 and dtype in ("BF16", "F16", "F32")
                and module_path_of(k) in quantizable):
            return "copy"
        if module_path_of(k) in raised:
            return "raised"
        if q4_re.search(k) and shape[1] % GROUP == 0:
            return "q4_0"
        if embed_re.search(k):
            return "embed"
        return "other" if args.other_bits else "copy"

    nbytes = lambda k: math.prod(shapes[k][0]) * 2
    shards, cur, size = [], [], 0
    for k in keys:
        if cur and size + nbytes(k) > args.shard_size_gb * 1024**3:
            shards.append(cur)
            cur, size = [], 0
        cur.append(k)
        size += nbytes(k)
    if cur:
        shards.append(cur)

    weight_map, overrides, counts = {}, {}, {"q4_0": 0, "raised": 0, "embed": 0, "other": 0, "copy": 0}
    handles = {p: safe_open(str(p), "pt") for p in files}
    for i, shard in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        tensors: dict[str, mx.array] = {}
        for k in shard:
            t = handles[key_file[k]].get_tensor(k)
            base, how = k[: -len(".weight")], kind(k)
            counts[how] += 1
            if how == "q4_0":
                d16, q = q4_0(t)
                if gguf:
                    gguf.check(k, d16, q)
                tensors[k], tensors[base + ".scales"], tensors[base + ".biases"] = to_mlx_affine(d16, q)
            elif how in ("raised", "embed", "other"):
                bits = {"raised": args.raise_bits, "embed": args.embed_bits, "other": args.other_bits}[how]
                w = mx.array(t.to(torch.float32).numpy()).astype(mx.bfloat16)
                # The widest group that divides the row; none (31B's vision
                # MLP is 4304 wide): left float.
                group = next((g for g in (args.other_group_size, 32) if w.shape[-1] % g == 0), None)
                if group is None:
                    tensors[k] = w
                    counts[how] -= 1
                    counts["copy"] += 1
                else:
                    tensors[k], tensors[base + ".scales"], tensors[base + ".biases"] = mx.quantize(
                        w, group_size=group, bits=bits, mode="affine")
                    overrides[module_path_of(k)] = {"group_size": group, "bits": bits}
            else:
                arr = mx.array(t.to(torch.float32).numpy())
                tensors[k] = arr.astype({torch.bfloat16: mx.bfloat16, torch.float16: mx.float16}.get(t.dtype, mx.float32))
            for tk in (k, base + ".scales", base + ".biases"):
                if tk in tensors:
                    weight_map[tk] = name
        mx.eval(*tensors.values())
        # No {"format": "mlx"}: raw tensors (the audio tower's convs) keep
        # the checkpoint's layout, and mlx-lm's sanitize() only moves them
        # into MLX's when the file doesn't claim to be MLX already.
        mx.save_safetensors(str(out / name), tensors)
        print(f"  {name}: {len(tensors)} tensors", flush=True)
        del tensors

    covered = gguf.checked | {gguf.gguf_name(k) for k in keys if kind(k) == "raised"} - {None} if gguf else set()
    if gguf and covered != gguf.q4_0:
        raise SystemExit(f"GGUF Q4_0 tensors not produced here: {sorted(gguf.q4_0 - covered)[:8]}")
    total = sum((out / f).stat().st_size for f in set(weight_map.values()))
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
    for item in src.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and item.name != "model.safetensors.index.json":
            shutil.copy2(item, out / item.name)
    config["quantization"] = {"group_size": GROUP, "bits": 4, "mode": "affine", **overrides}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"q4_0 (QAT grid): {counts['q4_0']}{' (all match the GGUF)' if gguf else ''}; raised to {args.raise_bits}-bit: {counts['raised']}; "
          f"embeddings {args.embed_bits}-bit: "
          f"{counts['embed']}; other {args.other_bits}-bit: {counts['other']}; copied: {counts['copy']}; {total / 1e9:.2f} GB")
    print("QAT_ALIGNED_DONE")


if __name__ == "__main__":
    main()
