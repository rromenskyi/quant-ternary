"""llama-quantize --tensor-type flags for a qat_sensitivity.py budget point: the
raised mlx-lm modules as GGUF tensor names, at q8_0.

  python gguf_types.py E2B-sens.json 100   ->  --tensor-type blk.3.attn_k=q8_0 ...
"""
import json, re, sys
GGUF = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down", "per_layer_input_gate": "inp_gate", "per_layer_projection": "proj"}
curve = json.load(open(sys.argv[1]))["curve"]
point = next(c for c in curve if c["budget_mb"] == float(sys.argv[2]))
flags = []
for path in point["raised"]:
    m = re.search(r"layers\.(\d+)\.(.+)$", path)
    if not m or m.group(2) not in GGUF:
        sys.exit(f"no GGUF name for {path}")
    flags.append(f"--tensor-type blk.{m.group(1)}.{GGUF[m.group(2)]}=q8_0")
print(" ".join(flags))
