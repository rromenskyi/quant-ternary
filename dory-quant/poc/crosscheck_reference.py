"""One-off correctness gate, not part of the pipeline: builds a tiny random
Dory model in both the (already-tested) mlx_lm fork and this project's new
torch dory_reference.py, copies the *same* weights into both (splitting
mlx's fused qkvg_proj/gate_up_proj back into the checkpoint's separate
q/k/v/g_proj and gate/up_proj, the same split test_dory_sanitize_... in the
fork round-trips), and compares logits on the same input. Must pass before
any real calibration run -- this is the only thing standing between "I
believe my torch port is right" and "I checked".

    MLX_ENABLE_TF32=0 python crosscheck_reference.py
    # MLX_LM_FORK: a local ipsupport-llc/mlx-lm checkout, if mlx_lm isn't
    # already pip-installed from it in the current environment.
"""

import os
import sys

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_map_with_path

if fork := os.environ.get("MLX_LM_FORK"):
    sys.path.insert(0, fork)
from mlx_lm.models import dory as mdory

sys.path.insert(0, ".")
from dory_reference import DoryConfig, DoryModel

mx.random.seed(0)
H, K, d, I, hid = 4, 2, 32, 256, 128
args = mdory.ModelArgs(
    model_type="dory", vocab_size=1000, hidden_size=hid, intermediate_size=I,
    num_hidden_layers=16, num_attention_heads=H, num_key_value_heads=K, head_dim=d,
    sliding_window=16, n_input_layers=4, n_recurrent_layers=8, n_output_layers=4,
    n_recurrent_loops=2,
)
mmodel = mdory.Model(args)
# As a trained checkpoint stores them (multiples of initializer_range), not
# literal 1.0: left at ones, nGPT attention is so sharp (confirmed live, and
# already noted in the fork's test_dory_sliding_window_cache) that MLX's
# fused attention kernel and torch's SDPA round differently enough on
# near-tied softmax entries to disagree by order-1 after 2 recurrent loops --
# a test-harness artifact of degenerate scale values, not a bug in either
# implementation.
mmodel.update(tree_map_with_path(
    lambda p, v: mx.full(v.shape, args.initializer_range) if p.split(".")[-1] in ("sqk", "s_gate") else v,
    mmodel.parameters(),
))
mx.eval(mmodel.parameters())

cfg = DoryConfig(
    vocab_size=1000, hidden_size=hid, intermediate_size=I, num_hidden_layers=16,
    num_attention_heads=H, num_key_value_heads=K, head_dim=d, sliding_window=16,
    n_input_layers=4, n_recurrent_layers=8, n_output_layers=4, n_recurrent_loops=2,
    layer_types=args.layer_types, theta_sliding=args.theta_sliding, theta_full=args.theta_full,
    ngpt_alpha_init=args.ngpt_alpha_init, initializer_range=args.initializer_range,
)
tmodel = DoryModel(cfg)

def t(x):
    return torch.from_numpy(np.array(x, copy=True)).float()

tmodel.embeddings.weight.data.copy_(t(mmodel.backbone.embeddings.weight))
tmodel.lm_head.weight.data.copy_(t(mmodel.lm_head.weight))
tmodel.logit_scale.data.copy_(t(mmodel.logit_scale))

for i, (ml, tl) in enumerate(zip(mmodel.backbone.layers, tmodel.layers)):
    if ml.is_attn:
        tl.attn_alpha.data.copy_(t(ml.attn_alpha))
        mm, tm = ml.mixer, tl.mixer
        w = np.array(mm.qkvg_proj.weight)
        tm.q_proj.weight.data.copy_(t(w[: H * d]))
        tm.k_proj.weight.data.copy_(t(w[H * d : H * d + K * d]))
        tm.v_proj.weight.data.copy_(t(w[H * d + K * d : H * d + 2 * K * d]))
        tm.g_proj.weight.data.copy_(t(w[H * d + 2 * K * d :]))
        tm.o_proj.weight.data.copy_(t(mm.o_proj.weight))
        tm.sqk.data.copy_(t(mm.sqk))
        tm.s_gate.data.copy_(t(mm.s_gate))
    else:
        tl.mlp_alpha.data.copy_(t(ml.mlp_alpha))
        mm, tm = ml.mixer, tl.mixer
        w = np.array(mm.gate_up_proj.weight)
        tm.gate_proj.weight.data.copy_(t(w[:I]))
        tm.up_proj.weight.data.copy_(t(w[I:]))
        tm.down_proj.weight.data.copy_(t(mm.down_proj.weight))
        tm.suv_gate.data.copy_(t(mm.suv_gate))
        tm.suv_up.data.copy_(t(mm.suv_up))

tmodel.eval()

mx.random.seed(1)
B, T = 2, 11
x_mx = mx.random.randint(0, 1000, (B, T))
x_np = np.array(x_mx)
x_t = torch.from_numpy(x_np).long()

with torch.no_grad():
    out_t = tmodel(x_t).numpy()
out_m = np.array(mmodel(x_mx).astype(mx.float32))

diff = np.abs(out_t - out_m)
print("max abs diff", diff.max())
assert diff.max() < 1e-2, "torch reference does not match the mlx model"
print("OK: dory_reference.py matches mlx_lm.models.dory")
