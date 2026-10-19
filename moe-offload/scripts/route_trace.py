"""Which experts each MoE layer picks, token by token, while a local model
generates: wraps SwitchGLU.__call__ (every MoE model in mlx-lm routes
through it) and records the indices. Prefill and decode kept apart."""
import json, sys, time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models import switch_layers

trace = []          # (layer_id, step, [[experts of token]...])
layer_ids = {}
step = [0]

orig = switch_layers.SwitchGLU.__call__
def traced(self, x, indices):
    lid = layer_ids.setdefault(id(self), len(layer_ids))
    idx = indices.reshape(-1, indices.shape[-1]).tolist()
    trace.append((lid, step[0], idx))
    return orig(self, x, indices)
switch_layers.SwitchGLU.__call__ = traced

model_path, out, prompt_file = sys.argv[1], sys.argv[2], sys.argv[3]
model, tok = load(model_path)
prompt = open(prompt_file).read()
msgs = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
n_layers = None
t = time.time()
# Steps: each forward call (prefill chunk or one decode token) bumps step
# when the first layer is seen again.
last_first = [None]
orig2 = traced
def stepping(self, x, indices):
    lid = layer_ids.setdefault(id(self), len(layer_ids))
    if lid == 0:
        step[0] += 1
    return orig2(self, x, indices)
switch_layers.SwitchGLU.__call__ = stepping
generate(model, tok, text, max_tokens=int(sys.argv[4]) if len(sys.argv) > 4 else 300, verbose=False)
print("seconds", round(time.time() - t, 1), "layers", len(layer_ids), "calls", len(trace), file=sys.stderr)
json.dump({"layers": len(layer_ids), "trace": trace}, open(out, "w"))
