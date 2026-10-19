"""Prototype: run a stacked-expert MLX MoE model with its routed experts
left on disk. Each SwitchGLU keeps an LRU cache of experts in memory; the
experts a step needs and doesn't have are read with pread straight from
the safetensors files (one contiguous slice per expert and tensor, page
cache bypassed), and the layer runs on a compact stack of just the
experts it needs, through the usual gather_qmm path.

  python offload_proto.py MODEL_DIR CACHE_SHARE PROMPT_FILE MAX_TOKENS [--full]

--full: the same model fully in memory, for the baseline.
"""
import fcntl, glob, json, os, struct, sys, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models import switch_layers

F_NOCACHE = 48
DTYPES = {"U32": (np.uint32, mx.uint32), "BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, mx.float16),
          "F32": (np.float32, mx.float32), "U8": (np.uint8, mx.uint8)}
POOL = ThreadPoolExecutor(8)
STATS = {"read_bytes": 0, "read_s": 0.0, "hits": 0, "misses": 0, "steps": 0}


def tensor_index(path):
    """name -> (file, absolute offset, nbytes, dtype, shape)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        for name, t in h.items():
            a, b = t["data_offsets"]
            out[name] = (f, 8 + n + a, b - a, t["dtype"], t["shape"])
    return out


class FileReader:
    def __init__(self):
        self.fds = {}

    def fd(self, f):
        if f not in self.fds:
            fd = os.open(f, os.O_RDONLY)
            fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[f] = fd
        return self.fds[f]

    def read(self, f, off, n):
        return os.pread(self.fd(f), n, off)


READER = FileReader()


class ExpertStore:
    """One SwitchGLU's experts: on disk, up to C of them in memory, each
    with its own arrays (no stacked array to update). Decoding computes
    each needed expert with quantized_matmul on its own arrays; a call with
    many tokens (a prompt) stacks just the experts it needs."""
    PARTS = [(p, k) for p in ("gate_proj", "up_proj", "down_proj") for k in ("weight", "scales", "biases")]

    def __init__(self, glu, prefix, index, share):
        self.glu = glu
        self.slices = {}
        for p, k in self.PARTS:
            name = f"{prefix}.{p}.{k}"
            if name not in index:
                continue
            f, off, nbytes, dt, shape = index[name]
            self.experts = shape[0]
            self.slices[(p, k)] = (f, off, nbytes // shape[0], dt, shape[1:])
            setattr(getattr(glu, p), k, mx.zeros((1,), dtype=DTYPES[dt][1]))
        self.capacity = max(1, int(self.experts * share))
        self.cache = OrderedDict()       # expert -> {part: array}, least recently used first
        self.q = {p: (getattr(glu, p).group_size, getattr(glu, p).bits, getattr(glu, p).mode) for p in ("gate_proj", "up_proj", "down_proj")}

    def _read_expert(self, e):
        # Bytes only here (a reader thread); arrays are made on the caller's.
        return e, {key: READER.read(f, off + e * stride, stride) for key, (f, off, stride, dt, shape) in self.slices.items()}

    def _arrays(self, raw):
        out = {}
        for key, data in raw.items():
            f, off, stride, dt, shape = self.slices[key]
            a = mx.array(np.frombuffer(data, dtype=DTYPES[dt][0]).reshape(shape))
            out[key] = a.view(mx.bfloat16) if dt == "BF16" else a
        return out

    def experts_for(self, used, keep):
        """The arrays of each of `used`; cached (LRU) when `keep`."""
        missing = [e for e in used if e not in self.cache]
        STATS["hits"] += len(used) - len(missing)
        STATS["misses"] += len(missing)
        got = {}
        if missing:
            t = time.perf_counter()
            raws = list(POOL.map(self._read_expert, missing))
            STATS["read_s"] += time.perf_counter() - t
            for e, raw in raws:
                got[e] = self._arrays(raw)
                STATS["read_bytes"] += sum(v[2] for v in self.slices.values())
        out = {e: self.cache[e] if e in self.cache else got[e] for e in used}
        for e in used:
            if e in self.cache:
                self.cache.move_to_end(e)
            elif keep:
                self.cache[e] = got[e]
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return out

    def qmm(self, x, arrays, p):
        gs, bits, mode = self.q[p]
        return mx.quantized_matmul(x, arrays[(p, "weight")], arrays[(p, "scales")], arrays.get((p, "biases")),
                                   transpose=True, group_size=gs, bits=bits, mode=mode)


STORES = {}
orig_call = switch_layers.SwitchGLU.__call__


def offloaded_call(self, x, indices):
    store = STORES.get(id(self))
    if store is None:
        return orig_call(self, x, indices)
    idx_np = np.array(indices)          # waits for the router: what's needed
    flat = idx_np.reshape(-1, idx_np.shape[-1])
    used = sorted(set(flat.reshape(-1).tolist()))
    if flat.shape[0] <= 4:
        # Decoding: each needed expert on its own arrays.
        arrays = store.experts_for(used, keep=True)
        xs = x.reshape(-1, x.shape[-1])
        outs = {}
        rows = []
        for t in range(flat.shape[0]):
            xt = xs[t:t + 1]
            row = []
            for e in flat[t].tolist():
                if (t, e) not in outs:
                    a = arrays[e]
                    h = self.activation(store.qmm(xt, a, "up_proj"), store.qmm(xt, a, "gate_proj"))
                    outs[(t, e)] = store.qmm(h, a, "down_proj")
                row.append(outs[(t, e)])
            rows.append(mx.concatenate(row, axis=0))
        y = mx.stack(rows)                        # [tokens, k, D]
        return y.reshape(*idx_np.shape, x.shape[-1])
    # A prompt: a stack of just the experts it needs, for once.
    arrays = store.experts_for(used, keep=len(used) <= store.capacity)
    for p, k in ExpertStore.PARTS:
        if (p, k) in store.slices:
            setattr(getattr(self, p), k, mx.stack([arrays[e][(p, k)] for e in used]))
    pos = {e: i for i, e in enumerate(used)}
    y = orig_call(self, x, mx.array(np.vectorize(pos.get, otypes=[np.uint32])(idx_np)))
    mx.eval(y)
    for p, k in ExpertStore.PARTS:
        if (p, k) in store.slices:
            setattr(getattr(self, p), k, mx.zeros((1,), dtype=DTYPES[store.slices[(p, k)][3]][1]))
    return y


FIRST = [None]


def main():
    path, share, prompt_file, max_tokens = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    full = "--full" in sys.argv
    sync_only = "--sync-only" in sys.argv
    t0 = time.perf_counter()
    model, tok = load(path, lazy=not (full or sync_only))
    if sync_only:
        def synced(self, x, indices):
            np.array(indices)
            return orig_call(self, x, indices)
        switch_layers.SwitchGLU.__call__ = synced
    if not (full or sync_only):
        index = tensor_index(path)
        for name, m in model.named_modules():
            if isinstance(m, switch_layers.SwitchGLU):
                STORES[id(m)] = ExpertStore(m, name, index, share)
                if FIRST[0] is None:
                    FIRST[0] = id(m)
        switch_layers.SwitchGLU.__call__ = offloaded_call
        mx.eval(model.parameters())     # everything but the routed experts
    load_s = time.perf_counter() - t0
    msgs = [{"role": "user", "content": open(prompt_file).read()}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompt = mx.array(tok.encode(text))[None]
    cache = make_prompt_cache(model)
    t = time.perf_counter()
    logits = model(prompt, cache=cache)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
    first = time.perf_counter()
    base = dict(STATS)
    n = 0
    for _ in range(max_tokens):
        logits = model(y[:, None], cache=cache)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        n += 1
        STATS["steps"] += 1
        if y.item() in (tok.eos_token_ids if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}):
            break
    end = time.perf_counter()
    decode_tokens = n
    dec = {k: STATS[k] - base[k] for k in ("read_bytes", "read_s", "hits", "misses", "steps")}
    class R: pass
    last = R(); last.prompt_tokens = prompt.shape[1]
    print(json.dumps({
        "model": os.path.basename(path.rstrip("/")), "mode": "full" if full else "full, router synced" if sync_only else f"offload {share:.0%}",
        "load_s": round(load_s, 1), "prompt_tokens": last.prompt_tokens, "prefill_s": round(first - t, 2),
        "decode_tokens": decode_tokens, "decode_tok_s": round(decode_tokens / (end - first), 2),
        "peak_GB": round(mx.get_peak_memory() / 1e9, 2), "stores": len(STORES),
        "decode_hit_rate": round(dec["hits"] / max(1, dec["hits"] + dec["misses"]), 3) if not full else None,
        "decode_read_MB_per_step": round(dec["read_bytes"] / max(1, dec["steps"]) / 1e6, 1) if not full else None,
        "decode_read_s_share": round(dec["read_s"] / max(1e-9, end - first), 2) if not full else None,
        "decode_steps": dec["steps"],
    }))


if __name__ == "__main__":
    main()
