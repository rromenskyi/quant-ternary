"""Per-expert byte sizes of a local MLX MoE checkpoint (stacked experts):
how much one routed expert costs to read from disk, per layer and token."""
import json, struct, sys, glob, os, re, collections

def headers(path):
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        yield f, h

def main(path):
    cfg = json.load(open(os.path.join(path, "config.json")))
    tc = cfg.get("text_config", cfg)
    E = tc.get("num_experts") or tc.get("num_local_experts") or tc.get("n_routed_experts")
    K = tc.get("num_experts_per_tok") or tc.get("top_k_experts")
    expert_bytes = collections.Counter()   # per layer: bytes of all experts' tensors
    total = 0
    for f, h in headers(path):
        for name, t in h.items():
            b = t["data_offsets"][1] - t["data_offsets"][0]
            total += b
            if t["shape"] and t["shape"][0] == E and re.search(r"(switch|experts)", name):
                m = re.search(r"layers\.(\d+)\.", name)
                expert_bytes[int(m.group(1)) if m else -1] += b
    layers = len(expert_bytes)
    all_experts = sum(expert_bytes.values())
    per_expert_layer = all_experts / layers / E if layers else 0
    print(json.dumps({
        "model": os.path.basename(path.rstrip("/")), "experts": E, "top_k": K, "moe_layers": layers,
        "total_GB": round(total / 1e9, 2), "routed_experts_GB": round(all_experts / 1e9, 2),
        "dense_GB": round((total - all_experts) / 1e9, 2),
        "one_expert_one_layer_MB": round(per_expert_layer / 1e6, 3),
        "per_token_if_no_cache_MB": round(per_expert_layer * K * layers / 1e6, 1),
    }))

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
