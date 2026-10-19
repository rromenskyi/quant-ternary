"""From a routing trace (route_trace.py): during decoding, how many of the
experts a token needs are already in a per-layer LRU cache of a given
size, so how much has to come off the SSD per token."""
import json, sys
from collections import OrderedDict

def main(path, experts, expert_mb, gbps):
    d = json.load(open(path))
    L = d["layers"]
    # Decode steps: a few tokens per call (one, or a speculative draft
    # checked with it: MTP); the step needs every expert any of them picked.
    per_layer = [[] for _ in range(L)]
    prefill_touched = [set() for _ in range(L)]
    tokens_in_steps = 0
    for lid, step, idx in d["trace"]:
        if len(idx) <= 8:
            per_layer[lid].append(sorted({e for t in idx for e in t}))
            if lid == 0: tokens_in_steps += len(idx)
        else:
            for t in idx:
                prefill_touched[lid].update(t)
    k = len(d["trace"][0][2][0])
    steps = len(per_layer[0])
    reuse = sum(len(set(a) & set(b)) for seq in per_layer for a, b in zip(seq, seq[1:])) / max(1, sum(len(seq) - 1 for seq in per_layer)) / k
    print(json.dumps({"layers": L, "experts": experts, "top_k": k, "decode_steps": steps, "tokens_checked_per_step": round(tokens_in_steps / max(1, steps), 2),
                      "prefill_experts_touched_share": round(sum(map(len, prefill_touched)) / (L * experts), 3),
                      "next_token_reuse": round(reuse, 3)}))
    for frac in [0.125, 0.25, 0.375, 0.5, 0.75]:
        cap = max(k, int(experts * frac))
        hits = misses = 0
        for seq in per_layer:
            lru = OrderedDict()
            for tok in seq:
                for e in tok:
                    if e in lru:
                        hits += 1; lru.move_to_end(e)
                    else:
                        misses += 1; lru[e] = True
                        if len(lru) > cap: lru.popitem(last=False)
        rate = hits / (hits + misses)
        mb_per_token = misses / steps * expert_mb
        print(json.dumps({"cache_share": frac, "cache_GB_of_experts": round(cap * L * expert_mb / 1000, 2),
                          "hit_rate": round(rate, 3), "SSD_MB_per_step": round(mb_per_token, 1),
                          "SSD_ms_per_step": round(mb_per_token / 1000 / gbps * 1000, 1)}))

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
