"""The teacher's top-k next-token log-probabilities for every data shard.

    python teacher_topk.py --master masters/gemma-4-12B-qat-unq --data data --out teacher --k 32

For each data/<name>.npy (eval first, then train shards in order) writes
teacher/<name>.lp.npy (float16 [rows, seq, k], log-softmax over the full
vocabulary) and teacher/<name>.idx.npy (int32 token ids). The student and
teacher don't fit the GPU together, so the teacher runs once, ahead of
training. Resumable per shard: a shard whose two files exist is skipped;
files are written to .tmp and renamed. progress.json carries done/total
and the measured tokens/s.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ternary_lib import load_text_model, save_npy, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = [data / "eval.npy"] + sorted(data.glob("train_*.npy"))
    todo = [s for s in shards if not ((out / f"{s.stem}.lp.npy").exists() and (out / f"{s.stem}.idx.npy").exists())]
    prog = {"done": len(shards) - len(todo), "total": len(shards), "k": args.k, "tok_s": None}
    write_json(out / "progress.json", prog)
    if not todo:
        print(f"all {len(shards)} shards done")
        return

    model, _ = load_text_model(args.master)
    model.eval()
    for s in todo:
        rows = np.load(s)
        lps, idxs = [], []
        t0 = time.time()
        with torch.inference_mode():
            for i in range(0, len(rows), args.batch):
                ids = torch.from_numpy(rows[i: i + args.batch].astype(np.int64)).cuda()
                logits = model(input_ids=ids).logits.float()
                lp = torch.log_softmax(logits, dim=-1)
                v, ix = lp.topk(args.k, dim=-1)
                lps.append(v.half().cpu().numpy())
                idxs.append(ix.int().cpu().numpy())
                del logits, lp
        save_npy(out / f"{s.stem}.idx.npy", np.concatenate(idxs))
        save_npy(out / f"{s.stem}.lp.npy", np.concatenate(lps))
        prog["done"] += 1
        prog["tok_s"] = rows.size / (time.time() - t0)
        write_json(out / "progress.json", prog)
        print(f"{s.stem}: {prog['done']}/{prog['total']}  {prog['tok_s']:.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
