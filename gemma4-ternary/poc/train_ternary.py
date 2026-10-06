"""Ternary distillation of Gemma 4 (text decoder), resumable.

    python train_ternary.py --master masters/gemma-4-12B-qat-unq --data data \\
        --teacher teacher --out run --tokens 50000000 --batch 2 --accum 4 --lr 3e-5

Every text-decoder Linear trains as a latent bf16 weight seen through a
ternary straight-through estimator on MLX's 2-bit grid (ternary_lib). The
loss is KL(teacher || student) over the teacher's top-k tokens (teacher_topk).
Embeddings (tied with the LM head), norms and the vision/audio embedders
stay frozen at the master's values.

Run state, all under --out:
  ckpt/step_NNNNNN/   params + SRAdamW moments in ~2 GB safetensors parts,
                      meta.json (step, data cursor, RNG) and DONE, written
                      to a tmp dir and renamed; the last --keep are kept.
  metrics.jsonl       one line per optimizer step, per eval and per
                      reference eval; on resume, lines past the checkpoint
                      are dropped so a curve never doubles back.
  status.json         what the dashboard reads: phase, step, ETA, last save.

Stop with SIGTERM or SIGINT (stop.sh): the run finishes the optimizer step,
saves and exits 0. Rerun the same command to continue. A fresh run first
evaluates the reference points (--refs: the QAT q4_0 grid, MLX 4-bit, and
2-bit round-to-nearest at the ternary size) and the ternary start, so the curve has its baselines.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ternary_lib import (SRAdamW, bake, load_text_model, ref_quantizer, set_quant_strength, ternarize,
                         TEXT_LINEAR, write_json)  # noqa: E402

STOP = False


def _on_signal(signum, _frame):
    global STOP
    STOP = True
    log(f"signal {signum}: saving after this step, then exiting")


def log(msg: str) -> None:
    try:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    except BrokenPipeError:  # tee went away (e.g. Ctrl-C); keep running to save
        pass


class Data:
    """Rows of every train shard with their teacher top-k, in a seeded
    permutation per epoch; position = global row cursor."""

    def __init__(self, data: Path, teacher: Path, seed: int):
        self.data, self.teacher, self.seed = data, teacher, seed
        self.shards = sorted(p.stem for p in data.glob("train_*.npy"))
        self.rows = np.load(data / f"{self.shards[0]}.npy", mmap_mode="r").shape[0]
        self.n = len(self.shards) * self.rows
        self._cache: dict[str, tuple] = {}
        self._perm_epoch, self._perm = -1, None

    def _shard(self, name):
        if name not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[name] = (np.load(self.data / f"{name}.npy", mmap_mode="r"),
                                 np.load(self.teacher / f"{name}.lp.npy", mmap_mode="r"),
                                 np.load(self.teacher / f"{name}.idx.npy", mmap_mode="r"))
        return self._cache[name]

    def batch(self, cursor: int, size: int):
        ids, lps, idxs = [], [], []
        for c in range(cursor, cursor + size):
            epoch, pos = divmod(c, self.n)
            if epoch != self._perm_epoch:
                self._perm = np.random.default_rng(self.seed + epoch).permutation(self.n)
                self._perm_epoch = epoch
            r = int(self._perm[pos])
            x, lp, ix = self._shard(self.shards[r // self.rows])
            ids.append(x[r % self.rows]); lps.append(lp[r % self.rows]); idxs.append(ix[r % self.rows])
        t = lambda a, dt: torch.from_numpy(np.stack(a).astype(dt)).cuda(non_blocking=True)
        return t(ids, np.int64), t(lps, np.float32), t(idxs, np.int64)


def kd_loss(logits, t_lp, t_idx):
    """KL(teacher || student) restricted to the teacher's top-k tokens."""
    s_lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, t_idx)
    return (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean()


@torch.no_grad()
def evaluate(model, data: Path, teacher: Path, batch: int, max_rows: int) -> dict:
    x = np.load(data / "eval.npy")[:max_rows]
    lp = np.load(teacher / "eval.lp.npy")[:max_rows]
    ix = np.load(teacher / "eval.idx.npy")[:max_rows]
    was = model.training
    model.eval()
    set_quant_strength(model, 1.0)  # always score the real ternary model
    kl = agree = nll = n = 0.0
    for i in range(0, len(x), batch):
        ids = torch.from_numpy(x[i: i + batch].astype(np.int64)).cuda()
        t_lp = torch.from_numpy(lp[i: i + batch].astype(np.float32)).cuda()
        t_ix = torch.from_numpy(ix[i: i + batch].astype(np.int64)).cuda()
        s = torch.log_softmax(model(input_ids=ids).logits.float(), dim=-1)
        kl += (t_lp.exp() * (t_lp - s.gather(-1, t_ix))).sum().item()
        agree += (s.argmax(-1) == t_ix[..., 0]).sum().item()
        nll -= s[:, :-1].gather(-1, ids[:, 1:, None]).sum().item()
        n += ids.numel()
        del s
    model.train(was)
    rows = len(x)
    return {"kl": kl / n, "top1": agree / n, "ppl": math.exp(nll / (n - rows))}


def lr_at(step, total, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, total - args.warmup)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def save_ckpt(out: Path, step: int, names, params, opt, meta: dict, keep: int, part_bytes: float) -> None:
    ck = out / "ckpt"
    ck.mkdir(exist_ok=True)
    tmp = ck / f"tmp_step_{step:06d}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    part, size, k = {}, 0, 0
    for n, p, m, v in zip(names, params, opt.m, opt.v):
        part[f"p.{n}"], part[f"m.{n}"], part[f"v.{n}"] = p.detach().contiguous(), m, v
        size += 3 * p.numel() * p.element_size()
        if size >= part_bytes:
            save_file(part, str(tmp / f"part_{k:04d}.safetensors"))
            part, size, k = {}, 0, k + 1
    if part:
        save_file(part, str(tmp / f"part_{k:04d}.safetensors"))
    meta = dict(meta, opt_t=opt.t, cuda_rng=torch.cuda.get_rng_state().tolist())
    (tmp / "meta.json").write_text(json.dumps(meta))
    (tmp / "DONE").write_text("")
    final = ck / f"step_{step:06d}"
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)
    done = sorted(d for d in ck.glob("step_*") if (d / "DONE").exists())
    for d in done[:-keep]:
        shutil.rmtree(d, ignore_errors=True)


def latest_ckpt(out: Path):
    done = sorted(d for d in (out / "ckpt").glob("step_*") if (d / "DONE").exists())
    return done[-1] if done else None


@torch.no_grad()
def load_ckpt(d: Path, names, params, opt) -> dict:
    idx = {n: i for i, n in enumerate(names)}
    for f in sorted(d.glob("part_*.safetensors")):
        with safe_open(str(f), framework="pt", device="cuda") as sf:
            for key in sf.keys():
                kind, n = key.split(".", 1)
                i = idx[n]
                {"p": params, "m": opt.m, "v": opt.v}[kind][i].copy_(sf.get_tensor(key))
    meta = json.loads((d / "meta.json").read_text())
    opt.t = meta["opt_t"]
    torch.cuda.set_rng_state(torch.tensor(meta["cuda_rng"], dtype=torch.uint8))
    return meta


@torch.no_grad()
def load_params(d: Path, names, params) -> int:
    idx = {n: i for i, n in enumerate(names)}
    n = 0
    for f in sorted(d.glob("part_*.safetensors")):
        with safe_open(str(f), framework="pt", device="cuda") as sf:
            for key in sf.keys():
                if key.startswith("p."):
                    params[idx[key[2:]]].copy_(sf.get_tensor(key))
                    n += 1
    if n != len(names):
        raise ValueError(f"{d}: {n} weights for {len(names)} ternary Linears")
    return n


def append_metric(out: Path, rec: dict) -> None:
    with open(out / "metrics.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def trim_metrics(out: Path, step: int) -> None:
    f = out / "metrics.jsonl"
    if not f.exists():
        return
    keep = [l for l in f.read_text().splitlines() if l and json.loads(l).get("step", 0) <= step]
    tmp = f.with_suffix(".tmp")
    tmp.write_text("".join(l + "\n" for l in keep))
    os.replace(tmp, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, required=True, help="train tokens for the whole run")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100, help="optimizer steps")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--init-ckpt", help="fresh run: start from this checkpoint's latent weights (optimizer and schedule start fresh)")
    ap.add_argument("--quant-warmup", type=int, default=0,
                    help="optimizer steps over which the ternary projection ramps in linearly (0: on from the start)")
    ap.add_argument("--group", type=int, default=128, help="MLX 2-bit group size (128: 2.25 bits/weight, 64: 2.5)")
    ap.add_argument("--pattern", default=TEXT_LINEAR, help="regex of the Linears to ternarize")
    ap.add_argument("--affine-pattern", help="of those, the Linears kept on the MLX affine grid instead "
                    "(hybrid, e.g. 'self_attn' for 4-bit attention)")
    ap.add_argument("--affine-bits", type=int, default=4)
    ap.add_argument("--affine-group", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=150, help="optimizer steps")
    ap.add_argument("--eval-rows", type=int, default=64)
    ap.add_argument("--save-every-min", type=float, default=45)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--part-gb", type=float, default=2.0)
    ap.add_argument("--refs", default="q4_0,rtn4g64,rtn2g128",
                    help="reference evals on a fresh run, comma-separated: q4_0 (the QAT grid), rtn<bits>g<group> (MLX affine), or ''")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out, data, teacher = Path(args.out), Path(args.data), Path(args.teacher)
    out.mkdir(parents=True, exist_ok=True)
    seq = json.loads((data / "meta.json").read_text())["seq"]
    per_step = args.batch * args.accum
    total_steps = args.tokens // (per_step * seq)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    status = {"phase": "starting", "total_steps": total_steps, "tokens_target": total_steps * per_step * seq,
              "tokens_per_step": per_step * seq, "pid": os.getpid()}
    write_json(out / "status.json", status)

    ck = latest_ckpt(out)
    status["phase"] = "loading"
    write_json(out / "status.json", status)
    model, _ = load_text_model(args.master)
    if ck is None and args.refs:
        done_refs = set()
        if (out / "metrics.jsonl").exists():
            done_refs = {json.loads(l).get("ref") for l in (out / "metrics.jsonl").read_text().splitlines() if l}
        todo = [r for r in args.refs.split(",") if r and r not in done_refs]
        rx = re.compile(args.pattern)
        names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and rx.search(n)]
        orig = {n: model.get_submodule(n).weight.detach().clone() for n in names} if todo else {}
        for ref in todo:
            status["phase"] = f"reference eval {ref}"
            write_json(out / "status.json", status)
            bake(model, names, ref_quantizer(ref))
            r = evaluate(model, data, teacher, args.batch, args.eval_rows)
            append_metric(out, {"ref": ref, "step": 0, **r})
            log(f"ref {ref}: KL {r['kl']:.4f} top-1 {r['top1']:.2%} ppl {r['ppl']:.2f}")
            with torch.no_grad():
                for n in names:
                    model.get_submodule(n).weight.copy_(orig[n])
        del orig
        gc.collect(); torch.cuda.empty_cache()

    lin_names = ternarize(model, args.group, args.pattern, args.affine_pattern, args.affine_bits, args.affine_group)
    for p in model.parameters():
        p.requires_grad_(False)
    names = [f"{n}.weight" for n in lin_names]
    params = [model.get_submodule(n).weight for n in lin_names]
    for p in params:
        p.requires_grad_(True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = SRAdamW(params, weight_decay=args.weight_decay)
    log(f"{len(params)} ternary Linears, {sum(p.numel() for p in params) / 1e9:.2f}B params; "
        f"{total_steps} optimizer steps of {per_step * seq} tokens")

    step, cursor = 0, 0
    if ck is not None:
        status["phase"] = f"resuming {ck.name}"
        write_json(out / "status.json", status)
        meta = load_ckpt(ck, names, params, opt)
        step, cursor = meta["step"], meta["cursor"]
        trim_metrics(out, step)
        log(f"resumed {ck.name}: step {step}, cursor {cursor}")
    else:
        trim_metrics(out, 0)
        if args.init_ckpt:
            n_loaded = load_params(Path(args.init_ckpt), names, params)
            log(f"initialized {n_loaded} weights from {args.init_ckpt}")
        r = evaluate(model, data, teacher, args.batch, args.eval_rows)
        append_metric(out, {"eval": True, "step": 0, "tokens": 0, **r})
        log(f"ternary start: KL {r['kl']:.4f} top-1 {r['top1']:.2%} ppl {r['ppl']:.2f}")

    ds = Data(data, teacher, args.seed)
    last_save = time.time()
    status.update(phase="training", last_save=last_save)
    t_hist: list[float] = []
    while step < total_steps and not STOP:
        t0 = time.time()
        lr = lr_at(step, total_steps, args)
        lam = min(1.0, (step + 1) / args.quant_warmup) if args.quant_warmup else 1.0
        set_quant_strength(model, lam)
        loss_sum = 0.0
        for _ in range(args.accum):
            ids, t_lp, t_ix = ds.batch(cursor, args.batch)
            cursor += args.batch
            loss = kd_loss(model(input_ids=ids).logits, t_lp, t_ix) / args.accum
            loss.backward()
            loss_sum += loss.item()
            del loss
        opt.step(lr)
        step += 1
        dt = time.time() - t0
        t_hist = (t_hist + [dt])[-50:]
        tok_s = per_step * seq / (sum(t_hist) / len(t_hist))
        append_metric(out, {"step": step, "tokens": step * per_step * seq, "loss": loss_sum, "lr": lr, "lam": lam,
                            "tok_s": tok_s, "t": time.time()})
        if step % args.eval_every == 0 or step == total_steps:
            r = evaluate(model, data, teacher, args.batch, args.eval_rows)
            append_metric(out, {"eval": True, "step": step, "tokens": step * per_step * seq, **r})
            set_quant_strength(model, lam)
            log(f"step {step}: eval KL {r['kl']:.4f} top-1 {r['top1']:.2%} ppl {r['ppl']:.2f}")
        if step % 10 == 0:
            log(f"step {step}/{total_steps} loss {loss_sum:.4f} lr {lr:.2e} lam {lam:.2f} {tok_s:.0f} tok/s")
        status.update(step=step, tok_s=tok_s, eta_s=(total_steps - step) * per_step * seq / tok_s)
        save_due = time.time() - last_save > args.save_every_min * 60
        if save_due or STOP or step == total_steps:
            status["phase"] = "saving"
            write_json(out / "status.json", status)
            t1 = time.time()
            save_ckpt(out, step, names, params, opt, {"step": step, "cursor": cursor}, args.keep, args.part_gb * 1e9)
            last_save = time.time()
            log(f"saved step {step} in {last_save - t1:.0f}s")
            status.update(phase="training", last_save=last_save, last_save_step=step)
        write_json(out / "status.json", status)

    status["phase"] = "stopped" if STOP and step < total_steps else "done"
    write_json(out / "status.json", status)
    log(f"{status['phase']} at step {step}")


if __name__ == "__main__":
    main()
