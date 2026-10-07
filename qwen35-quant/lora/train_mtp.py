"""Retrain a Qwen3.5 checkpoint's MTP head for the model it now sits on
(e.g. after a LoRA merge shifted the backbone's hidden states).

The backbone is frozen; the head (built like gptq_qwen35.py builds it, from
the checkpoint's own mtp.* tensors) learns what it does at inference: from
the final hidden state at t and the token at t+1, the token at t+2. Data:
the chat conversations the model will serve (rendered with its template)
plus wikitext / code chunks, so the head stays general. Top-1 accuracy of
the head on held-out conversations (teacher-forced: what greedy drafting
gets right) is measured before and after. The new head replaces the old
one's tensors in the checkpoint, everything else unchanged.

    python train_mtp.py --model MERGED --train sft.jsonl --train synth.jsonl --eval sft.eval.jsonl \\
        --text wikitext:/workspace/data/wiki.train.raw --text code:/workspace/data/code.txt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))
from gptq_qwen35 import build_mtp, mtp_tensors  # noqa: E402

from train_lora import as_template_input  # noqa: E402


def conversations(tok, path: str, max_len: int) -> list[torch.Tensor]:
    out = []
    for line in open(path):
        conv = json.loads(line)
        text = tok.apply_chat_template(as_template_input(conv), tools=conv.get("tools"), tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"][:max_len]
        if len(ids) > 16:
            out.append(torch.tensor(ids))
    return out


def text_chunks(tok, source: str, n: int, tokens: int) -> list[torch.Tensor]:
    text = open(source.split(":", 1)[1], encoding="utf-8", errors="ignore").read()
    stride = max(1, len(text) // n)
    out = []
    for i in range(n):
        ids = tok(text[i * stride: i * stride + tokens * 8], add_special_tokens=False)["input_ids"][:tokens]
        if len(ids) == tokens:
            out.append(torch.tensor(ids))
    return out


class Drafter(torch.nn.Module):
    """The head on top of a frozen backbone: hidden states and embeddings
    from the backbone, the rest trainable."""

    def __init__(self, model, head):
        super().__init__()
        self.model = model
        self.lm = model.model.language_model
        self.head = head

    def forward(self, ids: torch.Tensor):
        """Logits for positions 2..T-1 of ids [1, T] and their targets."""
        with torch.no_grad():
            hidden = self.lm(input_ids=ids, use_cache=False).last_hidden_state   # final, normed
            emb = self.lm.embed_tokens(ids[:, 1:])
        h = self.head
        x = h.fc(torch.cat([h.pre_fc_norm_embedding(emb.float()), h.pre_fc_norm_hidden(hidden[:, :-1].float())], -1))
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)[None]
        rope = self.lm.rotary_emb(x, pos)
        for layer in h.layers:
            x = layer(x, position_embeddings=rope, attention_mask=None, position_ids=pos)
        logits = self.model.lm_head(h.norm(x)[:, :-1].to(self.model.lm_head.weight.dtype)).float()
        return logits[0], ids[0, 2:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the checkpoint whose head is retrained (updated in place)")
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--text", action="append", default=[], help="name:path of plain text, mixed in")
    ap.add_argument("--text-chunks", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)
    path = Path(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    train = [x for f in args.train for x in conversations(tok, f, args.max_len)]
    train += [x for s in args.text for x in text_chunks(tok, s, args.text_chunks, 1024)]
    evals = conversations(tok, args.eval, args.max_len)
    print(f"train {len(train)} sequences, eval {len(evals)}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    old = mtp_tensors(path)
    # Next to the checkpoint, not in it: the quant pipeline copies its files.
    save_file({k: v.contiguous() for k, v in old.items()}, str(path.parent / f"{path.name}.mtp-before-retrain.safetensors"))
    head, _ = build_mtp(model.model.language_model, old)
    head = head.float().cuda().train()
    for p in head.parameters():
        p.requires_grad_(True)
    drafter = Drafter(model, head)

    def accuracy():
        head.eval()
        right = total = 0
        with torch.no_grad():
            for ids in evals:
                logits, target = drafter(ids[None].cuda())
                right += int((logits.argmax(-1) == target).sum())
                total += target.numel()
        head.train()
        return right / max(total, 1)

    print(f"head top-1 before: {accuracy():.4f}", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(train) / args.accum) * args.epochs
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    step, t0, running, count = 0, time.time(), 0.0, 0
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(epoch).shuffle(order)
        for j, idx in enumerate(order):
            logits, target = drafter(train[idx][None].cuda())
            loss = F.cross_entropy(logits, target) / args.accum
            loss.backward()
            running += float(loss.detach()) * args.accum
            count += 1
            if (j + 1) % args.accum == 0 or j + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{steps} loss {running / count:.4f} {time.time() - t0:.0f}s", flush=True)
                    running, count = 0.0, 0
        print(f"epoch {epoch} head top-1 {accuracy():.4f}", flush=True)

    # The head's tensors back into the checkpoint's shards, by their names.
    state = {"mtp." + k: v.detach().to(torch.bfloat16).cpu() for k, v in head.state_dict().items()}
    index = json.load(open(path / "model.safetensors.index.json"))["weight_map"]
    for shard in sorted({index[k] for k in state}):
        tensors = load_file(str(path / shard))
        for k in state:
            if index[k] == shard:
                assert tensors[k].shape == state[k].shape, k
                tensors[k] = state[k].contiguous()
        save_file(tensors, str(path / shard), metadata={"format": "pt"})
    print(f"MTP_TRAIN_DONE {len(state)} head tensors written", flush=True)


if __name__ == "__main__":
    main()
