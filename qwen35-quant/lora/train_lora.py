"""LoRA on a Qwen3.5 checkpoint's language model from chat conversations
(build_dataset.py / synth.py: OpenAI messages + tools).

- Rendered with the checkpoint's own chat template (tool-call arguments as
  objects, as mlx_lm.server passes them), so training text = inference text.
- Loss on the assistant turns only: from after each turn's think block
  through its <|im_end|>. The data has no reasoning; the template's empty
  <think></think> stays out of the loss, so the model isn't taught to stop
  thinking.
- Adapters on the attention only by default (full attention q/k/v/o, the
  delta rule's in_proj_qkv / in_proj_z / out_proj): on Nemotron-Nano-4B,
  adapting attention and MLP together cost general coding skill
  (nemotron-extreme-quant FINDINGS).

    python train_lora.py --model SNAPSHOT --train sft.jsonl --eval sft.eval.jsonl --out adapter/
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

TURN = re.compile(r"<\|im_start\|>assistant\n(?:<think>\n.*?</think>\n\n)?(.*?<\|im_end\|>)", re.DOTALL)
TARGETS = {
    "attention": r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj))",
    "attention+mlp": r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)",
}


def as_template_input(conv: dict) -> list[dict]:
    messages = json.loads(json.dumps(conv["messages"]))
    for m in messages:
        for tc in m.get("tool_calls", []):
            if isinstance(tc["function"]["arguments"], str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    return messages


def encode(tok, conv: dict, max_len: int):
    text = tok.apply_chat_template(as_template_input(conv), tools=conv.get("tools"), tokenize=False)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_len:
        return None
    spans = [m.span(1) for m in TURN.finditer(text)]
    labels = [-100] * len(ids)
    for i, (a, _) in enumerate(offsets):
        if any(s <= a < e for s, e in spans):
            labels[i] = ids[i]
    if all(x == -100 for x in labels):
        return None
    return torch.tensor(ids), torch.tensor(labels)


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="attention", choices=sorted(TARGETS))
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=8192)
    args = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    train = [x for f in args.train for c in load(f) if (x := encode(tok, c, args.max_len))]
    evals = [x for c in load(args.eval) if (x := encode(tok, c, args.max_len))]
    print(f"train {len(train)} eval {len(evals)} (dropped over {args.max_len} tokens)", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    config = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                        target_modules=TARGETS[args.targets], task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(train) / args.accum) * args.epochs
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    def loss_of(ids, labels):
        out = model(input_ids=ids[None].cuda(), labels=labels[None].cuda())
        return out.loss

    def evaluate():
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for ids, labels in evals:
                k = int((labels != -100).sum())
                total += float(loss_of(ids, labels)) * k
                n += k
        model.train()
        return total / max(n, 1)

    print(f"eval loss before: {evaluate():.4f}", flush=True)
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(epoch).shuffle(order)
        running, count = 0.0, 0
        for j, idx in enumerate(order):
            ids, labels = train[idx]
            loss = loss_of(ids, labels) / args.accum
            loss.backward()
            running += float(loss) * args.accum
            count += 1
            if (j + 1) % args.accum == 0 or j + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step}/{steps} loss {running / count:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                    running, count = 0.0, 0
        print(f"epoch {epoch} eval loss {evaluate():.4f}", flush=True)
    model.save_pretrained(args.out)
    json.dump(vars(args), open(f"{args.out}/train_args.json", "w"), indent=2)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
