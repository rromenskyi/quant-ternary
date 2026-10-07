"""Base model vs base + LoRA on what the adapter is for (build_dataset.py's
held-out conversations), with the agent's system prompt and tools:

- first step: on each held-out goal, the model's first move -- a tool call
  that parses (known tool, an action of it, params an object), the same
  tool / action as the reference, or a plain reply where the reference has
  one;
- reflex: greetings and small talk with the tools available must get a
  plain reply, no call (the Nemotron LoRA's failure, nemotron-extreme-quant
  FINDINGS 2.5).

Sampling as the agent runs it (temperature 1.0, top_p 0.95), the model's
own thinking included.

    python eval_lora.py --model SNAPSHOT --adapter adapter/ --eval sft.eval.jsonl --out eval.json
    python eval_lora.py --backend mlx --model BASE_MLX --compare LORA_MLX --eval sft.eval.jsonl --out eval.json

The MLX backend compares the quantized models people run (and is much
faster than transformers' generation without the fast conv kernel).
"""

from __future__ import annotations

import argparse
import json
import re

from build_dataset import system_prompt


def as_template_input(conv: dict) -> list[dict]:
    messages = json.loads(json.dumps(conv["messages"]))
    for m in messages:
        for tc in m.get("tool_calls", []):
            if isinstance(tc["function"]["arguments"], str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    return messages

CALL = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>", re.DOTALL)
PARAM = re.compile(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", re.DOTALL)
REFLEX = ["привет", "спасибо!", "как дела?", "ты кто?", "hi", "thanks", "what can you do?", "ок, понял"]


def parse(text: str):
    """The first tool call of a reply: (tool, action, params) or ("", "", None) for a plain reply,
    or None for a malformed call."""
    answer = text.split("</think>", 1)[-1]
    m = CALL.search(answer)
    if not m:
        return None if "<tool_call>" in answer else ("", "", None)
    params = dict(PARAM.findall(m.group(2)))
    action = params.get("action", "").strip()
    try:
        p = json.loads(params.get("params", "{}") or "{}")
    except json.JSONDecodeError:
        return None
    return (m.group(1), action, p) if isinstance(p, dict) else None


def hf_generator(model, tok):
    import torch

    def generate(messages, tools, max_new=1536):
        prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=0.95)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    return generate


def mlx_generator(path):
    from mlx_lm import generate as mlx_generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(path)
    sampler = make_sampler(temp=1.0, top_p=0.95)

    def generate(messages, tools, max_new=1536):
        prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        return mlx_generate(model, tok, prompt, max_tokens=max_new, sampler=sampler)
    return generate


def evaluate(generate, evals, tools, valid, samples):
    first = {"valid": 0, "same": 0, "malformed": 0, "n": 0}
    for conv in evals:
        msgs = as_template_input(conv)
        ref = next(m for m in msgs[2:] if m["role"] == "assistant")
        ref_call = ref.get("tool_calls", [{}])[0].get("function") if ref.get("tool_calls") else None
        ref_key = (ref_call["name"], ref_call["arguments"].get("action")) if ref_call else ("", "")
        for _ in range(samples):
            got = parse(generate(msgs[:2], tools))
            first["n"] += 1
            if got is None:
                first["malformed"] += 1
                continue
            tool, action, _ = got
            ok = tool == "" or (tool in valid and action in valid[tool])
            first["valid"] += ok
            first["same"] += ok and (tool, action) == ref_key
    reflex = {"calls": 0, "n": 0}
    template = evals[0]["messages"][0]["content"]
    for text in REFLEX:
        msgs = [{"role": "system", "content": system_prompt(template, "2026-10-07")}, {"role": "user", "content": text}]
        for _ in range(samples):
            got = parse(generate(msgs, tools, max_new=768))
            reflex["n"] += 1
            reflex["calls"] += got is None or got[0] != ""
    return {"first_step": first, "reflex": reflex}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "mlx"], default="hf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", help="hf: a LoRA adapter to compare against the base")
    ap.add_argument("--compare", help="mlx: a second MLX model (e.g. the LoRA build) to compare")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    evals = [json.loads(line) for line in open(args.eval)]
    tools = evals[0]["tools"]
    valid = {t["function"]["name"]: set(re.findall(r"^\s+- (\w+):", t["function"]["description"], re.MULTILINE))
             for t in tools}
    results = {}
    if args.backend == "mlx":
        import mlx.core as mx

        for name, path in (("base", args.model), ("lora", args.compare)):
            if not path:
                continue
            mx.random.seed(0)
            results[name] = evaluate(mlx_generator(path), evals, tools, valid, args.samples)
            print(name, json.dumps(results[name]), flush=True)
    else:
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
        results["base"] = evaluate(hf_generator(model, tok), evals, tools, valid, args.samples)
        print("base", json.dumps(results["base"]), flush=True)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).eval()
            results["lora"] = evaluate(hf_generator(model, tok), evals, tools, valid, args.samples)
            print("lora", json.dumps(results["lora"]), flush=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
