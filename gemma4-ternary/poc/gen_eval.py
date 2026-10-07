"""Free-generation checks for MLX chat models: what teacher-forced KL can't
show (facts, code that runs, Russian, tool calls, thoughts that end, loops,
multi-turn). Every check is scripted; the same prompts go to every model.

    python gen_eval.py ~/.lmstudio/models/roman220220/gemma-4-12B-it-ternary-pilot \\
        ~/.lmstudio/models/roman220220/gemma-4-12B-it-qat-mlx --json out.json

Greedy decoding. Reported per model: pass rate per category, the share of
thinking replies whose thought channel closed within --max-think tokens,
and a repetition score (the share of repeated 4-grams in a reply; loops
push it towards 1).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

# (category, prompt, check) -- check gets the visible answer (thought removed)
FACTS = [
    ("What is the capital of Australia? Answer with the city name only.", r"canberra"),
    ("What is the capital of Canada? Answer with the city name only.", r"ottawa"),
    ("Which planet is known as the Red Planet? One word.", r"mars"),
    ("What is the chemical symbol for gold? Answer with the symbol only.", r"\bau\b"),
    ("Who wrote 'Romeo and Juliet'? Name only.", r"shakespeare"),
    ("How many continents are there on Earth? Answer with a number.", r"\b7\b|seven"),
    ("What is the largest ocean on Earth? One or two words.", r"pacific"),
    ("In which country is the city of Kyoto? One word.", r"japan"),
    ("What is the boiling point of water at sea level in degrees Celsius? Number only.", r"\b100\b"),
    ("Which element has atomic number 1? One word.", r"hydrogen"),
]
MATH = [
    ("I have 3 apples and buy 2 bags with 4 apples each. How many apples do I have now? End with 'Answer: <number>'.", "11"),
    ("What is 17 * 23? End with 'Answer: <number>'.", "391"),
    ("Is 391 a prime number? End with 'Answer: yes' or 'Answer: no'.", "no"),
    ("A train travels 60 km/h for 2.5 hours. How many km does it travel? End with 'Answer: <number>'.", "150"),
    ("What is 15% of 240? End with 'Answer: <number>'.", "36"),
    ("If x + 7 = 19, what is x? End with 'Answer: <number>'.", "12"),
]
RUSSIAN = [
    ("Какая столица Франции? Ответь одним словом по-русски.", r"париж"),
    ("Сколько будет 12 умножить на 11? Ответь числом.", r"\b132\b"),
    ("Назови автора романа «Война и мир». Только имя и фамилию.", r"толст"),
    ("Переведи на русский: 'Good morning'. Только перевод.", r"доброе утро"),
]
# code: the reply's python block must define the function and pass the asserts
CODE = [
    ("Write a Python function `fib(n)` that returns the n-th Fibonacci number iteratively (fib(0)=0, fib(1)=1). "
     "Reply with one ```python code block only.",
     "assert fib(0)==0 and fib(1)==1 and fib(10)==55 and fib(20)==6765"),
    ("Write a Python function `is_palindrome(s)` that ignores case and non-alphanumeric characters. "
     "Reply with one ```python code block only.",
     "assert is_palindrome('A man, a plan, a canal: Panama') and not is_palindrome('hello')"),
    ("Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list without "
     "using sort(). Reply with one ```python code block only.",
     "assert merge_sorted([1,4,7],[2,3,9])==[1,2,3,4,7,9] and merge_sorted([],[1])==[1]"),
    ("Write a Python function `count_words(text)` returning a dict of lowercase word -> count (split on "
     "whitespace, strip .,!?). Reply with one ```python code block only.",
     "assert count_words('Hi hi, there!')=={'hi':2,'there':1}"),
]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {
        "type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]
TOOL_PROMPTS = [("What's the weather in Paris right now, in celsius?", "paris"),
                ("Is it raining in Tokyo? Check the weather.", "tokyo")]
MULTI = [  # (turns, check on the last answer)
    (["My name is Alex and I live in Lisbon.", "What city do I live in? One word."], r"lisbon"),
    (["Where is Sydney? One short sentence.", "And Paris? One short sentence."], r"france"),
    (["Remember the number 42.", "Now add 8 to the number I asked you to remember. Answer with a number."], r"\b50\b"),
]


def split_thought(text: str) -> tuple[str, str | None, bool]:
    """(visible answer, thought or None, thought closed)."""
    m = re.search(r"<\|channel>thought\n?(.*?)(<channel\|>|$)", text, re.S)
    if not m:
        return text, None, True
    closed = m.group(2) == "<channel|>"
    return text[m.end():] if closed else "", m.group(1), closed


def repetition(text: str) -> float:
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def run_code(answer: str, test: str) -> bool:
    m = re.search(r"```(?:python)?\n(.*?)```", answer, re.S)
    if not m:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(m.group(1) + "\n" + test + "\n")
    try:
        return subprocess.run([sys.executable, f.name], capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def chat(model, tok, messages, think: bool, max_tokens: int, tools=None) -> str:
    kw = {"tools": tools} if tools else {}
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=think, **kw)
    return generate(model, tok, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0), verbose=False)


def evaluate(path: str, max_answer: int, max_think: int) -> dict:
    model, tok = load(path)
    res: dict[str, list] = {k: [] for k in ("facts", "math_think", "russian", "code", "tools", "multi_turn")}
    reps, closed, n_think = [], 0, 0

    def ask(msgs, think, mx_):
        nonlocal closed, n_think
        out = chat(model, tok, msgs, think, mx_)
        ans, thought, ok = split_thought(out)
        if think:
            n_think += 1
            closed += ok
        reps.append(repetition(out))
        return ans, out

    for q, rx in FACTS:
        a, _ = ask([{"role": "user", "content": q}], False, max_answer)
        res["facts"].append(bool(re.search(rx, a.lower())))
    for q, want in MATH:
        a, _ = ask([{"role": "user", "content": q}], True, max_think)
        m = re.findall(r"answer:\s*\**\s*([\w.]+)", a.lower())
        res["math_think"].append(bool(m) and m[-1].strip(".") == want)
    for q, rx in RUSSIAN:
        a, _ = ask([{"role": "user", "content": q}], False, max_answer)
        res["russian"].append(bool(re.search(rx, a.lower())))
    for q, test in CODE:
        a, _ = ask([{"role": "user", "content": q}], False, 2 * max_answer)
        res["code"].append(run_code(a, test))
    for q, city in TOOL_PROMPTS:
        out = chat(model, tok, [{"role": "user", "content": q}], False, max_answer, tools=TOOLS)
        reps.append(repetition(out))
        ok = False
        m = re.search(r"<tool_call>(.*?)</tool_call>", out, re.S) or re.search(r"call:get_weather\{(.*?)\}", out, re.S)
        if m:
            body = m.group(1)
            ok = "get_weather" in out and city in body.lower()
        res["tools"].append(ok)
    for turns, rx in MULTI:
        msgs = []
        for t in turns:
            msgs.append({"role": "user", "content": t})
            a, _ = ask(msgs, False, max_answer)
            msgs.append({"role": "assistant", "content": a})
        res["multi_turn"].append(bool(re.search(rx, msgs[-1]["content"].lower())))
    out = {k: sum(v) / len(v) for k, v in res.items()}
    out["thought_closed"] = closed / n_think if n_think else None
    out["repetition"] = sum(reps) / len(reps)
    out["detail"] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--max-answer", type=int, default=256)
    ap.add_argument("--max-think", type=int, default=1536)
    ap.add_argument("--json")
    args = ap.parse_args()
    all_res = {}
    for p in args.models:
        r = evaluate(p, args.max_answer, args.max_think)
        name = p.rstrip("/").split("/")[-1]
        all_res["/".join(p.rstrip("/").split("/")[-2:])] = r  # <org>/<model>, no local paths
        print(f"{name}: " + "  ".join(f"{k} {v:.0%}" for k, v in r.items()
                                      if k not in ("detail", "repetition") and v is not None)
              + f"  repetition {r['repetition']:.2f}", flush=True)
    if args.json:
        json.dump(all_res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
