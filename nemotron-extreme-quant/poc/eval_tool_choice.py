"""Tool-choice eval against LLMTray's real chat tools (FINDINGS 2.5 follow-up).

Where eval_tool_reflex.py measures one tool (generate_image) on one prompt,
this sends LLMTray's full tool list and its default tool-use rule (both from
`LLMTray --dump-tool-definitions`, saved as llmtray_tools.json) through an
OpenAI-compatible server, and scores
  - reflex: a tool call on a prompt that needs none (greetings, small talk,
    knowledge questions, code), and
  - choice: whether the right tool is called when one is needed.

    python eval_tool_choice.py <model name on the server> [runs=1] [--url http://127.0.0.1:8765]
                               [--tools llmtray_tools.json] [--temp 0.7]

The server is LLMTray's (its proxy switches to the named model). Needs
enough free GPU memory for the model plus its prompt cache: a 17 GB model at
the edge of a 26 GB Mac's Metal limit ran out of memory mid-run (FINDINGS
2.5, "Measured with LLMTray's tools").
"""
import argparse, json, time, urllib.request
from pathlib import Path

CASES = [
    ("привет", None), ("hi there!", None), ("как дела?", None), ("расскажи анекдот", None),
    ("что такое фотосинтез в двух словах?", None), ("напиши функцию на python, которая переворачивает строку", None),
    ("спасибо, всё понятно", None), ("who wrote War and Peace?", None),
    ("сколько будет 3847 * 29?", "calculate"), ("what is 15% of 2450?", "calculate"),
    ("какое сегодня число?", "get_current_date"), ("what day of the week is it today?", "get_current_date"),
    ("который час в Токио?", "get_current_time_in_city"),
    ("найди в интернете, что нового в Swift 6", "web_search"),
    ("какие последние новости про Apple?", "news"),
    ("что сейчас обсуждают на Hacker News?", "hackernews"),
    ("что википедия говорит про Киев?", "get_wikipedia_summary"),
    ("какое население и столица у Японии?", "get_country_info"),
    ("какие государственные праздники в Украине в 2026 году?", "get_public_holidays"),
    ("сколько будет 100 долларов в евро по сегодняшнему курсу?", "convert_currency"),
    ("нарисуй кота в шляпе", "generate_image"),
]
SYSTEM = ("You are a helpful assistant running locally on the user's Mac. Answer in the language the user writes in. "
          "Be concise and direct; use Markdown (lists, code blocks) when it helps readability.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("runs", nargs="?", type=int, default=1)
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--tools", default=str(Path(__file__).with_name("llmtray_tools.json")))
    ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args()
    defs = json.load(open(a.tools))
    system = SYSTEM + "\n\n" + defs["tool_use_policy"]
    reflex = choice_ok = total_none = total_task = 0
    for prompt, expected in CASES:
        got = []
        for _ in range(a.runs):
            body = {"model": a.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "tools": defs["tools"], "temperature": a.temp, "max_tokens": 400, "stream": False}
            req = urllib.request.Request(f"{a.url}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            msg = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
            calls = [c["function"]["name"] for c in (msg.get("tool_calls") or [])]
            got.append(calls[0] if calls else None)
        if expected is None:
            total_none += a.runs
            reflex += sum(g is not None for g in got)
        else:
            total_task += a.runs
            choice_ok += sum(g == expected for g in got)
        print(f"{prompt[:50]:50} expected={expected} got={got}", flush=True)
    print(f"\n{a.model}: reflex {reflex}/{total_none}, right tool {choice_ok}/{total_task} (temp {a.temp}, runs {a.runs})")


if __name__ == "__main__":
    main()
