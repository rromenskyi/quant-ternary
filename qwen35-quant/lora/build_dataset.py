"""SFT conversations for an ipsupport-code LoRA, from the agent's own traces.

ipsupport-code logs every step to ~/.config/ipsupport-code/traces.jsonl:
the user's goal, each assistant step (text, number of tool calls), each
call (tool, action, params), its observation and the final answer. One goal
becomes one conversation in the OpenAI format the agent sends (a captured
request gives the system prompt and the tool schemas):

  system, user (goal), assistant (tool_calls) / tool (observation) ...,
  assistant (final answer)

- Kept: goals that reached a final answer; goals answered without a tool
  stay too (tools available, a plain reply is right: the reflex the
  Nemotron LoRA picked up, nemotron-extreme-quant FINDINGS 2.5).
- Repaired: a malformed call (unknown tool or action, missing params,
  params as a JSON string...) and its error observation are left out; the
  corrected call after it stays.
- Dropped: goals with an agent error, sub-agents, or nothing after repair.
- The system prompt carries the goal's date and a neutral workspace path;
  long observations are cut.

    python build_dataset.py --traces ~/.config/ipsupport-code/traces.jsonl \\
        --request captured_request.json --out sft.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

MALFORMED = re.compile(
    r"missing required param|no tool named|unknown tool|no action given|unknown action|"
    r"belongs to tool|arrived as a json string|not its own tool",
    re.IGNORECASE,
)
WORKSPACE = "/Users/me/project"


def system_prompt(template: str, date: str) -> str:
    text = re.sub(r"Today is \d{4}-\d{2}-\d{2}\.", f"Today is {date}.", template)
    text = re.sub(r"your working directory is [^ ]+\.", f"your working directory is {WORKSPACE}.", text)
    return re.sub(r"You are the engine inside [^,]+,", "You are the engine inside ipsupport-code,", text)


def cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… ({len(text) - limit} more characters)"


def episodes(events: list[dict]):
    """(goal event, the events up to its final) per goal of a run."""
    goal, steps = None, []
    for e in events:
        if e["kind"] == "goal":
            goal, steps = e, []
        elif goal is not None:
            steps.append(e)
            if e["kind"] == "final":
                yield goal, steps
                goal, steps = None, []


def actions_of(tools: list) -> dict[str, set[str]]:
    """Each tool's actions, from its description ("  - name: {...}")."""
    return {t["function"]["name"]: set(re.findall(r"^\s+- (\w+):", t["function"]["description"], re.MULTILINE))
            for t in tools}


def conversation(goal: dict, steps: list[dict], template: str, tools: list, max_obs: int) -> dict | None:
    valid = actions_of(tools)
    tool_names = set(valid)
    if any(e["kind"] in ("error", "subagent", "subagent_done") for e in steps):
        return None
    messages = [
        {"role": "system", "content": system_prompt(template, goal["time"][:10])},
        {"role": "user", "content": goal["text"]},
    ]
    pending = None  # the assistant message whose calls are being collected
    n = 0
    i = 0
    while i < len(steps):
        e = steps[i]
        if e["kind"] == "tool_call":
            obs = next((o for o in steps[i + 1:] if o["kind"] == "observation"), None)
            bad = obs is not None and str(obs.get("is_error")) == "True" and MALFORMED.search(obs.get("content") or "")
            if bad:
                i = steps.index(obs) + 1
                continue
            if pending is None:
                pending = {"role": "assistant", "content": "", "tool_calls": []}
                messages.append(pending)
            n += 1
            call_id = f"call_{n}"
            params = e.get("params") or {}
            action = e.get("action") or ""
            if isinstance(params, str) or e["tool"] not in tool_names:
                return None   # a tool the agent no longer has
            if not action and len(valid[e["tool"]]) == 1:
                # The agent infers a tool's only action; taught explicitly.
                action = next(iter(valid[e["tool"]]))
            if action not in valid[e["tool"]]:
                return None   # a garbled call the agent repaired: not one to teach
            pending["tool_calls"].append({
                "id": call_id, "type": "function",
                "function": {"name": e["tool"], "arguments": json.dumps({"action": action, "params": params},
                                                                       ensure_ascii=False)},
            })
            if obs is not None:
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": cut(obs.get("content") or "", max_obs)})
                i = steps.index(obs) + 1
                pending = None
                continue
        elif e["kind"] == "assistant" and e.get("content") and int(e.get("tool_calls") or 0) > 0:
            # Text said along with the calls that follow.
            pending = {"role": "assistant", "content": e["content"], "tool_calls": []}
            messages.append(pending)
        elif e["kind"] == "final":
            text = (e.get("text") or "").strip()
            # The agent's stand-in when the model wrote no summary: not the model's words.
            if not text or text.startswith("(done — finished without"):
                return None
            messages.append({"role": "assistant", "content": text})
        i += 1
    messages = [m for m in messages if not (m["role"] == "assistant" and "tool_calls" in m and not m["tool_calls"])]
    if any(m["role"] == "assistant" and "finished without a written summary" in (m.get("content") or "")
           for m in messages):
        return None
    if messages[-1]["role"] != "assistant" or "tool_calls" in messages[-1]:
        return None
    return {"messages": messages, "tools": tools}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--request", required=True, help="a captured chat request of the agent (system prompt, tools)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-observation", type=int, default=4000)
    ap.add_argument("--eval-share", type=float, default=0.05)
    args = ap.parse_args()
    request = json.load(open(args.request))
    template = request["messages"][0]["content"]
    tools = request["tools"]
    runs: dict[str, list[dict]] = {}
    for line in open(Path(args.traces).expanduser()):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        runs.setdefault(e["run"], []).append(e)
    seen, out, stats = set(), [], {"episodes": 0, "kept": 0, "with_tools": 0, "duplicates": 0}
    for events in runs.values():
        for goal, steps in episodes(events):
            stats["episodes"] += 1
            conv = conversation(goal, steps, template, tools, args.max_observation)
            if conv is None:
                continue
            key = hashlib.sha256(json.dumps(conv["messages"][1:], ensure_ascii=False).encode()).hexdigest()
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            out.append(conv)
            stats["kept"] += 1
            stats["with_tools"] += any("tool_calls" in m for m in conv["messages"])
    random.Random(0).shuffle(out)
    n_eval = max(1, int(len(out) * args.eval_share))
    path = Path(args.out)
    with open(path, "w") as f:
        for c in out[n_eval:]:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(path.with_suffix(".eval.jsonl"), "w") as f:
        for c in out[:n_eval]:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(json.dumps({**stats, "train": len(out) - n_eval, "eval": n_eval}))


if __name__ == "__main__":
    main()
