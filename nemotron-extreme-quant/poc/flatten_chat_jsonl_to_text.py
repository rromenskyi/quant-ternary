"""Flatten a messages-format SFT JSONL (system/user/assistant/tool turns,
optionally with OpenAI-style tool_calls) into plain text, one file, so it
can be used as an extra GPTQ calibration source via
gptq_stock_convert.py's --extra-calib-file (which just treats any input as
raw text -- this script is the generic bridge from "chat JSONL" to that,
kept separate so the calibration script itself doesn't need to know about
any particular chat/tool-call schema).

Usage:
    python poc/flatten_chat_jsonl_to_text.py --input sft_dataset_combined.jsonl --output sft_calib.txt
"""

from __future__ import annotations

import argparse
import json


def flatten_message(msg: dict) -> str:
    parts = []
    content = msg.get("content")
    if content:
        parts.append(content)
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        parts.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    n_conversations = 0
    n_turns = 0
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_conversations += 1
            for msg in record["messages"]:
                text = flatten_message(msg)
                if text:
                    fout.write(text)
                    fout.write("\n")
                    n_turns += 1

    print(f"{args.input}: {n_conversations} conversations, {n_turns} turns -> {args.output}")


if __name__ == "__main__":
    main()
