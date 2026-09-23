"""Pipeline state, read from what the pipelines leave on disk -- shared by
gemma4_mlx_pipeline.sh (skip finished steps) and pipeline_dashboard.py.

Sources of truth:
  - $LOG_DIR/pipeline.log: `PIPE <ts> <pipeline> <START|DONE|SKIP|FAIL> <step> ...`
    lines written by pipeline_lib.sh;
  - calibration progress: <corrected>/gptq_progress_<name>.json
    ({"done_layers": [...]}) -- the calibrate scripts' own resume files;
  - the checkpoint's config.json for how many layers each component has.

CLI (used by the shell pipeline):
    python pipeline_status.py component-done --snapshot DIR --corrected DIR \
        --component text --variant 26b        # exit 0 if fully calibrated
    python pipeline_status.py summary --work /workspace   # JSON, for humans
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Which progress files make up one component, per variant. The 26B text pass
# writes two: attention + dense MLP ("text") and the routed experts
# ("text_moe", only for layers with enable_moe_block -- every layer on 26B).
PROGRESS_FILES = {
    "26b": {"text": ["text", "text_moe"], "vision": ["vision"]},
    "e4b": {"text": ["text"], "vision": ["vision"], "audio": ["audio"]},
}
CONFIG_KEY = {"text": "text_config", "vision": "vision_config", "audio": "audio_config"}


def num_layers(config: dict, component: str) -> int | None:
    sub = config.get(CONFIG_KEY[component]) or {}
    return sub.get("num_hidden_layers")


def done_layers(corrected: Path, name: str) -> list[int]:
    p = corrected / f"gptq_progress_{name}.json"
    if not p.exists():
        return []
    try:
        return sorted(json.loads(p.read_text())["done_layers"])
    except (json.JSONDecodeError, KeyError):
        return []


def calibration_state(config: dict, corrected: Path, variant: str) -> dict:
    """{component: {"total": N, "passes": {name: [done layers]}, "complete": bool}}"""
    out = {}
    for component, names in PROGRESS_FILES[variant].items():
        total = num_layers(config, component)
        if total is None:
            continue
        passes = {n: done_layers(corrected, n) for n in names}
        complete = all(len(set(v) & set(range(total))) == total for v in passes.values())
        out[component] = {"total": total, "passes": passes, "complete": complete}
    return out


def parse_pipeline_log(text: str) -> dict:
    """{pipeline: {"steps": {step: {"status", "ts", "detail"}}, "order": [...],
    "status": overall, "started": ts}} -- latest run of each pipeline only
    (a new `START _pipeline` line resets that pipeline's steps)."""
    pipes: dict[str, dict] = {}
    for line in text.splitlines():
        parts = line.split(" ", 5)
        if len(parts) < 5 or parts[0] != "PIPE":
            continue
        _, ts, pipe, status, step = parts[:5]
        detail = parts[5] if len(parts) > 5 else ""
        if step == "_pipeline":
            if status == "START":
                pipes[pipe] = {"steps": {}, "order": [], "status": "running", "started": int(ts), "detail": detail}
            elif pipe in pipes:
                pipes[pipe]["status"] = "done" if status == "DONE" else "failed"
                pipes[pipe]["ended"] = int(ts)
            continue
        p = pipes.setdefault(pipe, {"steps": {}, "order": [], "status": "running", "started": int(ts), "detail": ""})
        if step not in p["steps"]:
            p["order"].append(step)
        prev = p["steps"].get(step, {})
        p["steps"][step] = {
            "status": status,
            "ts": int(ts),
            "started": int(ts) if status == "START" else prev.get("started"),
            "detail": detail or prev.get("detail", ""),
        }
    return pipes


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("component-done")
    c.add_argument("--snapshot", required=True)
    c.add_argument("--corrected", required=True)
    c.add_argument("--component", required=True)
    c.add_argument("--variant", required=True, choices=sorted(PROGRESS_FILES))
    s = sub.add_parser("summary")
    s.add_argument("--work", default="/workspace")
    args = ap.parse_args()

    if args.cmd == "component-done":
        config = json.loads((Path(args.snapshot) / "config.json").read_text())
        state = calibration_state(config, Path(args.corrected), args.variant)
        return 0 if state.get(args.component, {}).get("complete") else 1

    log = Path(args.work) / "logs" / "pipeline.log"
    print(json.dumps(parse_pipeline_log(log.read_text() if log.exists() else ""), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
