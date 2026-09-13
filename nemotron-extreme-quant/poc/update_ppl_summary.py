"""Append/update one model's result in the shared ppl_summary.json that
poc/pipeline_dashboard.py and poc/make_model_readme.py both read -- keeps
the pipeline, the dashboard, and the uploaded model cards showing the same
numbers without hand-editing JSON after every run (see docs/session_
findings_2026-09-11.md section 7p for how this file's numbers were derived
before this script existed).

Usage:
    python poc/update_ppl_summary.py --summary-path /root/ppl_summary.json \
        --name "Ours: GPTQ mixed_3_6 (positional)" --ppl 5.9216 --note "14GB, 3.548 bpw"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--ppl", type=float, required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    path = Path(args.summary_path)
    data = json.loads(path.read_text()) if path.exists() else {"models": []}
    data["models"] = [m for m in data["models"] if m["name"] != args.name]
    data["models"].append({"name": args.name, "ppl": args.ppl, "note": args.note})
    path.write_text(json.dumps(data, indent=2))
    print(f"Updated {path}: {args.name} -> PPL={args.ppl}")


if __name__ == "__main__":
    main()
