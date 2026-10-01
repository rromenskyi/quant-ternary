#!/usr/bin/env python3
"""Render the eval questions to 16 kHz mono int16 WAVs with macOS `say`.

Usage: python make_eval_audio.py [--questions eval/questions.json] [--out eval/audio]

Deterministic for a given macOS voice set; the WAVs are small and committed
so a pod or another machine can reuse them without `say`.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(HERE / "eval/questions.json"))
    ap.add_argument("--out", default=str(HERE / "eval/audio"))
    ap.add_argument("--rate", type=int, default=16000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for q in json.loads(Path(args.questions).read_text()):
        wav = out / f"{q['id']}.wav"
        if wav.exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            aiff = Path(tmp) / "q.aiff"
            subprocess.run(["say", "-v", q["voice"], "-o", str(aiff), q["text"]], check=True)
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", f"LEI16@{args.rate}", "-c", "1", str(aiff), str(wav)],
                check=True,
            )
        print(wav)


if __name__ == "__main__":
    main()
