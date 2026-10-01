#!/usr/bin/env python3
"""Score vc_eval.py result dirs: user-ASR WER, reply keyword accuracy, reply
intelligibility (independent Whisper transcript of the reply audio vs. the
model's own text channel), plus the speed numbers from run.json.

  HF_HUB_OFFLINE=1 python score.py results/rtn/*/ --table results/rtn/table.md

Whisper transcripts are cached per result dir (whisper.json) so re-scoring is
instant. The ASR model is mlx-community/whisper-large-v3-turbo (1.6 GB).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + num_words(rest))
    if n < 10000 and n % 1000 == 0:
        return ONES[n // 1000] + " thousand"
    return str(n)


def normalize(text: str) -> list[str]:
    text = text.lower().replace("°", " degrees ").replace("%", " percent ")
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = re.sub(r"\d+", lambda m: " " + num_words(int(m.group())) + " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z' ]+", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def keyword_hit(text: str, alternatives_all: list[list[str]]) -> bool:
    norm = " " + " ".join(normalize(text)) + " "
    for alternatives in alternatives_all:
        if not any(" " + " ".join(normalize(a)) + " " in norm for a in alternatives):
            return False
    return True


def transcribe_dir(asr, rdir: Path, ids: list[str]) -> dict:
    cache_path = rdir / "whisper.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for qid in ids:
        if qid not in cache:
            res = asr.generate(str(rdir / f"{qid}.wav"), language="en", verbose=False)
            cache[qid] = res.text.strip()
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--questions", default=str(HERE / "eval/questions.json"))
    ap.add_argument("--asr", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--asr-processor", default="openai/whisper-large-v3-turbo",
                    help="HF repo with the tokenizer/processor files the MLX weights repo lacks")
    ap.add_argument("--table", default=None)
    args = ap.parse_args()
    questions = {q["id"]: q for q in json.loads(Path(args.questions).read_text())}

    from mlx_audio.stt import load

    asr = load(args.asr)
    if getattr(asr, "_processor", None) is None:
        from transformers import WhisperProcessor

        asr._processor = WhisperProcessor.from_pretrained(args.asr_processor)
    rows = []
    for d in args.dirs:
        rdir = Path(d)
        run_path = rdir / "run.json"
        if not run_path.exists():
            print(f"skip {rdir} (no run.json)", file=sys.stderr)
            continue
        run = json.loads(run_path.read_text())
        recs = [json.loads(p.read_text()) for p in sorted(rdir.glob("q*.json"))]
        whisper = transcribe_dir(asr, rdir, [r["id"] for r in recs])
        u_err = u_ref = r_err = r_ref = hits = 0
        per_q = []
        for r in recs:
            q = questions[r["id"]]
            ref = normalize(q["text"])
            ue = edit_distance(ref, normalize(r["user_transcript"]))
            text_words = normalize(r["assistant_text"])
            re_ = edit_distance(text_words, normalize(whisper[r["id"]]))
            hit = keyword_hit(r["assistant_text"], q["keywords"])
            u_err, u_ref = u_err + ue, u_ref + len(ref)
            r_err, r_ref = r_err + re_, r_ref + max(len(text_words), 1)
            hits += hit
            per_q.append({"id": r["id"], "user_wer": round(ue / len(ref), 3), "keyword": hit,
                          "reply_wer": round(re_ / max(len(text_words), 1), 3),
                          "text": r["assistant_text"], "whisper": whisper[r["id"]],
                          "latency_s": r["reply_latency_s"]})
        ms = run["ms_per_frame"]
        lat = [p["latency_s"] for p in per_q if p["latency_s"] is not None]
        row = {
            "variant": rdir.name,
            "n": len(recs),
            **{f"{k}_ms": ms.get(k) for k in ("perception", "rnnt", "llm", "tts", "codec", "rest", "total")},
            "p95_ms": run["frame_ms_p95"],
            "rtf": run["rtf"],
            "peak_gb": run["peak_memory_gb"],
            "param_gb": run["variant"].get("param_gb"),
            "user_wer": round(u_err / max(u_ref, 1), 3),
            "keyword_acc": round(hits / max(len(recs), 1), 3),
            "reply_wer": round(r_err / max(r_ref, 1), 3),
            "reply_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
            "gpu_busy_before": run["system_before"].get("gpu_util_pct_idle_samples"),
        }
        (rdir / "score.json").write_text(json.dumps({"summary": row, "per_question": per_q}, indent=1, ensure_ascii=False))
        rows.append(row)
        print(json.dumps(row))

    if args.table and rows:
        cols = ["variant", "n", "perception_ms", "rnnt_ms", "llm_ms", "tts_ms", "codec_ms", "rest_ms",
                "total_ms", "p95_ms", "rtf", "peak_gb", "param_gb", "user_wer", "keyword_acc", "reply_wer",
                "reply_latency_s", "gpu_busy_before"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
        Path(args.table).write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
