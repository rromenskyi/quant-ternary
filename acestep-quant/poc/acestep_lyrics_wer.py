"""How intelligible a generated song's vocals are: Whisper (mlx-whisper, large-v3-turbo) transcribes the
wav, and the word error rate against the lyrics that were asked for is the
score (section markers like [Verse] dropped). A rough, automatic stand-in
for listening -- lower is better; unsung or garbled vocals score near 1.

    python acestep_lyrics_wer.py <lyrics file> <wav> [<wav> ...]
"""
from __future__ import annotations

import json
import re
import sys


def words(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]", " ", text.lower())
    return re.findall(r"[a-z0-9']+", text)


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(1, len(ref))


def main() -> None:
    import mlx_whisper

    lyrics = words(open(sys.argv[1]).read())
    for wav in sys.argv[2:]:
        text = mlx_whisper.transcribe(wav, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="en")["text"]
        hyp = words(text)
        print(json.dumps({"wav": wav, "wer": round(wer(lyrics, hyp), 3), "transcript": text.strip()[:400]}))


if __name__ == "__main__":
    main()
