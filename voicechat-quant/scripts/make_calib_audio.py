#!/usr/bin/env python3
"""Build the GPTQ calibration set: spoken user turns rendered with macOS `say`.

The LLM of VoiceChat never sees text tokens alone -- its input at every 80 ms
frame is the fused (audio embedding + previous text/function token)
embedding -- so calibration has to be *speech*, streamed through the real
duplex session (pod: calib_capture.py). This script only makes the audio;
it is cheap and runs on the Mac. None of these prompts overlap eval/questions.json.

  python make_calib_audio.py --out calib/audio --n 256
"""

import argparse
import itertools
import json
import random
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Tessa", "Fred", "Ralph", "Albert"]

COUNTRIES = ["Germany", "Italy", "Spain", "Canada", "Australia", "India", "Mexico", "Kenya",
             "Norway", "Argentina", "Vietnam", "Poland", "Chile", "Turkey", "Ireland", "Peru"]
TOPICS = ["black holes", "the Roman Empire", "photosynthesis", "electric cars", "coffee",
          "the internet", "volcanoes", "jazz music", "honey bees", "the moon landing",
          "machine learning", "the Great Wall of China", "earthquakes", "vaccines",
          "chess", "the stock market", "rainbows", "dinosaurs", "solar panels", "sleep"]
TASKS = ["bake bread", "change a flat tire", "learn a new language", "save money",
         "start running", "fall asleep faster", "write a good email", "plant tomatoes",
         "make pancakes", "prepare for a job interview", "clean a laptop screen",
         "remember names", "reduce stress", "train a puppy", "brew green tea"]
FREE = [
    "Hi there, how are you doing today?",
    "Can you tell me a short joke?",
    "What's a good name for a black cat?",
    "I'm feeling a bit tired, any advice?",
    "Recommend a book for a long flight.",
    "What should I cook for dinner tonight?",
    "Thanks, that was really helpful.",
    "Could you repeat that more slowly please?",
    "What's the difference between weather and climate?",
    "Why is the sky blue?",
    "How far away is the sun?",
    "Give me three ideas for a weekend trip.",
    "What time zone is New York in?",
    "Is it better to rent or to buy a house?",
    "How many hours of sleep does an adult need?",
    "Explain what a credit score is.",
    "Tell me something interesting about octopuses.",
    "What's the best way to learn to play guitar?",
    "Help me plan a birthday party for my daughter.",
    "Which is heavier, a kilogram of feathers or a kilogram of steel?",
    "Can you set a timer for ten minutes?",
    "What's the weather usually like in London in April?",
    "I'd like to order a pizza, what toppings do you suggest?",
    "Translate good morning into Spanish.",
    "What is seventeen times six?",
    "Let's talk about movies. What's a classic I should watch?",
    "Sorry, I didn't catch that. Can you say it again?",
    "My phone battery drains quickly, what can I do?",
    "What are the symptoms of a cold?",
    "How do airplanes stay in the air?",
]
TEMPLATES = [
    ("What is the capital of {c}?", COUNTRIES),
    ("What language do people speak in {c}?", COUNTRIES),
    ("Tell me one interesting fact about {t}.", TOPICS),
    ("Can you explain {t} in simple words?", TOPICS),
    ("How do I {k}?", TASKS),
    ("What's the easiest way to {k}?", TASKS),
]


def prompts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    items = list(FREE)
    for tpl, fills in TEMPLATES:
        key = tpl[tpl.index("{") + 1]
        items += [tpl.format(**{key: f}) for f in fills]
    eval_texts = {q["text"].lower() for q in json.loads((HERE / "eval/questions.json").read_text())}
    items = [p for p in dict.fromkeys(items) if p.lower() not in eval_texts]
    rng.shuffle(items)
    return list(itertools.islice(itertools.cycle(items), n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "calib/audio"))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest = []
    for i, text in enumerate(prompts(args.n, args.seed)):
        voice = VOICES[i % len(VOICES)]
        rate = rng.choice([160, 175, 190, 205])
        wav = out / f"c{i:03d}.wav"
        manifest.append({"id": wav.stem, "text": text, "voice": voice, "rate": rate})
        if wav.exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            aiff = Path(tmp) / "c.aiff"
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} clips -> {out}")


if __name__ == "__main__":
    main()
