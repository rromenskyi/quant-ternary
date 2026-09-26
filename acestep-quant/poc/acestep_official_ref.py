"""Reference runs through the official ACE-Step 1.5 pipeline
(github.com/ace-step/ACE-Step-1.5, its own MLX backends for the DiT and the
LM on a Mac), to know what the mlx-audio port should reach -- run with that
repo's environment (`uv sync` there), from its checkout:

    python acestep_official_ref.py --repo <checkout> --out <dir> --lyrics-file f \
        --caption "..." --name tag --seeds 1 2 3 [--lm acestep-5Hz-lm-0.6B] [--duration 30]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lyrics-file", required=True)
    ap.add_argument("--caption", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--lm", default="acestep-5Hz-lm-0.6B")
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--config-path", default="acestep-v15-turbo", help="DiT: acestep-v15-turbo, acestep-v15-sft, acestep-v15-xl-turbo, ...")
    ap.add_argument("--guidance", type=float, default=1.0, help="DiT CFG (non-turbo only; official default for base/sft: 7.0)")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler

    dit = AceStepHandler()
    msg, ok = dit.initialize_service(project_root=args.repo, config_path=args.config_path, device="auto",
                                     offload_to_cpu=False)
    assert ok, msg
    llm = LLMHandler()
    msg, ok = llm.initialize(checkpoint_dir=os.path.join(args.repo, "checkpoints"), lm_model_path=args.lm,
                             backend="mlx", device="auto", offload_to_cpu=False, dtype=None)
    assert ok, msg

    os.makedirs(args.out, exist_ok=True)
    lyrics = open(args.lyrics_file).read()
    for seed in args.seeds:
        params = GenerationParams(task_type="text2music", thinking=True, caption=args.caption, lyrics=lyrics,
                                  vocal_language="en", duration=args.duration, inference_steps=args.steps,
                                  guidance_scale=args.guidance, seed=seed)
        tmp = os.path.join(args.out, "_tmp")
        t0 = time.time()
        result = generate_music(dit, llm, params=params, config=GenerationConfig(batch_size=1, audio_format="wav"),
                                save_dir=tmp)
        took = time.time() - t0
        assert result.success, result.status_message
        dest = os.path.join(args.out, f"{args.name}_s{seed}.wav")
        shutil.move(result.audios[0]["path"], dest)
        shutil.rmtree(tmp, ignore_errors=True)
        with open(os.path.join(args.out, "official.jsonl"), "a") as f:
            f.write(json.dumps({"name": args.name, "seed": seed, "lm": args.lm, "dit": args.config_path,
                                "steps": args.steps, "guidance": args.guidance, "seconds": round(took, 1),
                                "caption": args.caption, "wav": dest}) + "\n")
        print(f"{dest} {took:.1f}s", flush=True)


if __name__ == "__main__":
    main()
