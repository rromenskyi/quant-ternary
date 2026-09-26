"""Deterministic quantization metric for MiniMax Music 3's 8B LM: teacher
forcing. Sampled generation diverges at the first differing token, so WER on
a dozen songs measures seed luck, not the quantization. Here:

1. bf16 generates the songs once; every input the LM gets (the prompt's
   embeddings, then each frame's feedback embedding) is recorded;
2. the whole recorded sequence goes through the LM in one forward pass,
   bf16 and each quantized variant alike;
3. at the audio frames: relative error of the last hidden state (it's fed
   to the synthesis stage: `frame_hiddens` in the pipeline), top-1
   agreement and KL of the conditional semantic-token distribution.

    python minimax_tf_metric.py --songs <dir> --gptq <lm_gptq_corrected.safetensors> --out results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import ModularPipeline
from safetensors.torch import load_file

from minimax_quant_sim import fake_quant_

AUDIO_CODE_OFFSET, SEMANTIC_VOCAB = None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", required=True)
    ap.add_argument("--model", default="/workspace/mm3")
    ap.add_argument("--gptq", help="corrected LM weights from minimax_gptq_lm.py")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from diffusers.modular_pipelines.minimax_music3 import encoders as enc

    pipe = ModularPipeline.from_pretrained(args.model)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.to("cuda")
    lm = pipe.language_model
    backbone = lm.model

    recorded: list[torch.Tensor] = []

    def record(module, args_, kwargs):
        recorded.append(kwargs["inputs_embeds"].detach())

    handle = backbone.register_forward_pre_hook(record, with_kwargs=True)
    sequences = []
    for line in (Path(args.songs) / "songs.tsv").read_text().splitlines():
        if not line.strip():
            continue
        song, caption = line.split("\t", 1)
        lyrics = "\n".join(l.strip().lower() if l.strip().startswith("[") else l.strip()
                           for l in (Path(args.songs) / f"{song}.txt").read_text().splitlines() if l.strip())
        recorded.clear()
        with torch.no_grad():
            pipe(prompt=caption, lyrics=lyrics, audio_duration=args.duration,
                 generator=torch.Generator("cuda").manual_seed(1), output="audios")
        prompt_len = recorded[0].shape[1]
        sequences.append((song, torch.cat(recorded, dim=1), prompt_len))
        print(f"recorded {song}: {sequences[-1][1].shape[1]} positions ({prompt_len} prompt)", flush=True)
    handle.remove()

    mask = torch.ones(lm.config.vocab_size, dtype=torch.bool, device="cuda")
    mask[enc._AUDIO_CODE_OFFSET: enc._AUDIO_CODE_OFFSET + enc._SEMANTIC_VOCAB_SIZE] = False

    @torch.no_grad()
    def run():
        outs = []
        for song, embeds, prompt_len in sequences:
            h = backbone(inputs_embeds=embeds, use_cache=False).last_hidden_state[0, prompt_len - 1:]   # conditional row
            logits = lm.lm_head(h).float().masked_fill(mask, -float("inf"))
            outs.append((h.float(), torch.log_softmax(logits, -1)))
        return outs

    reference = run()
    original = {n: m.weight.detach().clone() for n, m in lm.named_modules()
                if isinstance(m, torch.nn.Linear) and "lm_head" not in n}

    def restore():
        with torch.no_grad():
            for n, m in lm.named_modules():
                if n in original:
                    m.weight.copy_(original[n])

    def score(label):
        rows = {}
        for (song, _, _), (h_ref, lp_ref), (h_q, lp_q) in zip(sequences, reference, run()):
            rel = ((h_q - h_ref).norm(dim=-1) / h_ref.norm(dim=-1).clamp_min(1e-8)).mean().item()
            cos = torch.nn.functional.cosine_similarity(h_q, h_ref, dim=-1).mean().item()
            top1 = (lp_q.argmax(-1) == lp_ref.argmax(-1)).float().mean().item()
            p = lp_ref.exp()
            kl = torch.where(p > 0, p * (lp_ref - lp_q), torch.zeros_like(p)).sum(-1).mean().item()
            rows[song] = {"hidden_rel_err": rel, "hidden_cos": cos, "top1_agree": top1, "kl": kl}
        mean = {k: sum(r[k] for r in rows.values()) / len(rows) for k in next(iter(rows.values()))}
        print(label, json.dumps({k: round(v, 5) for k, v in mean.items()}), flush=True)
        return {"mean": mean, "per_song": rows}

    results = {}
    for bits in (8, 4):
        restore()
        with torch.no_grad():
            for n, m in lm.named_modules():
                if n in original and m.weight.shape[1] % 64 == 0:
                    fake_quant_(m, bits, 64)
        results[f"rtn{bits}"] = score(f"RTN {bits}-bit")
    if args.gptq:
        restore()
        corrected = load_file(args.gptq)
        with torch.no_grad():
            for n, m in lm.named_modules():
                if n in corrected:
                    m.weight.copy_(corrected[n].to(m.weight.device, m.weight.dtype))
        results["gptq4"] = score("GPTQ 4-bit")
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
