"""The HF model card of a qat_mlx_pipeline.sh build (HF_CARD_TEMPLATE.md,
block A1 + B), from the numbers the pipeline measured -- one generator for
every size, so the cards agree with each other and with FINDINGS.

  python qat_card.py --model E2B --repo roman220220/gemma-4-E2B-it-qat-mlx --size-gb 4.04 \\
      --q4 149 --raised 126 --budget-mb 100 --ours 64.5,0.030,91.8 --master-ppl 66.1 \\
      --baseline "mlx-community/gemma-4-E2B-it-qat-4bit:4.33:66.2,0.067,87.6" \\
      --speed "this model=67.5" --speed "mlx-community qat-4bit=54.7" --out ../cards/gemma-4-E2B-it-qat-mlx.md
"""
import argparse


def row(label, size, metrics, ppl_column=True):
    ppl, kl, top1 = metrics
    return f"| {label} | {size} | {ppl} | {kl} | {top1}% |" if ppl_column else f"| {label} | {size} | {kl} | {top1}% |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)                 # E2B
    ap.add_argument("--repo", required=True)
    ap.add_argument("--size-gb", required=True)
    ap.add_argument("--q4", type=int, required=True)          # Linears left on the q4_0 grid
    ap.add_argument("--raised", type=int, required=True)      # Linears at 8-bit
    ap.add_argument("--budget-mb", required=True)
    ap.add_argument("--ours", required=True)                  # ppl,kl,top1
    ap.add_argument("--master-ppl", required=True)
    ap.add_argument("--baseline", action="append", default=[])  # repo:size:ppl,kl,top1
    ap.add_argument("--speed", action="append", default=[], metavar="LABEL=TOK/S",
                    help="decode speed rows on an M5, this model first (repeatable)")
    ap.add_argument("--speed-note", default="")
    ap.add_argument("--replaces", help="an earlier repo this one replaces, with why")
    ap.add_argument("--raised-note", default="", help="what the raised Linears are, from the scan")
    ap.add_argument("--no-audio", action="store_true", help="the model has no audio tower (26B, 31B)")
    ap.add_argument("--chat-only", action="store_true",
                    help="a chat-only checkpoint (12B, 31B): raw-text PPL is noise, the eval scores the text as a chat reply")
    ap.add_argument("--memory-note", default="", help="which Macs it fits")
    ap.add_argument("--float-note", default="", help="the weights left in float on purpose, for the Checked list")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = a.model
    towers = "vision" if a.no_audio else "vision and audio"
    modalities = "text + vision" if a.no_audio else "text + vision + audio"
    base = f"google/gemma-4-{m}-it"
    qat = f"google/gemma-4-{m}-it-qat-q4_0-unquantized"
    ours = a.ours.split(",")
    ppl_column = True
    rows = [row(f"**this model**", f"**{a.size_gb} GB**", [f"**{x}**" for x in ours], ppl_column)]
    if ppl_column:
        rows.insert(0, f"| QAT master weights, bf16 ([{qat}](https://huggingface.co/{qat})) | — | {a.master_ppl} | — | — |")
    for b in a.baseline:
        repo, size, metrics = b.split(":")
        rows.append(row(f"[{repo}](https://huggingface.co/{repo})", f"{size} GB", metrics.split(","), ppl_column))
    header = ("| model | size | PPL | KL to QAT master | top-1 agree |\n|---|---|---|---|---|" if ppl_column
              else "| model | size | KL to QAT master | top-1 agree |\n|---|---|---|---|")
    speed = ""
    if a.speed:
        lines = []
        for i, item in enumerate(a.speed):
            label, value = item.rsplit("=", 1)
            lines.append(f"| **{label}** | **{value}** |" if i == 0 else f"| {label} | {value} |")
        speed = f"""
### Speed (MacBook Air M5, mlx-lm, decode)

| build | tokens/s |
|---|---|
{chr(10).join(lines)}

{a.speed_note}
"""
    replaces = ""
    if a.replaces:
        replaces = f"\n## This replaces an earlier release\n\n{a.replaces}\n"
    if a.chat_only:
        measured_intro = """Wikitext-2 (test), 128 windows of 512 tokens, scored as the model's chat
reply (the chat template, the thinking channel closed). KL divergence and
top-1 agreement are measured to the bf16 QAT master weights, the model this
one reproduces."""
        how_to_read = """How to read it:
- **KL and top-1 measure faithfulness;** a lower KL is closer.
- Why as a chat reply: on raw text this checkpoint, Google's own master and
  q4_0 GGUF included, scores in the thousands -- it reads text outside a chat
  turn as its own reasoning. Framed as its reply, it's the model it is.
- PPL compares this build with its master, not with other models: instruct
  models' confidence differs by size."""
    else:
        measured_intro = """Raw-text perplexity on wikitext-2 (test), 128 windows of 512 tokens, BOS at
the start of each. KL divergence and top-1 agreement are measured to the bf16
QAT master weights, the model this one reproduces."""
        how_to_read = """How to read it:
- **KL and top-1 measure faithfulness;** a lower KL is closer.
- PPL a little *below* the master's is within the noise of a raw-text test.
- It also reflects how QAT works: the network was trained through the 4-bit
  weights, so the q4_0 model is the one that was trained, and the bf16 master
  is its shadow copy."""
    card = f"""---
license: apache-2.0
base_model: {qat}
pipeline_tag: image-text-to-text
tags:
  - mlx
  - gemma4
  - quantized
  - qat
  - multimodal
  - vision
{"" if a.no_audio else "  - audio"}
---

<p align="center">
  <img src="llmtray-banner.png" alt="LLMTray" width="100%">
</p>

# Gemma 4 {m} — Google's QAT on its exact q4_0 grid, MLX ({modalities})

> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

Google trained Gemma 4 {m} with quantization-aware training (QAT) for
llama.cpp's **q4_0**. This is that model in MLX, **on the very grid the QAT
trained for**. The text decoder's 4-bit weights are identical, bit for bit,
to Google's own [q4_0 GGUF](https://huggingface.co/google/gemma-4-{m}-it-qat-q4_0-gguf).
On top of that, the {a.raised} Linears that lose the most at 4-bit are kept
at 8-bit, which costs {a.budget_mb} MB. The {towers} {"tower is" if a.no_audio else "towers are"}
included.{(chr(10) + chr(10) + a.memory_note) if a.memory_note else ""}
{replaces}
## How it's made

- **The QAT grid, exactly.** Google's GGUF is q4_0 of the
  `-qat-q4_0-unquantized` master weights: every block's scale and every code
  matches. MLX's affine 4-bit with group 32 stores `scale·q + bias`; with
  `scale = d` and `bias = -8d` that is q4_0's grid itself. {a.q4} Linears are
  written this way, and each is checked against the GGUF at conversion. The
  one difference is that the fp16 scale is rounded to bf16, MLX's scale dtype:
  under 1/64 of a step per weight.
- **Why not the usual conversions:**
  - `mlx_lm.convert -q` refits every group's min/max;
  - mlx-community's qat builds use group 64, which spans two q4_0 blocks,
    and keep every MLP at 8-bit.

  Both move the weights off the grid the QAT trained for.
- **A few Linears at 8-bit, chosen by measurement.** Each of the text Linears
  was raised to 8-bit on its own and scored by the KL divergence it removes,
  against the bf16 QAT master weights, per MB it adds. The best
  {a.raised} fit in +{a.budget_mb} MB. {a.raised_note}
- **Everything else follows Google's own split:**
  - embeddings at 6-bit, as the GGUF's Q6_K;
  - the {towers} {"tower, which was" if a.no_audio else "towers, which were"} never QAT-trained, at 8-bit;
  - norms as they are.

## Measured

{measured_intro}

{header}
{chr(10).join(rows)}

{how_to_read}
{speed}
### Checked

- It answers in chat.
- It names the animal in a photo.
{"" if a.no_audio else "- It transcribes speech through the audio tower." + chr(10)}{("- No large weight is left unquantized but " + a.float_note + ".") if a.float_note else "- No large weight is left unquantized."}
- It loads in LLMTray's runtime ([ipsupport-llc/mlx-lm](https://github.com/ipsupport-llc/mlx-lm)).

## Usage

```bash
pip install "mlx-lm @ git+https://github.com/ipsupport-llc/mlx-lm.git"
```

```python
from mlx_lm import load, generate

model, tokenizer = load("{a.repo}")
messages = [{{"role": "user", "content": "What is the capital of France?"}}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=200))
```

{"Images go" if a.no_audio else "Images and audio go"} through `mlx_lm.multimodal` in the same fork, which is
what LLMTray uses.

## Method / code

[rromenskyi/quant-ternary/gemma4-quant](https://github.com/rromenskyi/quant-ternary/tree/main/gemma4-quant)
has the code and the full lab notes (`docs/FINDINGS.md`, "Gemma 4 QAT in MLX on the q4_0 grid"):
- `qat_mlx_pipeline.sh`: one pipeline for every Gemma 4 size;
- `qat_aligned_convert.py`;
- `qat_sensitivity.py`;
- `qat_eval.py`.

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

Runs in LLMTray as a chat model, with {towers.replace(" and ", " and ")}.

## License

Licensed under the **Apache License 2.0**, the same license as the base model — see [`LICENSE`](LICENSE).

Modified from [{qat}](https://huggingface.co/{qat}) (the QAT release of [{base}](https://huggingface.co/{base})): quantized to MLX. The text decoder's Linears are on Google's q4_0 grid; {a.raised} of them, the embeddings and the {towers} {"tower" if a.no_audio else "towers"} are at 6/8-bit. The weights and configuration files in this repo are therefore modified versions of the original, not the original files.
"""
    open(a.out, "w").write(card)


if __name__ == "__main__":
    main()
