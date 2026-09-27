<!--
HF MODEL-CARD TEMPLATE (quant-ternary). Copy this into <project>/cards/<repo>.md
and fill the <PLACEHOLDERS>. Keeps every public card on one IPSupport brand.

Canonical links (do not diverge):
  IPSupport site  https://www.ipsupport.us
  LLMTray (name)  https://www.ipsupport.us/llmtray/
  LLMTray badges  download → https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg
                  stars    → https://github.com/ipsupport-llc/llmtray
  IPSupport Code  https://ipsupport-llc.github.io/ipsupport-code/

Rules:
  - License must NOT be freer than the base model's. Keep base_model in YAML.
    Always include a "## License" section with a "Modified from <base>: <what
    was changed>" modification notice (HF content policy: no misrepresentation).
  - Only the ipsupport-code LoRA models are CODE/agent models. Never label a
    chat/vision/image/music/drafter/plain-quant model as a coding model.
  - Pick ONE Block A variant. Always end with Block B before "## License".
-->
---
license: <apache-2.0 | other>
# if license: other →  license_name: <name>   license_link: LICENSE
base_model: <org/base-model>
pipeline_tag: <text-generation | image-text-to-text | text-to-image | image-to-image | text-to-audio>
tags:
  - mlx            # or gguf / mflux etc.
  - quantized
  - <arch>         # nemotron / gemma4 / z-image / flux2 / ace-step
  # + gptq, jang, lora, tool-calling, coding-agent, vision, audio, mtp, ...
---

<!-- Hero banner right after frontmatter, before the H1.
     CODE models → ipsupport-code-banner.png ; everything else → llmtray-banner.png
     (both live in cards_assets/ and are uploaded into each repo alongside README) -->
<p align="center">
  <img src="<llmtray-banner.png | ipsupport-code-banner.png>" alt="<LLMTray | IPSupport Code>" width="100%">
</p>

# <Human title: model, recipe, format>

<!-- ===== Block A — pick ONE, place right after the H1 ===== -->

<!-- A1. NON-CODE (chat/vision/image/music) -->
> ### ▶ Run it locally in [LLMTray](https://www.ipsupport.us/llmtray/)
> A free, native macOS app for local AI on Apple Silicon — chat, images,
> music, agents and an OpenAI-compatible API. Point it at this model; nothing
> leaves your Mac.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)

<!-- A2. CODE / AGENT (only the ipsupport-code LoRA models)
> ### ▶ A coding-agent model — built for [IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)
> This model is fine-tuned for **IPSupport Code**, a local AI coding agent for
> real repositories (analyze · fix · test · report). It works best paired with
> that agent; tool-calling is trained in (see "Tool calling is trained in"
> below). You can also run it locally in [LLMTray](https://www.ipsupport.us/llmtray/).
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)
-->

<!-- A3. GGUF build (ollama/llama.cpp — LLMTray runs MLX, so point MLX users to the sibling)
> ### ▶ GGUF build for ollama / llama.cpp
> This is the **GGUF** build (ollama / llama.cpp). If you use
> [LLMTray](https://www.ipsupport.us/llmtray/) — IPSupport's local AI app for Apple
> Silicon, which runs MLX — use the MLX sibling <org/mlx-sibling> instead.
>
> [![Download LLMTray](https://img.shields.io/badge/Download-LLMTray%20for%20Mac-2f7d4f?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ipsupport-llc/llmtray/releases/latest/download/LLMTray-Full.dmg)
> [![GitHub stars](https://img.shields.io/github/stars/ipsupport-llc/llmtray?style=for-the-badge&logo=github)](https://github.com/ipsupport-llc/llmtray)
-->

<ONE-PARAGRAPH "what is this and why": base model, what was done (recipe /
quant / fine-tune), what it's for. Then all technical sections: recipe tables,
measurements, usage, method/code — keep these accurate and specific.>

<!-- ===== Block B — always, right before "## License" ===== -->

## The IPSupport local-AI stack

Local-first AI tools for macOS by [IPSupport](https://www.ipsupport.us) — nothing
leaves your Mac.

- **[LLMTray](https://www.ipsupport.us/llmtray/)** — your local AI
  workstation for macOS: chat with local LLMs, generate and edit images, make
  music, run agents, and serve an OpenAI-compatible API. Downloads models from
  Hugging Face in-app, with per-model profiles.
- **[IPSupport Code](https://ipsupport-llc.github.io/ipsupport-code/)** — your
  AI coding agent for real repositories: analyze, fix, test, report.

<PER-CARD LINE — one of:
  chat/vision:  Runs in LLMTray as a chat model, with vision[ and audio].
  image:        LLMTray uses this model for in-chat image generation[ and editing] (it drives mflux under the hood).
  music:        LLMTray uses this model for in-chat music generation.
  drafter:      LLMTray pairs this drafter with the main model for faster local decoding.
  plain quant:  Runs in LLMTray as a local chat LLM.
  code model:   Run this model in LLMTray, or wire it into IPSupport Code as the agent's backend.
  GGUF:         This is the GGUF build (ollama / llama.cpp). For the MLX build LLMTray runs, see <org/mlx-sibling>.
>

## License

Licensed under the **<License Name>**, the same license as the base model — see [`LICENSE`](LICENSE).

Modified from [<org/base-model>](https://huggingface.co/<org/base-model>): <one sentence stating exactly what was changed — quantized to N-bit, fine-tuned, converted to MLX/GGUF, etc.>. The weights and configuration files in this repo are therefore modified versions of the original, not the original files.
