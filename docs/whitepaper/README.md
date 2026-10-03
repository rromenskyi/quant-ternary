# IPSupport quantization whitepaper

**[IPSupport-quantization-whitepaper.pdf](../../IPSupport-quantization-whitepaper.pdf)** (repo root) — *Any Model. Your Hardware. Your Data.*: how IPSupport LLC compresses and fine-tunes open models (language, vision, speech, image, music) to run on less hardware, faster, and on the client's own tasks, across all 23 public releases.

Every number in it comes from a project's `docs/FINDINGS.md` or model card in this repo; nothing is estimated.

## Build

```bash
python make_figures.py                                   # figures/*.pdf (matplotlib)
tectonic -X compile --outdir ../.. IPSupport-quantization-whitepaper.tex   # the PDF, into the repo root
```

`tectonic` (`brew install tectonic`) fetches the LaTeX packages it needs on first run. The text font is [Inter](https://rsms.me/inter/) (SIL Open Font License, `fonts/OFL-LICENSE.txt`), shipped in `fonts/` so the build is reproducible.

When a number changes in a FINDINGS file, change it here too: the figures' data sits in `make_figures.py`, each block citing its source.
