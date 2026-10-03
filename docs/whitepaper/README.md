# IPSupport quantization whitepaper

**[IPSupport-quantization-whitepaper.pdf](IPSupport-quantization-whitepaper.pdf)** — *Frontier Models on a Laptop Budget*: how IPSupport LLC quantizes language, vision, speech, image and music models for Apple Silicon, across all 23 public releases.

Every number in it comes from a project's `docs/FINDINGS.md` or model card in this repo; nothing is estimated.

## Build

```bash
python make_figures.py                                   # figures/*.pdf (matplotlib)
tectonic -X compile IPSupport-quantization-whitepaper.tex   # the PDF
```

`tectonic` (`brew install tectonic`) fetches the LaTeX packages it needs on first run. The text font is [Inter](https://rsms.me/inter/) (SIL Open Font License, `fonts/OFL-LICENSE.txt`), shipped in `fonts/` so the build is reproducible.

When a number changes in a FINDINGS file, change it here too: the figures' data sits in `make_figures.py`, each block citing its source.
