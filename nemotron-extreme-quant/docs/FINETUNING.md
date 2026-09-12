# Fine-tuning Nemotron: reference notes

Separate topic from the quantization work in `docs/RUNBOOK.md` — this is
about teaching a Nemotron model our own Java + infra conventions for a
future internal coding assistant. Nothing here has been executed yet;
these are reference notes from research so we don't have to re-derive
them later.

## Approach, if we do this

**Fine-tune (LoRA/QLoRA) + RAG, not either alone.**
- Fine-tune bakes in *style/defaults* (e.g. "our Postgres modules always
  use multi-AZ + this parameter group") — things the model should just
  "know" without being told every time.
- RAG supplies *current facts* at query time (actual up-to-date file
  contents) — fine-tuning alone goes stale the moment the repo changes;
  RAG doesn't.
- Train on the base **bf16** checkpoint, **merge** the LoRA delta into the
  weights afterward, and only then run the result through our existing
  quantization pipeline. Merging first means the deployed model's size/
  speed are unchanged from an unfine-tuned quantized model of the same
  bit-width/architecture — LoRA adds no parameters or inference overhead
  once merged.

**Dataset idea**: no PR/diff history needed for a "write new code in our
style" assistant — reverse-instruction generation from *current* code is
enough. Have a strong model read an existing file/module and write the
natural-language request a developer would give to produce *exactly this
code with its specific choices* (not a generic description), 2-3 phrasing
variants per artifact. Scrub secrets/internal identifiers first. Cover all
our standard patterns, not just one, to avoid overfitting to whichever
type is overrepresented. Rough scale for a style-tuning LoRA: ~5-20k
examples (~5-20M tokens), 2-3 epochs.

## Axolotl: ready-made NemotronH support

The open question going in was how to LoRA-target a hybrid Mamba2+
Attention+MoE architecture (non-`nn.Linear` mixer, MoE experts as 3D
tensors) — **Axolotl already has this solved**, with working example
configs for our exact models:

- `examples/nemotron-h/nemotron-3_5-lightning-30b-a3b-qlora.yaml` — our 30B-A3B model
- `examples/nemotron-h/120b-a12b-qlora.yaml` — Nemotron-3-Super-120B-A12B
- `examples/nemotron/nemotron-mini-4b-qlora.yaml` — Nemotron-Mini-4B (non-hybrid)
- Repo: https://github.com/axolotl-ai-cloud/axolotl
- Docs: https://docs.axolotl.ai/

### Architecture facts that drive the LoRA config (from Axolotl's README)

- Three block types per layer: Mamba2 (SSM), Attention (sparse — only a
  minority of layers), MoE.
- MLP activation is `relu2` (`mlp_hidden_act`), not the usual `hidden_act`.
- MoE experts store `up_proj`/`down_proj` as **3D `nn.Parameter` tensors**
  (`[num_experts, out_dim, in_dim]`), not `nn.Linear` modules — there is
  no `gate_proj`.

### Required config settings (already correct in the example YAMLs)

```yaml
lora_qkv_kernel: false   # attention lives in NemotronHBlock.mixer, not layer.self_attn
lora_o_kernel: false     # same reason
lora_mlp_kernel: false   # relu2 activation + 3D expert params, unsupported by this kernel

lora_target_modules:     # regular attention projections, LoRA works normally here
  - q_proj
  - k_proj
  - v_proj
  - o_proj

# To also fine-tune the MoE experts (not just attention), add:
# lora_target_parameters:
#   - up_proj
#   - down_proj
```

Requires `pip install mamba-ssm causal-conv1d` (fast CUDA kernels) —
mandatory for `sample_packing: true` and `context_parallel_size > 1`,
since only these kernels correctly reset SSM state at packed-sample
boundaries via `seq_idx` (the plain transformers fallback silently drops
it, corrupting state across samples).

## Model size / hardware, if we do this

Considered going with **Nemotron-3-Super-120B-A12B** instead of the 30B-
A3B we've been quantizing, on the reasoning that cost isn't the
constraint for this — QLoRA on 120B needs roughly a single 80GB GPU
minimum per Axolotl's own single-GPU example config, more realistically
2x80GB (H100 SXM or A100 PCIe) for headroom on context/parallelism and
to double as a serving node afterward via tensor-parallel inference
(vLLM/TGI). Nemotron-3-Ultra-550B has no Axolotl config — that scale is
NVIDIA's own NeMo Megatron-Bridge territory, a much bigger multi-node
undertaking, not "take a YAML and run it."

Nothing has been provisioned for this track — the only GPU currently in
play is the existing A100 pod used for the unrelated 30B stock-GPTQ
quantization work in `docs/RUNBOOK.md`.
