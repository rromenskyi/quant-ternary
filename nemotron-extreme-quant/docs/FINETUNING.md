# Fine-tuning Nemotron: the ipsupport-code LoRA pipeline

Separate topic from the quantization work in `docs/RUNBOOK.md`, though the
two compose: this project's fine-tuning flow is always *train LoRA on the
bf16 checkpoint → merge → then run the merged model through the
quantization pipeline*, never quantize-then-fine-tune. Merging before
quantizing means the deployed model's size/speed match an unfine-tuned
quantized model of the same architecture — LoRA adds no parameters or
inference overhead once merged.

This fine-tune's actual purpose: fix tool-calling reliability failures
observed in real usage of
[ipsupport-code](https://github.com/ipsupport-llc/ipsupport-code), a local
terminal coding agent, running on quantized Nemotron models produced by
this project. Executed on both
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (MoE) and
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (dense) — the two architectures
need different LoRA target configs, documented below, and the dense model
run surfaced a real training bug (see `docs/FINDINGS.md`) that the MoE run
never hit.

## Tooling: Axolotl

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) has ready-made
NemotronH support (hybrid Mamba2 + Attention + MoE — a non-`nn.Linear`
mixer, MoE experts stored as 3D tensors), with working example configs:

- `examples/nemotron-h/nemotron-3_5-lightning-30b-a3b-qlora.yaml`
- `examples/nemotron/nemotron-mini-4b-qlora.yaml`
- Docs: https://docs.axolotl.ai/

Architecture facts that drive the LoRA config:

- Attention lives in `NemotronHBlock.mixer`, not `layer.self_attn` — and
  MLP uses `relu2` activation — so Axolotl's fused LoRA kernels
  (`lora_qkv_kernel`, `lora_o_kernel`, `lora_mlp_kernel`) don't apply here
  and must be explicitly disabled:

  ```yaml
  lora_mlp_kernel: false
  lora_qkv_kernel: false
  lora_o_kernel: false
  ```

- On the **MoE** model, `up_proj`/`down_proj` are 3D `nn.Parameter` tensors
  (`[num_experts, out_dim, in_dim]`), not `nn.Linear` modules — LoRA
  targets them via `lora_target_parameters`, not `lora_target_modules`:

  ```yaml
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
  lora_target_parameters:
    - up_proj
    - down_proj
  ```

- On the **dense** 4B model, there are no MoE experts at all —
  `up_proj`/`down_proj` are plain `nn.Linear`, so they go through the
  normal `lora_target_modules` path (no `lora_target_parameters` needed).
  See `docs/FINDINGS.md` for why targeting them *together with* attention
  turned out to matter a great deal for this specific model.

- Requires `pip install mamba-ssm causal-conv1d` (fast CUDA kernels) — the
  `transformers` reference fallback for Mamba2's chunk-scan tried to
  allocate a single 45GB tensor during the training backward pass on the
  4B model (forward-only calibration never hits this, so it's easy to miss
  until an actual training run OOMs). See `docs/FINDINGS.md` and
  `poc/setup_pod.sh`'s axolotl-venv setup for the install gotchas
  (`--no-build-isolation`, `--no-deps`, and a torch-version regression
  check — `mamba_ssm`'s own dependency resolution was observed to silently
  upgrade torch to an incompatible CUDA stack).

## Dataset

SFT conversations in OpenAI-style `{"messages": [...]}` chat format,
matching real ipsupport-code tool-calling schema (`role: assistant`
messages with a `tool_calls` array; `role: tool` messages with results).
Sourced from real usage sessions where the base model's tool-calling
failed (malformed JSON, wrong parameter names, etc.), plus synthetic
conversations covering the same failure patterns. See
`poc/flatten_chat_jsonl_to_text.py` for turning this into plain text for
GPTQ calibration mix-in after the LoRA is merged.

**A real gap found in this dataset** (documented in detail in
`docs/FINDINGS.md`): the original set was 100% tool-call-heavy, with zero
conversations where the correct response is "just answer directly, no
tool needed." This silently taught the model to never close its
`<think>` block for a plain reply, since the chat template only renders a
proper `<think>...reasoning...</think>` block when a message's
`reasoning_content` field is present and non-empty. Fixed by adding
~125 synthetic conversations (greetings, clarifying questions, plain
factual answers, and a handful of English tool-calling examples for
language balance) — each with a short, real `reasoning_content`, not an
empty string.

## 30B-A3B (MoE) recipe

```yaml
adapter: qlora
lora_r: 32
lora_alpha: 64
lora_dropout: 0        # required with lora_target_parameters -- axolotl's
                        # ParamWrapper path for 3D expert tensors doesn't
                        # support dropout, forced to 0
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
lora_target_parameters: [up_proj, down_proj]
```

**Merging requires `--merge_method legacy`**: axolotl's default
("memory_efficient") merge silently *drops* LoRA weights applied via
`lora_target_parameters` on this architecture — the fused expert tensors
are incompatible with its merge path. `legacy` merge correctly reports
"Applied LoRA to N/M tensors" instead of silently no-op'ing on the expert
weights. Confirmed by checking the actual number of touched tensors after
merge, not just that the merge command exited 0.

Result: `roman220220/Nemotron-3.5-Lightning-30B-A3B-JANG-GPTQ-ipsupport-code-lora`
(final quantized+merged model), adapter alone at
`roman220220/ipsupport-code-nemotron-lora`. Validated via real usage —
zero malformed tool calls across a real session, after the fix.

## Nemotron-3-Nano-4B (dense) recipe

First attempt used the same-shaped config as the 30B run (`lora_r=32`,
`lora_alpha=64`, targeting all of `q_proj/k_proj/v_proj/o_proj` +
`up_proj/down_proj` as plain `lora_target_modules`, since this model has
no MoE experts to need the parameter-targeting path). Training completed
cleanly and the `<think>`-closing fix (above) worked — but real usage
surfaced a much worse regression: the model started generating
non-compiling, syntactically broken C++ and occasionally degenerated into
a repetition loop, on a model that (per the base checkpoint, tested
separately) is otherwise fine at this.

Root-caused via a compile-and-run eval harness (not just reading the
output) to: **training `q/k/v/o_proj` and `up_proj/down_proj` LoRA
modules *together* breaks general code competency**, independent of any
downstream quantization. Ablating a jointly-trained adapter — zeroing
either half's `lora_B` matrices before merging and testing each half
alone — showed neither half individually reproduces the damage; only
training both together does, at any rank/epoch count tested. Full
methodology and numbers in `docs/FINDINGS.md`.

**Fix**: LoRA targets only `q_proj/k_proj/v_proj/o_proj` — dropping
`up_proj`/`down_proj` entirely restores code-generation quality to the
un-fine-tuned base model's own level:

```yaml
adapter: qlora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
```

`poc/run_4b_lora_retrain.sh` exposes `LORA_TARGET_MODULES`,
`LORA_R`/`LORA_ALPHA`/`LORA_DROPOUT`/`NUM_EPOCHS` as environment
variables specifically so this isn't a hardcoded assumption for future
models — test attention-only vs. full-module-set LoRA on any new
architecture before committing to one.

Merge: default (non-`legacy`) axolotl merge method works fine here (no
MoE experts to drop in the first place).

Result: `roman220220/NVIDIA-Nemotron-3-Nano-4B-JANG-GPTQ-ipsupport-code-lora`,
adapter alone at `roman220220/NVIDIA-Nemotron-3-Nano-4B-ipsupport-code-lora`
(private).

## Style/domain fine-tuning (not executed)

An earlier, separate idea explored before the tool-calling reliability
work took priority: fine-tuning on a team's own code style/conventions
(paired with RAG for up-to-date facts, since style bakes in but facts go
stale). Reverse-instruction generation from existing code (have a strong
model read a file and write the request that would produce *that exact
code*) was the proposed dataset approach, at a rough scale of ~5-20k
examples / 2-3 epochs. Nothing here was executed — noted only so the
groundwork doesn't need re-deriving if it becomes relevant again.
