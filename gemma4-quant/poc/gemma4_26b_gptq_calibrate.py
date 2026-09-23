"""GPTQ calibration for google/gemma-4-26B-A4B-it (the MoE variant of Gemma 4)
-- no mlx dependency, meant to run on a rented CUDA pod (same calibrate/splice
split as the sibling E4B script: this half runs anywhere with torch+
transformers, the mlx-only splice step happens back on a Mac).

This is the MoE sibling of gemma4_gptq_calibrate.py (written for the dense-only
google/gemma-4-E4B-it). Read that script first -- it establishes the batched/
resumable calibration pattern this script reuses verbatim for the "vision"
component and for the text tower's plain nn.Linear submodules (attention +
dense mlp). The genuinely new part here is calibrating the routed MoE experts
(`Gemma4TextExperts`), which are NOT nn.Linear submodules -- they're raw
`nn.Parameter` 3D tensors ([E, out, in]) consumed inside a Python
per-expert loop, so they can't be hooked the way a normal Linear can.

Real submodule/weight names (confirmed directly against transformers==5.17.0's
installed modeling_gemma4.py source, and against the real checkpoint's
safetensors header for layers 0, 1, 5, 15, 28, 29 this session):
  text:   model.model.language_model.layers[i].self_attn.{q,k,v,o}_proj
          model.model.language_model.layers[i].mlp.{gate,up,down}_proj
              -- plain nn.Linear, EVERY layer (num_kv_shared_layers=0, so
              k_proj/v_proj/k_norm/v_norm are always real attributes -- but
              v_proj can be the *value* None (not merely absent) whenever
              `config.attention_k_eq_v and not is_sliding` is true for that
              layer: `Gemma4TextAttention.__init__` sets
              `self.v_proj = nn.Linear(...) if not self.use_alternative_attention
              else None`. `layer.get_submodule("self_attn.v_proj")` raises
              AttributeError in exactly this case (nn.Module.get_submodule
              rejects a resolved attribute that isn't an nn.Module instance,
              confirmed by reading torch's get_submodule source this
              session) -- so the sibling script's existing
              try/except-AttributeError skip already handles this correctly.
              A plain `hasattr(layer.self_attn, "v_proj")` check would NOT
              work here (the attribute exists and is merely `None`), so this
              script intentionally keeps the get_submodule/AttributeError
              pattern instead.
          model.model.language_model.layers[i].experts.{gate_up_proj,down_proj}
              -- raw nn.Parameter, shape [num_experts, out, in]; calibrated
              via the hook-based capture below, NOT via get_submodule/Linear
              hooking.
          model.model.language_model.layers[i].router.*
              -- NEVER touched (not even listed as a calibration target
              anywhere in this file): routing precision is disproportionately
              sensitive to perturbation, so this script leaves router.proj,
              router.scale, router.per_expert_scale, router.norm completely
              untouched, at full bf16 precision, forever.
  vision: model.model.vision_tower.encoder.layers[i].self_attn.{q,k,v,o}_proj.linear
          model.model.vision_tower.encoder.layers[i].mlp.{gate,up,down}_proj.linear
              -- Gemma4ClippableLinear always wraps a real nn.Linear at
              `.linear` regardless of `use_clipped_linears` (confirmed by
              reading Gemma4ClippableLinear.__init__: the clamp buffers are
              only created `if self.use_clipped_linears`, but `self.linear`
              is unconditional) -- this model's vision tower has
              use_clipped_linears=False (no clamp buffers), but the `.linear`
              submodule path is identical to the E4B sibling script's vision
              path, so it is reused completely verbatim.
  (no audio component -- audio_config is null for this variant.)

Forward-pass mechanics that motivate the MoE capture design below (all read
directly from transformers==5.17.0's Gemma4TextDecoderLayer.forward /
Gemma4TextExperts.forward / Gemma4TextRouter.forward this session):

  Gemma4TextDecoderLayer.forward (abbreviated):
      residual = hidden_states
      hidden_states = input_layernorm(hidden_states)
      hidden_states, _ = self_attn(hidden_states, ...)
      hidden_states = post_attention_layernorm(hidden_states)
      hidden_states = residual + hidden_states

      residual = hidden_states                      # <-- routing input
      hidden_states = pre_feedforward_layernorm(hidden_states)
      hidden_states = mlp(hidden_states)             # dense mlp, ALWAYS run

      if enable_moe_block:                           # true for EVERY layer
                                                       # of this model (global
                                                       # config flag)
          hidden_states_1 = post_feedforward_layernorm_1(hidden_states)
          hidden_states_flat = residual.reshape(-1, residual.shape[-1])
          _, top_k_weights, top_k_index = router(hidden_states_flat)
          hidden_states_2 = pre_feedforward_layernorm_2(hidden_states_flat)
          hidden_states_2 = experts(hidden_states_2, top_k_index, top_k_weights)
          ...

  Gemma4TextExperts.forward (abbreviated):
      for expert_idx in expert_hit:                  # expert_mask from top_k_index
          token_idx = ...
          current_state = hidden_states[token_idx]                     # gate_up_proj's calib INPUT
          gate, up = F.linear(current_state, gate_up_proj[expert_idx]).chunk(2, -1)
          current_hidden_states = act_fn(gate) * up                    # down_proj's calib INPUT
          current_hidden_states = F.linear(current_hidden_states, down_proj[expert_idx])
          ...

`Gemma4TextExperts` is wrapped in `@use_experts_implementation`, which
dispatches its *actual* forward body via
`experts_interface.get_interface(config._experts_implementation, original_forward)`
-- i.e. a different (e.g. vectorized) implementation than the loop shown
above may run in practice depending on `config._experts_implementation`.
This does NOT affect the capture strategy below: a `register_forward_pre_hook`
fires on `nn.Module.__call__` before any dispatch happens, and the
(hidden_states_2, top_k_index, top_k_weights) inputs plus the token/expert
assignment implied by top_k_index are identical regardless of which forward
body actually executes (they all have to produce the router's mathematical
result). So hooking the whole `experts` module and replicating the
mask/token_idx bucketing in plain Python (mirroring the loop shown above) is
robust to which underlying experts implementation actually ran.

MoE calibration strategy (implemented by `moe_hook_factory` /
`build_expert_xp` / `run_moe_layer_gptq` below):
  1. A forward-pre-hook on `layer.experts` captures its three positional
     inputs per calibration example and immediately buckets
     `hidden_states_2` rows per expert (replicating the real
     expert_mask/token_idx logic above), accumulating across all examples,
     capped at `--max-rows-per-expert` per expert to bound memory (MoE
     experts are large: `gate_up_proj` alone is ~1GB per layer in bf16 --
     128*1408*2816*2 bytes). `top_k_weights` is captured by the hook but
     deliberately NOT stored anywhere -- GPTQ's Hessian only depends on the
     *inputs* fed to a Linear, not the post-hoc per-token combination weight
     applied after down_proj, so keeping it would just waste memory.
  2. `build_expert_xp` zero-pads each expert's accumulated rows to a common
     row count `n_max`, producing `Xp: [E, n_max, in_features]`. This
     padding is EXACT for GPTQ's Hessian, not an approximation -- see
     gptq.py's `_raw_hessian_batched` docstring: zero rows don't perturb
     X^T X. Experts that never got routed any calibration tokens end up
     with an all-zero Xp; `_raw_hessian_batched`'s dead-column handling
     (diag==0 -> treated as identity) then makes GPTQ degrade gracefully to
     plain rounding for that expert instead of crashing on a singular
     Hessian.
  3. `run_moe_layer_gptq` calls `gptq_nbit_batched(..., scheme="affine")` on
     `gate_up_proj` using the Xp from step 2, then recomputes each expert's
     `intermediate = act_fn(gate) * up` from the SAME gathered rows using the
     REAL (uncorrected -- fetched live from the model parameter, GPTQ never
     mutates the live model in place) `gate_up_proj[expert_idx]` weight --
     exactly mirroring what Gemma4TextExperts.forward itself feeds
     down_proj -- and calls `gptq_nbit_batched` again for `down_proj`.
  4. Both `gate_up_proj` and `down_proj` are assigned `--ffn-bits` (the same
     bit-width as the dense `mlp.*_proj` submodules): the key names used
     (`layers.{i}.experts.gate_up_proj` / `...down_proj`) contain no "attn"
     substring, so `bits_for()` (identical dispatch to the sibling script)
     naturally resolves them to `ffn_bits` -- verified by inspection of
     `bits_for`'s body below, no separate expert bit-category needed.
`scheme="affine"` is used for both experts and dense submodules (matches
MLX's `mx.quantize(mode="affine")`, required for the later MLX splice step --
see gptq.py's `gptq_nbit` docstring for why this specific grid match matters).

Two independent batching granularities for the text component:
  --layers-per-batch     (attention + dense mlp; default 6, same as the E4B
                          sibling -- these are ordinary Linears, cheap to
                          batch several layers' worth at once)
  --moe-layers-per-batch (MoE experts only; default 1 -- MoE experts are
                          ~40x heavier per layer than a dense mlp.*_proj
                          Linear (gate_up_proj alone is ~1GB/layer in bf16,
                          vs a few MB for a dense Linear), and GPTQ's
                          correction pass upcasts to float32 (~4x further),
                          so silently reusing --layers-per-batch's default of
                          6 here would multiply peak GPU memory during the
                          MoE correction pass by roughly 6x*40x relative to
                          the dense path for no accuracy benefit. Each
                          granularity gets its own resumability sidecar
                          (gptq_progress_text.json for the dense pass,
                          gptq_progress_text_moe.json for the MoE pass) and
                          writes to its own batch-file prefix
                          (text_batch_<layers>.safetensors /
                          text_moe_batch_<layers>.safetensors), following the
                          exact same `{"done_layers": [...]}` sidecar shape
                          and `layers.{i}.{submodule}` key convention as the
                          E4B sibling script.

Usage (on the pod, CUDA):
    python gemma4_26b_gptq_calibrate.py \
        --model-dir google/gemma-4-26B-A4B-it \
        --output-dir /workspace/gemma4-26b-corrected \
        --component text --layers all --device cuda \
        --attn-bits 8 --ffn-bits 4 --group-size 64 \
        --layers-per-batch 6 --moe-layers-per-batch 1

    python gemma4_26b_gptq_calibrate.py --component vision ... (same flags,
        --moe-layers-per-batch is simply unused for this component)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gptq import gptq_nbit, gptq_nbit_batched  # noqa: E402

TEXT_SUBMODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
VISION_SUBMODULES = [
    "self_attn.q_proj.linear", "self_attn.k_proj.linear", "self_attn.v_proj.linear", "self_attn.o_proj.linear",
    "mlp.gate_proj.linear", "mlp.up_proj.linear", "mlp.down_proj.linear",
]

# Same calibration prompts as the E4B sibling script -- reused verbatim.
TEXT_PROMPTS = [
    "The history of the Roman Empire spans over a thousand years, beginning with the "
    "traditional founding of Rome in 753 BC and continuing through the fall of the "
    "Western Empire in 476 AD.",
    "Photosynthesis is the process by which green plants and some other organisms use "
    "sunlight to synthesize foods from carbon dioxide and water.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]\n    return left",
    "Q: What is the capital of France?\nA: The capital of France is Paris, a city known "
    "for its art, culture, and history.",
    "Climate change refers to long-term shifts in temperatures and weather patterns, "
    "mainly caused by human activities, especially the burning of fossil fuels.",
    "Once upon a time, in a small village nestled between two mountains, there lived an "
    "old woman who could speak to birds.",
    "The stock market experienced significant volatility today as investors reacted to "
    "new inflation data released by the Federal Reserve.",
    "To install this package, run pip install followed by the package name, then import "
    "it in your Python script before use.",
]

# Same real COCO images as the E4B sibling script -- reused verbatim.
IMAGE_URLS = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # two cats on a couch
    "http://images.cocodataset.org/val2017/000000000139.jpg",
    "http://images.cocodataset.org/val2017/000000000285.jpg",
    "http://images.cocodataset.org/val2017/000000000632.jpg",
    "http://images.cocodataset.org/val2017/000000000724.jpg",
    "http://images.cocodataset.org/val2017/000000000776.jpg",
]


def layer_indices(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    return [int(x) for x in spec.split(",")]


def progress_path(output_dir: Path, name: str) -> Path:
    return output_dir / f"gptq_progress_{name}.json"


def load_progress(output_dir: Path, name: str) -> set[int]:
    p = progress_path(output_dir, name)
    if not p.exists():
        return set()
    return set(json.loads(p.read_text())["done_layers"])


def save_progress(output_dir: Path, name: str, done_layers: set[int]) -> None:
    progress_path(output_dir, name).write_text(json.dumps({"done_layers": sorted(done_layers)}))


def load_calibration_inputs(component: str, args, processor):
    """Returns a list of kwargs dicts, one per calibration example, ready to
    feed straight to the text/vision tower's forward pass. Identical to the
    E4B sibling script's text/vision branches (this model has no audio
    tower, so that branch doesn't exist here)."""
    if component == "text":
        examples = []
        for prompt in TEXT_PROMPTS[: args.examples]:
            enc = processor(text=prompt, return_tensors="pt")
            examples.append({"input_ids": enc["input_ids"]})
        return examples

    if component == "vision":
        import requests
        from PIL import Image

        examples = []
        for url in IMAGE_URLS[: args.examples]:
            img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            enc = processor(images=img, return_tensors="pt")
            examples.append({
                "pixel_values": enc["pixel_values"],
                "pixel_position_ids": enc["image_position_ids"],
            })
        return examples

    raise ValueError(component)


# --- MoE expert calibration -------------------------------------------------


def moe_hook_factory(layer_idx: int, expert_rows: dict, row_counts: dict, max_rows_per_expert: int, num_experts: int):
    """Forward-pre-hook for one layer's `Gemma4TextExperts` module.

    Captures the module's three positional inputs (hidden_states_2,
    top_k_index, top_k_weights) and immediately buckets hidden_states_2 rows
    per expert, replicating Gemma4TextExperts.forward's own
    expert_mask/token_idx gathering logic exactly (see this file's module
    docstring) so the captured rows are bit-identical to what the real
    forward pass feeds gate_up_proj. `top_k_weights` is intentionally never
    stored: GPTQ's Hessian only depends on the *inputs* to a Linear, not the
    post-hoc per-token combination weight applied after down_proj.

    expert_rows: mutated in place, {(layer_idx, expert_idx): [row tensors]}.
    row_counts: mutated in place, {(layer_idx, expert_idx): int}.
    """

    def hook(module, inputs):
        hidden_states_2, top_k_index, _top_k_weights = inputs
        hidden_states_2 = hidden_states_2.detach()
        top_k_index = top_k_index.detach()
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_t in expert_hit:
            e = int(expert_idx_t[0])
            if e == num_experts:
                continue
            key = (layer_idx, e)
            already = row_counts.get(key, 0)
            if already >= max_rows_per_expert:
                continue
            _top_k_pos, token_idx = torch.where(expert_mask[e])
            rows = hidden_states_2[token_idx].to(torch.float32).cpu()
            remaining = max_rows_per_expert - already
            if rows.shape[0] > remaining:
                rows = rows[:remaining]
            if rows.shape[0] == 0:
                continue
            expert_rows.setdefault(key, []).append(rows)
            row_counts[key] = already + rows.shape[0]

    return hook


def build_expert_xp(rows_per_expert: dict[int, torch.Tensor], num_experts: int, in_features: int) -> torch.Tensor:
    """rows_per_expert: {expert_idx: [n_e, in_features] float32 CPU tensor}
    (an expert missing from the dict, or present with n_e==0, means it never
    got routed any calibration tokens). Returns Xp: [E, n_max, in_features],
    zero-padded per expert along the sample axis -- exact for GPTQ's
    Hessian, not an approximation (see gptq.py's `_raw_hessian_batched`
    docstring: zero rows don't perturb X^T X)."""
    n_max = max((t.shape[0] for t in rows_per_expert.values()), default=0)
    n_max = max(n_max, 1)  # keep a valid (non-zero-sized) sample axis even if nothing was ever captured
    Xp = torch.zeros(num_experts, n_max, in_features, dtype=torch.float32)
    for e, rows in rows_per_expert.items():
        if rows.shape[0] == 0:
            continue
        Xp[e, : rows.shape[0]] = rows
    return Xp


def run_moe_layer_gptq(
    experts_module,
    rows_per_expert: dict[int, torch.Tensor],
    bits: int,
    group_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Runs the two batched-GPTQ corrections (gate_up_proj, down_proj) for
    one MoE layer's Gemma4TextExperts module, given already-gathered
    per-expert calibration rows (the gate_up_proj *input* rows, i.e.
    hidden_states_2 rows routed to that expert, as captured by
    `moe_hook_factory`'s hook). Returns
    {"gate_up_proj": W_hat, "down_proj": W_hat}, each [E, out, in] bf16.

    Router (`layer.router.*`) is never referenced anywhere in this function
    or its caller -- it is simply not a calibration target.
    """
    num_experts = experts_module.num_experts
    hidden_dim = experts_module.hidden_dim
    intermediate_dim = experts_module.intermediate_dim

    Xp_gate_up = build_expert_xp(rows_per_expert, num_experts, hidden_dim)
    W_gate_up = experts_module.gate_up_proj.detach().to(torch.float32).cpu()
    result_gate_up = gptq_nbit_batched(
        W=W_gate_up, Xp=Xp_gate_up, bits=bits, group_size=group_size, device=device, scheme="affine",
    )

    # down_proj's calibration input is act_fn(gate) * up, recomputed from the
    # SAME gathered rows using the REAL (uncorrected) gate_up_proj weight --
    # this is calibration data, not a shortcut, and must match exactly what
    # Gemma4TextExperts.forward itself feeds down_proj at inference time.
    down_rows: dict[int, torch.Tensor] = {}
    for e, rows in rows_per_expert.items():
        if rows.shape[0] == 0:
            continue
        gate, up = F.linear(rows, W_gate_up[e]).chunk(2, dim=-1)
        down_rows[e] = (experts_module.act_fn(gate) * up).to(torch.float32)

    Xp_down = build_expert_xp(down_rows, num_experts, intermediate_dim)
    W_down = experts_module.down_proj.detach().to(torch.float32).cpu()
    result_down = gptq_nbit_batched(
        W=W_down, Xp=Xp_down, bits=bits, group_size=group_size, device=device, scheme="affine",
    )

    return {
        "gate_up_proj": result_gate_up["W_hat"].to(torch.bfloat16).contiguous(),
        "down_proj": result_down["W_hat"].to(torch.bfloat16).contiguous(),
    }


# --- Dense (attention + mlp, or vision) calibration -------------------------


def run_dense_batches(
    tower,
    layers_list,
    layers: list[int],
    calib_inputs: list[dict],
    submodules: list[str],
    output_dir: Path,
    progress_name: str,
    batch_prefix: str,
    layers_per_batch: int,
    max_rows_per_module: int,
    bits_for,
    group_size: int,
    device: str,
) -> None:
    """Batched/resumable GPTQ calibration for ordinary nn.Linear submodules
    (attention q/k/v/o_proj, dense mlp gate/up/down_proj, or the vision
    tower's equivalents) -- identical pattern to the E4B sibling script."""
    done_layers = load_progress(output_dir, progress_name)
    if done_layers:
        print(f"Resuming {progress_name}: {len(done_layers)} layer(s) already done: {sorted(done_layers)}", flush=True)
    remaining = [i for i in layers if i not in done_layers]
    if not remaining:
        print(f"Nothing left to do for {progress_name}.", flush=True)
        return
    print(f"Calibrating {progress_name} layers {remaining}, {layers_per_batch} at a time ...", flush=True)

    batches = [remaining[i : i + layers_per_batch] for i in range(0, len(remaining), layers_per_batch)]

    for batch_num, batch in enumerate(batches):
        print(f"\n--- {progress_name} batch {batch_num + 1}/{len(batches)}: layers {batch} ---", flush=True)
        captured: dict[str, list[torch.Tensor]] = {}
        row_counts: dict[str, int] = {}
        handles = []

        def make_hook(key: str):
            def hook(module, inputs):
                if row_counts.get(key, 0) >= max_rows_per_module:
                    return
                x = inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]).cpu()
                captured.setdefault(key, []).append(x)
                row_counts[key] = row_counts.get(key, 0) + x.shape[0]
            return hook

        target_modules: dict[str, torch.nn.Linear] = {}
        for i in batch:
            layer = layers_list[i]
            for sub in submodules:
                try:
                    module = layer.get_submodule(sub)
                except AttributeError:
                    continue  # e.g. self_attn.v_proj is None on some layers (attention_k_eq_v)
                key = f"layers.{i}.{sub}"
                target_modules[key] = module
                handles.append(module.register_forward_pre_hook(make_hook(key)))

        print(f"  Running up to {len(calib_inputs)} calibration example(s) ...", flush=True)
        with torch.no_grad():
            for i, kwargs in enumerate(calib_inputs):
                print(f"    [{i + 1}/{len(calib_inputs)}]", flush=True)
                kwargs = {k: v.to(device) for k, v in kwargs.items()}
                tower(**kwargs)
                if all(row_counts.get(k, 0) >= max_rows_per_module for k in target_modules):
                    print("    (row cap reached for all modules in this batch, skipping remaining examples)", flush=True)
                    break

        for h in handles:
            h.remove()

        print("  Running GPTQ correction for this batch ...", flush=True)
        corrected: dict[str, torch.Tensor] = {}
        key_bits: dict[str, int] = {}
        for key, module in target_modules.items():
            if key not in captured:
                print(f"    {key}: no activations captured (module never ran), skipping", flush=True)
                continue
            bits = bits_for(key)
            X = torch.cat(captured[key], dim=0)
            W = module.weight.detach().to(torch.float32).cpu()
            result = gptq_nbit(W, X, bits=bits, group_size=group_size, device=device, scheme="affine")
            corrected[key] = result["W_hat"].to(torch.bfloat16).contiguous()
            key_bits[key] = bits
            print(f"    {key}: {tuple(W.shape)}, bits={bits}, calib rows={X.shape[0]}", flush=True)

        batch_path = output_dir / f"{batch_prefix}_{'_'.join(str(i) for i in batch)}.safetensors"
        save_file(corrected, str(batch_path), metadata={"group_size": str(group_size), "key_bits": json.dumps(key_bits)})
        print(f"  Wrote {batch_path}", flush=True)
        done_layers.update(batch)
        save_progress(output_dir, progress_name, done_layers)

        del captured, row_counts, corrected
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()


def run_moe_batches(
    tower,
    layers_list,
    layers: list[int],
    calib_inputs: list[dict],
    output_dir: Path,
    moe_layers_per_batch: int,
    max_rows_per_expert: int,
    ffn_bits: int,
    group_size: int,
    device: str,
) -> None:
    """Batched/resumable GPTQ calibration for the MoE experts
    (`layers[i].experts.{gate_up_proj,down_proj}`). Router is never touched.
    """
    progress_name = "text_moe"
    done_layers = load_progress(output_dir, progress_name)
    if done_layers:
        print(f"Resuming {progress_name}: {len(done_layers)} layer(s) already done: {sorted(done_layers)}", flush=True)
    remaining = [i for i in layers if i not in done_layers]
    if not remaining:
        print(f"Nothing left to do for {progress_name}.", flush=True)
        return
    print(f"Calibrating {progress_name} layers {remaining}, {moe_layers_per_batch} at a time ...", flush=True)

    batches = [remaining[i : i + moe_layers_per_batch] for i in range(0, len(remaining), moe_layers_per_batch)]

    for batch_num, batch in enumerate(batches):
        print(f"\n--- {progress_name} batch {batch_num + 1}/{len(batches)}: layers {batch} ---", flush=True)
        expert_rows: dict[tuple[int, int], list[torch.Tensor]] = {}
        row_counts: dict[tuple[int, int], int] = {}
        handles = []
        experts_modules: dict[int, torch.nn.Module] = {}

        for i in batch:
            layer = layers_list[i]
            if not getattr(layer, "enable_moe_block", False):
                # Not expected for this model (enable_moe_block is a global
                # config flag, true for all layers), but guard anyway rather
                # than assume.
                print(f"    layer {i}: enable_moe_block is False, skipping MoE calibration for it", flush=True)
                continue
            experts_module = layer.experts
            experts_modules[i] = experts_module
            handles.append(
                experts_module.register_forward_pre_hook(
                    moe_hook_factory(i, expert_rows, row_counts, max_rows_per_expert, experts_module.num_experts)
                )
            )

        print(f"  Running up to {len(calib_inputs)} calibration example(s) ...", flush=True)
        with torch.no_grad():
            for i, kwargs in enumerate(calib_inputs):
                print(f"    [{i + 1}/{len(calib_inputs)}]", flush=True)
                kwargs = {k: v.to(device) for k, v in kwargs.items()}
                tower(**kwargs)

        for h in handles:
            h.remove()

        print("  Running batched GPTQ correction for this batch ...", flush=True)
        corrected: dict[str, torch.Tensor] = {}
        key_bits: dict[str, int] = {}
        for i, experts_module in experts_modules.items():
            rows_per_expert: dict[int, torch.Tensor] = {}
            total_rows = 0
            for e in range(experts_module.num_experts):
                parts = expert_rows.get((i, e), [])
                rows = torch.cat(parts, dim=0) if parts else torch.zeros(0, experts_module.hidden_dim)
                rows_per_expert[e] = rows
                total_rows += rows.shape[0]
            if total_rows == 0:
                print(f"    layer {i}: no expert activations captured at all, skipping", flush=True)
                continue

            result = run_moe_layer_gptq(experts_module, rows_per_expert, ffn_bits, group_size, device)
            gate_up_key = f"layers.{i}.experts.gate_up_proj"
            down_key = f"layers.{i}.experts.down_proj"
            corrected[gate_up_key] = result["gate_up_proj"]
            corrected[down_key] = result["down_proj"]
            key_bits[gate_up_key] = ffn_bits
            key_bits[down_key] = ffn_bits
            hit_experts = sum(1 for r in rows_per_expert.values() if r.shape[0] > 0)
            print(
                f"    layer {i}: experts hit={hit_experts}/{experts_module.num_experts}, "
                f"total calib rows={total_rows}, bits={ffn_bits}",
                flush=True,
            )

        batch_path = output_dir / f"text_moe_batch_{'_'.join(str(i) for i in batch)}.safetensors"
        save_file(corrected, str(batch_path), metadata={"group_size": str(group_size), "key_bits": json.dumps(key_bits)})
        print(f"  Wrote {batch_path}", flush=True)
        done_layers.update(batch)
        save_progress(output_dir, progress_name, done_layers)

        del expert_rows, row_counts, corrected
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="HF repo id or local path, e.g. google/gemma-4-26B-A4B-it")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--component", required=True, choices=["text", "vision"])
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--layers-per-batch", type=int, default=6,
        help="batch size for attention+dense-mlp submodules (text) or all submodules (vision)",
    )
    parser.add_argument(
        "--moe-layers-per-batch", type=int, default=1,
        help="batch size for MoE expert submodules (text only) -- deliberately much smaller than "
        "--layers-per-batch's default; see module docstring for the memory reasoning",
    )
    parser.add_argument("--max-rows-per-module", type=int, default=8192)
    parser.add_argument(
        "--max-rows-per-expert", type=int, default=1024,
        help="cap on accumulated calibration rows per MoE expert (text only); analogous to "
        "--max-rows-per-module but per-expert, since each of the 128 experts only sees the "
        "subset of tokens routed to it (top_k_experts out of 128 per token) rather than every "
        "token in the batch",
    )
    parser.add_argument("--examples", type=int, default=6, help="calibration prompts/images to use")
    parser.add_argument(
        "--bits", type=int, default=None,
        help="uniform bit-width -- mutually exclusive with --attn-bits/--ffn-bits",
    )
    parser.add_argument("--attn-bits", type=int, default=None, help="bits for attention/self_attn submodules")
    parser.add_argument(
        "--ffn-bits", type=int, default=None,
        help="bits for mlp submodules AND MoE experts (gate_up_proj/down_proj) -- experts are not "
        "a separate bit category, they get the same bits as the dense mlp",
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    args = parser.parse_args()

    if args.bits is not None and (args.attn_bits is not None or args.ffn_bits is not None):
        parser.error("--bits is mutually exclusive with --attn-bits/--ffn-bits")
    if args.bits is None and (args.attn_bits is None or args.ffn_bits is None):
        if args.attn_bits is None and args.ffn_bits is None:
            args.bits = 8
        else:
            parser.error("--attn-bits and --ffn-bits must both be given together")
    attn_bits = args.attn_bits if args.attn_bits is not None else args.bits
    ffn_bits = args.ffn_bits if args.ffn_bits is not None else args.bits

    def bits_for(key: str) -> int:
        return attn_bits if ("attn" in key or "lconv1d" in key) else ffn_bits

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading google/gemma-4-26B-A4B-it ({args.model_dir}) for {args.component} calibration ...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16)
    model = model.to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_dir)

    tower = {"text": model.model.language_model, "vision": model.model.vision_tower}[args.component]
    layers_list = tower.layers if args.component == "text" else tower.encoder.layers
    num_layers = len(layers_list)

    requested = layer_indices(args.layers, num_layers)

    print(f"Loading {args.examples} real calibration example(s) for {args.component} ...", flush=True)
    calib_inputs = load_calibration_inputs(args.component, args, processor)

    if args.component == "vision":
        run_dense_batches(
            tower=tower,
            layers_list=layers_list,
            layers=requested,
            calib_inputs=calib_inputs,
            submodules=VISION_SUBMODULES,
            output_dir=output_dir,
            progress_name="vision",
            batch_prefix="vision_batch",
            layers_per_batch=args.layers_per_batch,
            max_rows_per_module=args.max_rows_per_module,
            bits_for=bits_for,
            group_size=args.group_size,
            device=args.device,
        )
    else:
        run_dense_batches(
            tower=tower,
            layers_list=layers_list,
            layers=requested,
            calib_inputs=calib_inputs,
            submodules=TEXT_SUBMODULES,
            output_dir=output_dir,
            progress_name="text",
            batch_prefix="text_batch",
            layers_per_batch=args.layers_per_batch,
            max_rows_per_module=args.max_rows_per_module,
            bits_for=bits_for,
            group_size=args.group_size,
            device=args.device,
        )
        run_moe_batches(
            tower=tower,
            layers_list=layers_list,
            layers=requested,
            calib_inputs=calib_inputs,
            output_dir=output_dir,
            moe_layers_per_batch=args.moe_layers_per_batch,
            max_rows_per_expert=args.max_rows_per_expert,
            ffn_bits=ffn_bits,
            group_size=args.group_size,
            device=args.device,
        )

    print(f"GEMMA4_26B_GPTQ_CALIBRATE_DONE:{args.component}")


if __name__ == "__main__":
    main()
