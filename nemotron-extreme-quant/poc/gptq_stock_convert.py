"""GPTQ-calibrate a NemotronH model at a single uniform bit-width, with NO
rotation and NO salient-weight overlay, then save a standard HF checkpoint
whose weights already sit exactly on the target affine quantization grid --
so a subsequent stock `mlx_lm.convert -q --q-bits <bits> --q-group-size
<group_size>` (or any other standard N-bit RTN quantizer using the same
group_size/scale convention) reproduces these Hessian-calibrated codes
instead of re-deriving them via naive round-to-nearest.

Why no rotation/salient here: those exist in this project to make *very*
low-bit (1-2 bit) ternary/binary quantization survivable -- at 3+ bits the
naive stock RTN conversion already works (see docs/session_findings_
2026-09-11.md's PPL measurement against F16), so plain GPTQ error
compensation (no incoherence processing, no full-precision outlier pins)
is expected to *improve* on stock RTN's quality at the exact same size and
inference speed (same MLX-native quantized-linear/gather_qmm path, no
custom kernel, no custom model-loading code needed at all).

All bit-width/group-size choices are CLI flags, not hardcoded constants --
this script is meant to be reusable across bit-widths (and, per the disk-
size back-of-envelope this session did for a hypothetical 70B model,
across model sizes) without editing source.

Usage:
    python poc/gptq_stock_convert.py \
        --model /root/nemotron30b-bf16-src --output /root/lightning30b-gptq3bit-src \
        --wikitext /root/llama.cpp/wikitext-2-raw/wiki.train.raw \
        --bits 3 --group-size 64 --calib-chunks 24 --calib-chunk-tokens 512 --moe-subbatch 12

Then, separately (stock mlx_lm, no custom code):
    mlx_lm.convert --hf-path /root/lightning30b-gptq3bit-src \
        --mlx-path /root/lightning30b-gptq3bit-mlx -q --q-bits 3 --q-group-size 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import create_repo, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptq import gptq_nbit, gptq_nbit_batched, hessian_diag
from quantize_full_moe_model import (
    ATTN_PROJECTIONS,
    MAMBA_PROJECTIONS,
    capture_all_activations,
    capture_single_block_activations,
    dense_projection_names,
    load_calibration_chunks,
)

# Mirrors mlx_lm.convert's own mixed_quant_predicate_builder EXACTLY (see
# mlx_lm/convert.py) so that when --quant-recipe is used, GPTQ calibrates
# each projection at the SAME bit-width mlx_lm.convert will later assign it
# via --quant-predicate <recipe> -- otherwise we'd hit the same grid-mismatch
# bug that motivated the affine scheme fix in the first place, just per-layer
# instead of globally. high_bits=6 for all recipes except mixed_3_4 (=4).
QUANT_RECIPES = {
    "mixed_2_6": {"low_bits": 2, "high_bits": 6},
    "mixed_3_4": {"low_bits": 3, "high_bits": 4},
    "mixed_3_6": {"low_bits": 3, "high_bits": 6},
    "mixed_4_6": {"low_bits": 4, "high_bits": 6},
}

# Reverse-engineered from JANG_2L-CRACK's (dealignai) published config.json
# quantization dict (fully public metadata, no download needed -- see docs/
# session_findings_2026-09-11.md section 7q) -- bits assigned purely by
# COMPONENT TYPE, uniform across every layer, not by position or per-layer
# score: generous precision where a component is cheap in total parameter
# count (attention, shared experts), the least precision on the single
# largest pool of parameters (routed MoE experts), with an asymmetric
# up/down split there this project's own recipes never considered (both
# always left up_proj at low_bits regardless of mode). JANG's own README
# describes it only as "3.73-bit affine (MLX)" with no mention of GPTQ/
# Hessian/calibration -- likely naive RTN with this hand-tuned allocation,
# which is why replicating the allocation through THIS project's calibrated
# pipeline is worth testing on its own merits, not just as a copy.
COMPONENT_BIT_RECIPES = {
    "jang": {
        "attention": 8,        # q_proj, k_proj, v_proj, o_proj
        "mamba": 6,            # in_proj, out_proj
        "moe_shared": 8,       # shared_experts up_proj + down_proj (1 expert/block)
        "moe_routed_up": 4,    # routed up_proj / switch_mlp.fc1 (128 experts/block)
        "moe_routed_down": 3,  # routed down_proj / switch_mlp.fc2 (128 experts/block)
        "lm_head": 8,
        "embeddings": 6,
        # Unused by GPTQ itself (mtp.* never reaches this script -- HF
        # transformers drops it on load before GPTQ ever sees it, see
        # inject_mtp_weights.py). Kept here anyway so this dict stays an
        # exact mirror of mlx_convert_recipe.py's copy, which DOES use
        # these (inject_mtp_weights.py imports its predicate builder).
        "mtp_attention": 8,
        "mtp_moe_shared": 8,
        "mtp_moe_routed_up": 6,
        "mtp_moe_routed_down": 6,
        "mtp_fusion": 8,
    },
    # For DENSE NemotronH variants with no MoE at all (e.g.
    # nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16: 42 layers, {mamba: 21,
    # mlp: 17, attention: 4}, no layers_block_type "moe" entries). Unlike
    # the 30B MoE model, nothing here is sparsely-activated -- every
    # component runs on every token, so there's no size-vs-decode-speed
    # tension (see docs/session_findings_2026-09-11.md section 7r): bits
    # can be allocated purely by how large each component is. attention
    # is cheap (only 4/42 blocks) -> generous bits; mlp is the bulk of
    # the model's parameters -> lowest bits, for the minimal-footprint goal.
    "jang-dense": {
        "attention": 8,
        "mamba": 6,
        "mlp": 3,
        "lm_head": 8,
        "embeddings": 6,
    },
    # Found (session_findings §7v) that 3-bit mlp costs real code-generation
    # quality independent of any LoRA issue (attention-only-LoRA bf16: 6/9
    # C++ compile pass; same model quantized jang-dense: 3/9) -- this variant
    # bumps mlp to match mamba's 6 bits, trading disk size for quality on
    # the same minimal-footprint-goal model.
    "jang-dense-mlp6": {
        "attention": 8,
        "mamba": 6,
        "mlp": 6,
        "lm_head": 8,
        "embeddings": 6,
    },
    "jang-dense-mlp8": {
        "attention": 8,
        "mamba": 6,
        "mlp": 8,
        "lm_head": 8,
        "embeddings": 6,
    },
    # Diagnostic: near-uniform 8-bit everywhere (mamba is 21/42 blocks --
    # the single largest component -- and had never been raised above 6-bit
    # in any variant tested so far). If this STILL doesn't recover to
    # unquantized-level C++ competency, the gap isn't about which component
    # gets how many bits at all -- it's something about the GPTQ/affine
    # quantization process itself (group_size, calibration) at any bit-width.
    "dense-8bit": {
        "attention": 8,
        "mamba": 8,
        "mlp": 8,
        "lm_head": 8,
        "embeddings": 8,
    },
}


def recipe_bits_for(
    recipe: str, proj_name: str, layer_idx: int, num_layers: int,
    upgrade_set: set[tuple[int, str]] | None = None,
) -> int:
    """proj_name should be the bare projection name (e.g. "v_proj",
    "down_proj") -- matches mlx_lm's substring check against the full
    parameter path, which for this project's naming is equivalent.

    If upgrade_set is given (sensitivity mode -- see compute_sensitivity_
    scores/select_sensitivity_upgrades below), it REPLACES the positional
    use_more_bits guess for v_proj/down_proj with an explicit per-(layer,
    proj) decision computed from real calibration data; lm_head is always
    high_bits either way, matching mlx_lm's own mixed_quant_predicate_
    builder in both modes."""
    low_bits, high_bits = QUANT_RECIPES[recipe]["low_bits"], QUANT_RECIPES[recipe]["high_bits"]
    if proj_name == "lm_head":
        return high_bits
    if proj_name in ("v_proj", "down_proj"):
        if upgrade_set is not None:
            return high_bits if (layer_idx, proj_name) in upgrade_set else low_bits
        use_more_bits = (
            layer_idx < num_layers // 8
            or layer_idx >= 7 * num_layers // 8
            or (layer_idx - num_layers // 8) % 3 == 2
        )
        if use_more_bits:
            return high_bits
    return low_bits


def compute_sensitivity_scores(
    model, block_types: list[str], dense_acts: dict, moe_acts: dict, percdamp: float = 0.01,
) -> dict[tuple[int, str], float]:
    """Per-(layer_idx, proj_name) score for the two projection kinds the
    mixed-precision recipes ever promote to high_bits (v_proj, down_proj) --
    a data-driven replacement for recipe_bits_for's positional guess.

    Raw error energy = sum_i H_ii * ||W[:, i]||^2, the aggregate Optimal-
    Brain-Damage-style saliency (Frantar et al.'s H_ii * w_ij^2, the same
    quantity gptq.hessian_diag exposes for methods.py's salient-weight
    selection) summed over every weight in the projection matrix -- i.e.
    the expected squared OUTPUT error this whole projection contributes if
    quantized aggressively.

    That raw quantity alone is confounded by depth: activation/weight scale
    tends to grow with depth (residual-stream accumulation), so a naive
    raw-error ranking mostly just recovers "later = bigger", not genuine
    per-layer fragility (confirmed empirically this session -- see docs/
    session_findings_2026-09-11.md section 7p's dry-run results, which came
    back nearly monotonic in layer index). Normalizing by this SAME
    projection's own output energy (sum ||Y||^2, Y = X @ W^T on the same
    calibration data) turns this into a scale-invariant RELATIVE sensitivity
    -- the fraction of this layer's own output signal at risk from
    quantization -- so a layer that's simply larger in absolute terms no
    longer automatically outranks one that's smaller but proportionally
    more fragile."""
    scores: dict[tuple[int, str], float] = {}
    for i, kind in enumerate(block_types):
        block = model.model.layers[i]
        if kind == "attention":
            X = dense_acts.get(i, {}).get("v_proj")
            module = getattr(block.mixer, "v_proj", None)
            if X is None or module is None:
                continue
            W = module.weight.detach().to(torch.float32).cpu()
            diag = hessian_diag(X, W.shape[1], percdamp)
            raw_error = float((diag * (W**2).sum(dim=0)).sum())
            with torch.no_grad():
                Y = F.linear(X.to(torch.float32), W)
            output_energy = float((Y**2).sum())
            scores[(i, "v_proj")] = raw_error / (output_energy + 1e-8)
        elif kind == "moe":
            expert_inputs, shared_inputs = moe_acts.get(i, ({}, {}))
            experts_module = block.mixer.experts
            up_param, down_param, act_fn = experts_module.up_proj, experts_module.down_proj, experts_module.act_fn
            total_raw, total_energy = 0.0, 0.0
            for e, X_up in expert_inputs.items():
                if X_up.shape[0] == 0:
                    continue
                # expert_inputs is captured at the *up_proj* input (hidden_size)
                # -- down_proj's actual input (intermediate_size) doesn't exist
                # until up_proj + activation run, exactly like quantize_moe_block
                # derives it on the fly (no hook can capture it directly).
                W_up = up_param.data[e].detach().to(torch.float32).cpu()
                with torch.no_grad():
                    X_down = act_fn(F.linear(X_up.to(torch.float32), W_up))
                W = down_param.data[e].detach().to(torch.float32).cpu()
                diag = hessian_diag(X_down.cpu(), W.shape[1], percdamp)
                total_raw += float((diag * (W**2).sum(dim=0)).sum())
                with torch.no_grad():
                    Y = F.linear(X_down, W)
                total_energy += float((Y**2).sum())
            X_shared = shared_inputs.get("down_proj")
            if X_shared is not None and X_shared.shape[0] > 0:
                W_shared = block.mixer.shared_experts.down_proj.weight.detach().to(torch.float32).cpu()
                diag = hessian_diag(X_shared, W_shared.shape[1], percdamp)
                total_raw += float((diag * (W_shared**2).sum(dim=0)).sum())
                with torch.no_grad():
                    Y_shared = F.linear(X_shared.to(torch.float32), W_shared)
                total_energy += float((Y_shared**2).sum())
            scores[(i, "down_proj")] = total_raw / (total_energy + 1e-8)
    return scores


def count_positional_upgrades(recipe: str, block_types: list[str]) -> int:
    """How many (layer, proj) pairs the POSITIONAL recipe would promote to
    high_bits -- sensitivity mode picks this many top-scoring pairs instead,
    so the two modes produce equal-sized (equal-bpw) models and are a fair
    apples-to-apples comparison, not confounded by one being simply bigger."""
    num_layers = len(block_types)
    count = 0
    for i, kind in enumerate(block_types):
        proj_name = "v_proj" if kind == "attention" else "down_proj" if kind == "moe" else None
        if proj_name is None:
            continue
        if recipe_bits_for(recipe, proj_name, i, num_layers) == QUANT_RECIPES[recipe]["high_bits"]:
            count += 1
    return count


def select_sensitivity_upgrades(
    scores: dict[tuple[int, str], float], count: int, min_gap: int = 2,
) -> set[tuple[int, str]]:
    """Top-`count` picks by score, with a soft anti-clustering constraint:
    among candidates of the SAME proj_name, skip one that's within
    `min_gap` layers of an already-picked candidate of that family, so a
    model doesn't end up with a long unbroken run of low-bit layers
    squeezed between two picks that scored high for near-identical reasons
    (adjacent layers' activations/Hessians are often correlated). This is
    the data-driven generalization of the positional recipe's "every 3rd
    layer" spacing -- score still decides WHICH layers matter, spacing just
    avoids wasting the upgrade budget on redundant neighbors. Doesn't
    change anything for this specific model (MoE blocks already interleave
    with mamba blocks, which are never candidates), but this script is
    meant to be reused on other architectures where high-scoring layers
    of the same family CAN sit right next to each other.

    Falls back to filling the remaining budget from gap-skipped candidates
    (in score order) if spacing can't be satisfied within `count`, so the
    model's total size still matches the positional recipe exactly."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected: set[tuple[int, str]] = set()
    picked_idx: dict[str, list[int]] = {}
    deferred: list[tuple[int, str]] = []

    for (idx, proj), _score in ranked:
        if len(selected) >= count:
            break
        too_close = any(abs(idx - other) < min_gap for other in picked_idx.get(proj, []))
        if too_close:
            deferred.append((idx, proj))
            continue
        selected.add((idx, proj))
        picked_idx.setdefault(proj, []).append(idx)

    for idx, proj in deferred:
        if len(selected) >= count:
            break
        selected.add((idx, proj))

    return selected


def quantize_dense_block(
    block, kind: str, activations: dict[str, torch.Tensor], bits: int, group_size: int,
    recipe: str | None = None, layer_idx: int = 0, num_layers: int = 1,
    upgrade_set: set[tuple[int, str]] | None = None, component_recipe: str | None = None,
) -> list[str]:
    proj_names = dense_projection_names(kind, block.mixer)
    quantized = []
    for proj_name in proj_names:
        module = getattr(block.mixer, proj_name, None)
        if module is None or proj_name not in activations:
            continue
        if component_recipe:
            component_bits = COMPONENT_BIT_RECIPES[component_recipe]
            if kind not in component_bits:
                raise SystemExit(
                    f"--component-recipe {component_recipe!r} has no bit-width entry for "
                    f"block kind {kind!r} (has: {sorted(component_bits)}) -- add one to "
                    f"COMPONENT_BIT_RECIPES for this model's architecture."
                )
            proj_bits = component_bits[kind]
        elif recipe:
            proj_bits = recipe_bits_for(recipe, proj_name, layer_idx, num_layers, upgrade_set)
        else:
            proj_bits = bits
        W = module.weight.detach().to(torch.float32).cpu()
        X = activations[proj_name]
        result = gptq_nbit(W, X, bits=proj_bits, group_size=group_size, device="cpu", scheme="affine")
        module.weight.data.copy_(result["W_hat"].to(module.weight.dtype))
        quantized.append(f"{proj_name}({proj_bits}b)")
    return quantized


def quantize_moe_block(
    block, expert_inputs: dict, shared_inputs: dict, min_expert_tokens: int, gptq_device: str,
    subbatch: int, bits: int, group_size: int,
    recipe: str | None = None, layer_idx: int = 0, num_layers: int = 1,
    upgrade_set: set[tuple[int, str]] | None = None, component_recipe: str | None = None,
) -> dict:
    experts_module = block.mixer.experts
    num_experts = experts_module.num_experts
    act_fn = experts_module.act_fn

    if component_recipe:
        cbits = COMPONENT_BIT_RECIPES[component_recipe]
        up_bits, down_bits = cbits["moe_routed_up"], cbits["moe_routed_down"]
        shared_up_bits = shared_down_bits = cbits["moe_shared"]
    elif recipe:
        up_bits = recipe_bits_for(recipe, "up_proj", layer_idx, num_layers, upgrade_set)
        down_bits = recipe_bits_for(recipe, "down_proj", layer_idx, num_layers, upgrade_set)
        shared_up_bits = shared_down_bits = None  # set per-projection below
    else:
        up_bits = down_bits = bits
        shared_up_bits = shared_down_bits = None

    valid_experts = [
        i for i in range(num_experts)
        if expert_inputs.get(i) is not None and expert_inputs[i].shape[0] >= min_expert_tokens
    ]
    skipped = num_experts - len(valid_experts)

    up_param = experts_module.up_proj
    down_param = experts_module.down_proj
    t_up_total = t_down_total = 0.0

    for start in range(0, num_experts, subbatch):
        sub = [i for i in range(start, min(start + subbatch, num_experts)) if i in valid_experts]
        if not sub:
            continue

        t0 = time.time()
        W_up_batch = up_param.data[sub].detach().to(torch.float32).cpu()
        X_up_list = [expert_inputs[i] for i in sub]
        n_max = max(x.shape[0] for x in X_up_list)
        Xp_up = torch.zeros(len(sub), n_max, W_up_batch.shape[-1], dtype=torch.float32)
        for j, x in enumerate(X_up_list):
            Xp_up[j, : x.shape[0]] = x
        result_up = gptq_nbit_batched(W_up_batch, Xp_up, bits=up_bits, group_size=group_size, device=gptq_device, scheme="affine")
        up_param.data[sub] = result_up["W_hat"].to(device=up_param.device, dtype=up_param.dtype)
        with torch.no_grad():
            X_down_list = [act_fn(F.linear(X_up_list[j], W_up_batch[j])) for j in range(len(sub))]
        t_up_total += time.time() - t0

        t1 = time.time()
        W_down_batch = down_param.data[sub].detach().to(torch.float32).cpu()
        n_max_d = max(x.shape[0] for x in X_down_list)
        Xp_down = torch.zeros(len(sub), n_max_d, W_down_batch.shape[-1], dtype=torch.float32)
        for j, x in enumerate(X_down_list):
            Xp_down[j, : x.shape[0]] = x
        result_down = gptq_nbit_batched(W_down_batch, Xp_down, bits=down_bits, group_size=group_size, device=gptq_device, scheme="affine")
        down_param.data[sub] = result_down["W_hat"].to(device=down_param.device, dtype=down_param.dtype)
        t_down_total += time.time() - t1

    for proj_name in ("up_proj", "down_proj"):
        module = getattr(block.mixer.shared_experts, proj_name)
        if component_recipe:
            proj_bits = shared_up_bits if proj_name == "up_proj" else shared_down_bits
        elif recipe:
            proj_bits = recipe_bits_for(recipe, proj_name, layer_idx, num_layers, upgrade_set)
        else:
            proj_bits = bits
        W = module.weight.detach().to(torch.float32).cpu()
        X = shared_inputs[proj_name]
        result = gptq_nbit(W, X, bits=proj_bits, group_size=group_size, device="cpu", scheme="affine")
        module.weight.data.copy_(result["W_hat"].to(module.weight.dtype))

    return {
        "quantized_experts": len(valid_experts), "skipped": skipped,
        "t_up": t_up_total, "t_down": t_down_total, "up_bits": up_bits, "down_bits": down_bits,
    }


CHECKPOINT_MANIFEST = "checkpoint_state.json"


# transformers' NemotronH config normalizes layers_block_type to its own
# internal names on load; mlx_lm's nemotron_h.py only recognizes the source
# checkpoint's original names (see its _block_type_to_char dict).
BLOCK_TYPE_HF_TO_MLX = {"linear_attention": "mamba", "full_attention": "attention"}


def _tokenizer_eos_token_ids(output_dir: str) -> list[int]:
    """Resolve the id(s) of the tokenizer's actual eos_token (as named in
    tokenizer_config.json) by looking it up in tokenizer.json's added-tokens
    table. Never hardcode a token id here -- some checkpoints' config.json
    ships an eos_token_id that doesn't match the tokenizer's chat-template
    eos_token at all (e.g. nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16's config.json
    says eos_token_id=2, a bare '</s>'-style stop, while its own
    tokenizer_config.json's eos_token is '<|im_end|>' (id 11) -- the token the
    chat template, and any chat/tool-call fine-tune, actually terminates
    turns with). Returns [] if the files are missing or the token isn't found.
    """
    tok_cfg_path = f"{output_dir}/tokenizer_config.json"
    tok_path = f"{output_dir}/tokenizer.json"
    if not (Path(tok_cfg_path).exists() and Path(tok_path).exists()):
        return []
    with open(tok_cfg_path) as f:
        eos_str = json.load(f).get("eos_token")
    if not eos_str:
        return []
    with open(tok_path) as f:
        added_tokens = json.load(f).get("added_tokens", [])
    return [t["id"] for t in added_tokens if t.get("content") == eos_str]


def fixup_config_for_mlx(output_dir: str) -> None:
    """transformers' save_pretrained() round-trip for this NemotronH config
    class breaks mlx_lm.convert in two ways, both patched here:
    1. Drops num_hidden_layers (derives layer count from layers_block_type
       internally and never re-serializes the field) -- mlx_lm's ModelArgs.
       from_dict requires it explicitly, failing with 'missing 1 required
       positional argument' otherwise.
    2. Renames layers_block_type entries to transformers' internal vocabulary
       ("linear_attention"/"full_attention" instead of "mamba"/"attention")
       -- mlx_lm.models.nemotron_h only recognizes the original names,
       failing with a KeyError in _block_type_to_char otherwise.
    3. Serializes time_step_limit's float('inf') as a non-standard
       {'__float__': 'Infinity'} dict -- mlx_lm.models.ssm's mx.clip() can't
       build an mx.array from that dict. The source checkpoint just omits
       the field entirely (mlx_lm derives the default (0.0, inf) itself in
       __post_init__ when unset) -- and it must be OMITTED, not set to
       null/None: transformers' strict config dataclass validates the field
       as list[float] | tuple[float, ...] with no None variant, so an
       explicit null in the JSON fails AutoConfig.from_pretrained (hit via
       mlx_lm's own tokenizer loading, which pulls in the HF config class).
    4. eos_token_id missing the tokenizer's actual chat-template eos token
       (see _tokenizer_eos_token_ids docstring) -- both mlx_lm and plain
       transformers generation stop on config/generation_config's
       eos_token_id, so a model fine-tuned to end turns with a token that's
       absent from that set never stops and degenerates into repeating it.
       Merged into whatever ids were already declared (as a sorted list),
       never overwritten, so a correct upstream int/list is left untouched
       and only ever widened.
    """
    eos_ids = set(_tokenizer_eos_token_ids(output_dir))

    def _merge_eos(cfg: dict) -> bool:
        if not eos_ids:
            return False
        existing = cfg.get("eos_token_id")
        existing_list = existing if isinstance(existing, list) else ([existing] if existing is not None else [])
        merged = sorted(set(existing_list) | eos_ids)
        if merged != existing_list:
            cfg["eos_token_id"] = merged
            return True
        return False

    path = f"{output_dir}/config.json"
    with open(path) as f:
        cfg = json.load(f)
    changed = False
    if "num_hidden_layers" not in cfg:
        cfg["num_hidden_layers"] = len(cfg["layers_block_type"])
        changed = True
    remapped = [BLOCK_TYPE_HF_TO_MLX.get(t, t) for t in cfg["layers_block_type"]]
    if remapped != cfg["layers_block_type"]:
        cfg["layers_block_type"] = remapped
        changed = True
    if isinstance(cfg.get("time_step_limit"), list):
        del cfg["time_step_limit"]
        changed = True
    changed = _merge_eos(cfg) or changed
    if changed:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)

    gen_path = f"{output_dir}/generation_config.json"
    if Path(gen_path).exists():
        with open(gen_path) as f:
            gen_cfg = json.load(f)
        if _merge_eos(gen_cfg):
            with open(gen_path, "w") as f:
                json.dump(gen_cfg, f, indent=2)


def save_checkpoint(model, tokenizer, output_dir: str, last_completed_block: int, args) -> None:
    """Save a resumable mid-run checkpoint: full model + tokenizer + a manifest
    recording how far quantization has progressed, so a later run can pick up
    at last_completed_block + 1 instead of redoing the whole thing. Pushes to
    a private HF repo if --checkpoint-hf-repo is given (cheap insurance against
    losing hours of GPU-side quantization work to a pod dying mid-run)."""
    print(f"--- checkpointing after block {last_completed_block} to {output_dir} ---", flush=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    fixup_config_for_mlx(output_dir)
    with open(f"{output_dir}/{CHECKPOINT_MANIFEST}", "w") as f:
        json.dump(
            {
                "last_completed_block": last_completed_block,
                "bits": args.bits,
                "quant_recipe": args.quant_recipe,
                "quant_recipe_mode": args.quant_recipe_mode,
                "group_size": args.group_size,
                "source_model": args.model,
            },
            f,
        )
    if args.checkpoint_hf_repo:
        # The local checkpoint above is the actual safety net (it's what
        # --resume-from-checkpoint reads); a failure pushing it to HF (auth,
        # network, rate limit) shouldn't crash a multi-hour quantization run
        # that has otherwise succeeded -- log and keep going.
        try:
            import subprocess

            create_repo(args.checkpoint_hf_repo, repo_type="model", private=True, exist_ok=True)
            subprocess.run(
                ["hf", "upload", args.checkpoint_hf_repo, output_dir, ".", "--repo-type", "model"],
                check=True,
            )
            print(f"--- checkpoint pushed to https://huggingface.co/{args.checkpoint_hf_repo} ---", flush=True)
        except Exception as e:
            print(f"--- WARNING: checkpoint push to HF failed ({e!r}), continuing with local checkpoint only ---", flush=True)


def resolve_resume(args) -> tuple[str, int]:
    """If --resume-from-checkpoint is set, returns (local path to load the model
    from, start_block to resume at) based on the checkpoint's manifest --
    otherwise returns (args.model, args.start_block) unchanged."""
    if not args.resume_from_checkpoint:
        return args.model, args.start_block

    src = args.resume_from_checkpoint
    if "/" in src and not Path(src).exists():
        local_dir = f"{args.output}-resume-src"
        print(f"Downloading checkpoint {src} -> {local_dir} ...", flush=True)
        snapshot_download(repo_id=src, local_dir=local_dir)
        src = local_dir

    manifest_path = f"{src}/{CHECKPOINT_MANIFEST}"
    if not Path(manifest_path).exists():
        raise SystemExit(f"--resume-from-checkpoint given but {manifest_path} not found -- not a checkpoint saved by this script")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if (
        manifest["bits"] != args.bits
        or manifest.get("quant_recipe") != args.quant_recipe
        or manifest.get("quant_recipe_mode", "positional") != args.quant_recipe_mode
        or manifest["group_size"] != args.group_size
    ):
        raise SystemExit(
            f"Checkpoint was made with bits={manifest['bits']} quant_recipe={manifest.get('quant_recipe')} "
            f"quant_recipe_mode={manifest.get('quant_recipe_mode', 'positional')} group_size={manifest['group_size']}, "
            f"but this run asked for bits={args.bits} quant_recipe={args.quant_recipe} "
            f"quant_recipe_mode={args.quant_recipe_mode} group_size={args.group_size} -- refusing to mix"
        )
    resume_block = manifest["last_completed_block"] + 1
    print(f"Resuming from {src}: blocks 0-{manifest['last_completed_block']} already done, continuing at block {resume_block}", flush=True)
    return src, resume_block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wikitext", required=True)
    parser.add_argument(
        "--extra-calib-file", action="append", default=[],
        help="additional plain-text file(s) to mix into the calibration corpus alongside --wikitext "
        "(repeatable). Any text file works -- to use a chat/tool-call JSONL dataset, flatten it first "
        "with poc/flatten_chat_jsonl_to_text.py. Chunks are drawn from each file the same way "
        "--wikitext is (see --extra-calib-chunks), then concatenated into one calibration set.",
    )
    parser.add_argument(
        "--extra-calib-chunks", type=int, default=None,
        help="chunks to draw from EACH --extra-calib-file. Defaults to --calib-chunks (same weight as wikitext).",
    )
    parser.add_argument(
        "--bits", type=int, default=None,
        help="uniform bit-width for every linear layer. Required unless --quant-recipe is given "
        "(the recipe assigns bits per-layer instead).",
    )
    parser.add_argument(
        "--quant-recipe", default=None, choices=list(QUANT_RECIPES),
        help="mixed-precision recipe matching mlx_lm.convert's --quant-predicate of the same name "
        "EXACTLY (see recipe_bits_for's docstring) -- overrides --bits per-layer. The downstream "
        "mlx_lm.convert run must use --quant-predicate <same recipe name> or the grids won't match.",
    )
    parser.add_argument(
        "--quant-recipe-mode", default="positional", choices=["positional", "sensitivity", "component"],
        help="positional (default): pick which layers get high_bits by the same llama.cpp-style "
        "position guess mlx_lm.convert's --quant-predicate uses. sensitivity: pick the SAME NUMBER "
        "of layers (equal-sized model) but choose WHICH ones by real per-layer GPTQ-Hessian saliency "
        "instead of guessing by position -- requires one-shot calibration (no --sequential). component: "
        "ignore --quant-recipe/--bits/layer position entirely and assign bits purely by COMPONENT TYPE "
        "via --component-recipe (see COMPONENT_BIT_RECIPES) -- reverse-engineered from JANG_2L-CRACK's "
        "published bit allocation, see docs/session_findings_2026-09-11.md section 7q. Both sensitivity "
        "and component modes need the downstream conversion to use poc/mlx_convert_recipe.py (not stock "
        "mlx_lm.convert --quant-predicate, which only knows the positional formula).",
    )
    parser.add_argument(
        "--component-recipe", default=None, choices=list(COMPONENT_BIT_RECIPES),
        help="required for --quant-recipe-mode component: which named component->bits map to use.",
    )
    parser.add_argument("--group-size", type=int, required=True, help="must match the group_size the downstream mlx_lm.convert -q run uses")
    parser.add_argument("--calib-chunks", type=int, default=24)
    parser.add_argument("--calib-chunk-tokens", type=int, default=512)
    parser.add_argument("--min-expert-tokens", type=int, default=8)
    parser.add_argument("--moe-subbatch", type=int, default=12)
    parser.add_argument("--gptq-device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--end-block", type=int, default=None, help="exclusive; default = num_hidden_layers")
    parser.add_argument(
        "--sequential", action="store_true",
        help="capture each block's calibration activations against the model with all prior blocks "
        "already quantized in place, instead of one-shot capture against the pristine model.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="save a resumable checkpoint (full model + manifest) every N blocks. 0 (default) disables checkpointing.",
    )
    parser.add_argument(
        "--checkpoint-hf-repo", default=None,
        help="private HF repo id to push each checkpoint to (e.g. user/nemotron-gptq3bit-checkpoint). "
        "Requires `hf auth login`. If omitted, checkpoints are only saved locally to --output.",
    )
    parser.add_argument(
        "--resume-from-checkpoint", default=None,
        help="local path or HF repo id of a checkpoint saved by a previous --checkpoint-every run; "
        "resumes at last_completed_block + 1 instead of --start-block.",
    )
    parser.add_argument(
        "--sensitivity-min-gap", type=int, default=2,
        help="with --quant-recipe-mode sensitivity: minimum layer-index gap enforced between two "
        "upgraded layers of the SAME projection family, so high-scoring neighbors don't both get "
        "picked at the expense of a differently-located layer -- generalizes the positional recipe's "
        "'every 3rd layer' spacing into a score-driven constraint. Set to 0 to disable (pure top-K).",
    )
    parser.add_argument(
        "--sensitivity-dry-run", action="store_true",
        help="with --quant-recipe-mode sensitivity: run calibration + scoring, print/save the "
        "sensitivity_manifest.json, then exit WITHOUT quantizing or saving a checkpoint -- for "
        "inspecting what the heuristic would pick on the real model before committing to a full run.",
    )
    parser.add_argument(
        "--sensitivity-manifest-in", default=None,
        help="with --quant-recipe-mode sensitivity: path to a sensitivity_manifest.json from a "
        "PREVIOUS --sensitivity-dry-run (or full run) on this exact model/recipe -- skips "
        "re-scoring (the slowest part of sensitivity mode) and reuses that manifest's upgrade "
        "choices verbatim. Calibration is still captured (GPTQ itself needs it regardless).",
    )
    parser.add_argument(
        "--cpu-threads", type=int, default=32,
        help="torch.set_num_threads for the CPU-bound sequential column loop inside GPTQ, "
        "regardless of --gptq-device (that flag only controls the matmul device).",
    )
    args = parser.parse_args()
    if args.quant_recipe_mode == "component":
        if args.component_recipe is None:
            raise SystemExit("--quant-recipe-mode component requires --component-recipe.")
    elif args.quant_recipe is None and args.bits is None:
        raise SystemExit("Either --bits or --quant-recipe is required.")
    if args.quant_recipe_mode == "sensitivity":
        if args.quant_recipe is None:
            raise SystemExit("--quant-recipe-mode sensitivity requires --quant-recipe.")
        if args.sequential:
            raise SystemExit(
                "--quant-recipe-mode sensitivity requires one-shot calibration (remove --sequential) "
                "-- it needs every layer's pristine-model activations up front to score them."
            )

    model_path, resume_start_block = resolve_resume(args)
    args.start_block = max(args.start_block, resume_start_block)

    torch.set_num_threads(args.cpu_threads)
    device = "cuda"
    print(f"Loading {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=False, device_map=device
    )
    model.eval()

    # Read layers_block_type off the LOADED model's config, not the raw
    # config.json -- some checkpoints (e.g. Nemotron-3-Nano-4B) don't
    # serialize layers_block_type at all, only a compact
    # hybrid_override_pattern string; transformers' NemotronHConfig
    # derives the per-layer list from that pattern at load time, but only
    # as a live attribute, never written back to the JSON file. A model
    # that's round-tripped through save_pretrained (e.g. an axolotl LoRA
    # merge output) hits a related issue instead: layers_block_type IS in
    # the file, but renamed to transformers' internal vocabulary
    # ("linear_attention"/"full_attention" instead of "mamba"/"attention",
    # see fixup_config_for_mlx's docstring above for the same issue on
    # this script's OWN output). Reading via model.config after load and
    # normalizing through BLOCK_TYPE_HF_TO_MLX handles both cases at once.
    block_types = [BLOCK_TYPE_HF_TO_MLX.get(t, t) for t in model.config.layers_block_type]
    end_block = args.end_block if args.end_block is not None else len(block_types)
    active_types = ["_"] * args.start_block + block_types[args.start_block : end_block]
    active_types += ["_"] * (len(block_types) - end_block)

    calib_ids = load_calibration_chunks(args.wikitext, tokenizer, args.calib_chunks, args.calib_chunk_tokens)
    extra_calib_chunks = args.extra_calib_chunks if args.extra_calib_chunks is not None else args.calib_chunks
    for extra_file in args.extra_calib_file:
        extra_ids = load_calibration_chunks(extra_file, tokenizer, extra_calib_chunks, args.calib_chunk_tokens)
        print(f"Mixing in {len(extra_ids)} chunks from {extra_file}", flush=True)
        calib_ids.extend(extra_ids)
    total_tokens = sum(ids.shape[1] for ids in calib_ids)
    if args.quant_recipe_mode == "component":
        bits_desc = f"component_recipe={args.component_recipe}"
    elif args.quant_recipe:
        bits_desc = f"quant_recipe={args.quant_recipe}"
    else:
        bits_desc = f"bits={args.bits}"
    print(f"Calibration: {len(calib_ids)} chunks, {total_tokens} tokens total, {bits_desc} group_size={args.group_size}", flush=True)

    dense_acts, moe_acts = {}, {}
    if not args.sequential:
        print("Running one-shot calibration forward pass ...", flush=True)
        t_cap = time.time()
        dense_acts, moe_acts = capture_all_activations(model, active_types, calib_ids, device)
        print(f"Calibration capture done in {time.time() - t_cap:.0f}s", flush=True)
    else:
        print("Sequential mode: capturing + quantizing block by block ...", flush=True)

    upgrade_set = None
    if args.quant_recipe_mode == "sensitivity" and args.sensitivity_manifest_in:
        print(f"Loading precomputed sensitivity manifest from {args.sensitivity_manifest_in} (skipping re-scoring) ...", flush=True)
        with open(args.sensitivity_manifest_in) as f:
            manifest_in = json.load(f)
        if manifest_in["recipe"] != args.quant_recipe or manifest_in["group_size"] != args.group_size:
            raise SystemExit(
                f"{args.sensitivity_manifest_in} was computed for recipe={manifest_in['recipe']} "
                f"group_size={manifest_in['group_size']}, but this run asked for recipe={args.quant_recipe} "
                f"group_size={args.group_size} -- refusing to reuse a mismatched manifest."
            )
        upgrade_set = {(i, proj_name) for i, proj_name in manifest_in["upgrades"]}
        print(f"Loaded {len(upgrade_set)} upgraded (layer, proj) pairs from manifest.", flush=True)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        with open(f"{args.output}/sensitivity_manifest.json", "w") as f:
            json.dump(manifest_in, f, indent=2)
    elif args.quant_recipe_mode == "sensitivity":
        print("Scoring per-layer sensitivity (GPTQ-Hessian saliency) ...", flush=True)
        t_score = time.time()
        scores = compute_sensitivity_scores(model, block_types, dense_acts, moe_acts)
        upgrade_count = count_positional_upgrades(args.quant_recipe, block_types)
        upgrade_set = select_sensitivity_upgrades(scores, upgrade_count, min_gap=args.sensitivity_min_gap)
        print(f"Sensitivity scoring done in {time.time() - t_score:.0f}s, upgrading {len(upgrade_set)}/{len(scores)} candidates to {QUANT_RECIPES[args.quant_recipe]['high_bits']}-bit:", flush=True)
        for (i, proj_name), score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            marker = "UPGRADE" if (i, proj_name) in upgrade_set else "       "
            print(f"  [{marker}] block {i:2d} {proj_name:10s} score={score:.4e}", flush=True)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        with open(f"{args.output}/sensitivity_manifest.json", "w") as f:
            json.dump(
                {
                    "recipe": args.quant_recipe,
                    "low_bits": QUANT_RECIPES[args.quant_recipe]["low_bits"],
                    "high_bits": QUANT_RECIPES[args.quant_recipe]["high_bits"],
                    "group_size": args.group_size,
                    "sensitivity_min_gap": args.sensitivity_min_gap,
                    "upgrades": sorted([i, proj_name] for i, proj_name in upgrade_set),
                },
                f, indent=2,
            )
        print(f"Wrote {args.output}/sensitivity_manifest.json", flush=True)
        if args.sensitivity_dry_run:
            print("--sensitivity-dry-run: stopping here, no quantization performed.", flush=True)
            return

    run_start = time.time()
    for i in range(args.start_block, end_block):
        kind = block_types[i]
        block = model.model.layers[i]
        t_block = time.time()

        if args.sequential:
            captured = capture_single_block_activations(model, block, kind, calib_ids, device)
            if kind == "moe":
                moe_acts[i] = captured
            else:
                dense_acts[i] = captured

        if kind == "moe":
            expert_inputs, shared_inputs = moe_acts[i]
            stats = quantize_moe_block(
                block, expert_inputs, shared_inputs, args.min_expert_tokens, args.gptq_device,
                args.moe_subbatch, args.bits, args.group_size,
                recipe=args.quant_recipe, layer_idx=i, num_layers=len(block_types), upgrade_set=upgrade_set,
                component_recipe=args.component_recipe if args.quant_recipe_mode == "component" else None,
            )
            print(
                f"[block {i}/{end_block - 1}] moe: quantized={stats['quantized_experts']} "
                f"skipped={stats['skipped']} up_bits={stats['up_bits']} down_bits={stats['down_bits']} "
                f"(up {stats['t_up']:.1f}s, down {stats['t_down']:.1f}s), "
                f"total_elapsed={time.time() - run_start:.0f}s", flush=True,
            )
        else:
            activations = dense_acts.get(i, {})
            quantized = quantize_dense_block(
                block, kind, activations, args.bits, args.group_size,
                recipe=args.quant_recipe, layer_idx=i, num_layers=len(block_types), upgrade_set=upgrade_set,
                component_recipe=args.component_recipe if args.quant_recipe_mode == "component" else None,
            )
            print(
                f"[block {i}/{end_block - 1}] {kind}: {quantized}, "
                f"block_time={time.time() - t_block:.1f}s, total_elapsed={time.time() - run_start:.0f}s", flush=True,
            )

        if args.checkpoint_every and (i - args.start_block + 1) % args.checkpoint_every == 0:
            save_checkpoint(model, tokenizer, args.output, i, args)

    model = model.to("cpu")
    model.generation_config.do_sample = True
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    fixup_config_for_mlx(args.output)
    print(f"\nSaved to {args.output}. Total quantization time: {time.time() - run_start:.0f}s", flush=True)
    if args.quant_recipe_mode == "component":
        print(
            f"Next step (CUSTOM converter -- stock mlx_lm.convert has no concept of component-type "
            f"bit allocation): python poc/mlx_convert_recipe.py --hf-path {args.output} "
            f"--mlx-path {args.output}-mlx --group-size {args.group_size} --mode component "
            f"--component-recipe {args.component_recipe}",
            flush=True,
        )
    elif args.quant_recipe_mode == "sensitivity":
        print(
            f"Next step (CUSTOM converter -- stock mlx_lm.convert --quant-predicate would re-derive "
            f"positional bits and corrupt this run's sensitivity-based choices): "
            f"python poc/mlx_convert_recipe.py --hf-path {args.output} "
            f"--mlx-path {args.output}-mlx --group-size {args.group_size} --mode sensitivity",
            flush=True,
        )
    elif args.quant_recipe:
        print(
            f"Next step (stock mlx_lm, no custom code): mlx_lm.convert --hf-path {args.output} "
            f"--mlx-path {args.output}-mlx -q --quant-predicate {args.quant_recipe} --q-group-size {args.group_size}",
            flush=True,
        )
    else:
        print(
            f"Next step (stock mlx_lm, no custom code): mlx_lm.convert --hf-path {args.output} "
            f"--mlx-path {args.output}-mlx -q --q-bits {args.bits} --q-group-size {args.group_size}",
            flush=True,
        )


if __name__ == "__main__":
    main()
