"""GPTQ-calibrate a NemotronH checkpoint's MTP (Multi-Token-Prediction)
head, using transformers' own NemotronHBlock class exactly the way
gptq_stock_convert.py already uses it for backbone attention/MoE blocks:
mtp.layers.0/1 in the real checkpoint ARE NemotronHBlock instances
(config.mtp_layers_block_type picks the mixer kind), just with three extra
siblings transformers' own model class never attaches -- eh_proj/enorm/
hnorm on the first, final_layernorm on the last -- since
NemotronHForCausalLM has no Python attribute to receive mtp.* weights at
all and drops them on load entirely (see extract_mtp_weights.py's
docstring, and docs/FINDINGS.md section 4.1).

Because these ARE real NemotronHBlock instances, gptq_stock_convert.py's
quantize_dense_block/quantize_moe_block work against them completely
UNCHANGED -- they only ever touch block.mixer, which has the exact same
submodule shape as a backbone block of the same kind.

Calibrates against real (hidden_state @ position p, real token @ p+1)
pairs from the LoRA-merged, pre-GPTQ bf16 model -- the same eh_proj fusion
mtp_step (mlx_lm's nemotron_h.py) does at inference, just with the REAL
next token standing in for whatever the backbone/draft would produce.
One-shot only (not sequential): the MTP head is a single small appendage,
not a multi-block stack an already-quantized upstream block could be
compensated against.

Output is a new bf16 safetensors file with the SAME key names
extract_mtp_weights.py produces, now GPTQ-corrected and already sitting on
the exact affine grid the given --group-size/--component-recipe (or
--bits) implies -- feed it straight into inject_mtp_weights.py as
--mtp-weights unchanged; its own quantize_model call re-derives the same
codes from already-on-grid values instead of re-quantizing them (same
trick gptq_nbit's docstring describes for the backbone).

Usage:
    python poc/gptq_mtp.py \
        --model /root/nemotron30b-bf16-RUN-merged \
        --mtp-weights /root/mtp-head-RUN/mtp_weights.safetensors \
        --mtp-config /root/mtp-head-RUN/mtp_config.json \
        --wikitext /root/llama.cpp/wikitext-2-raw/wiki.train.raw \
        --output /root/mtp-head-RUN/mtp_weights_gptq.safetensors \
        --component-recipe jang --group-size 64
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHBlock, NemotronHRMSNorm

from gptq import gptq_nbit
from gptq_stock_convert import COMPONENT_BIT_RECIPES, quantize_dense_block, quantize_moe_block
from quantize_full_moe_model import dense_projection_names, load_calibration_chunks

# transformers normalizes layers_block_type to its own vocabulary on load
# (see gptq_stock_convert.py's BLOCK_TYPE_HF_TO_MLX for the same issue in
# the other direction) -- the extracted mtp_config.json carries the
# checkpoint's original mlx-side names, so building transformers modules
# FROM them needs the inverse map.
BLOCK_TYPE_MLX_TO_HF = {"mamba": "linear_attention", "attention": "full_attention", "moe": "moe"}


class MTPBlock(NemotronHBlock):
    def __init__(self, config, layer_idx: int, is_first: bool, is_last: bool):
        super().__init__(config, layer_idx)
        eps = config.layer_norm_epsilon
        if is_first:
            self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
            self.enorm = NemotronHRMSNorm(config.hidden_size, eps=eps)
            self.hnorm = NemotronHRMSNorm(config.hidden_size, eps=eps)
        if is_last:
            self.final_layernorm = NemotronHRMSNorm(config.hidden_size, eps=eps)


def build_mtp_layers(base_config, pattern_mlx: list[str]) -> nn.ModuleList:
    mtp_config = copy.deepcopy(base_config)
    mtp_config.layers_block_type = [BLOCK_TYPE_MLX_TO_HF.get(t, t) for t in pattern_mlx]
    mtp_config.num_hidden_layers = len(pattern_mlx)
    n = len(pattern_mlx)
    return nn.ModuleList(
        [MTPBlock(mtp_config, i, is_first=(i == 0), is_last=(i == n - 1)) for i in range(n)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="LoRA-merged (or base) bf16 HF checkpoint dir")
    parser.add_argument("--mtp-weights", required=True, help="extract_mtp_weights.py's output safetensors")
    parser.add_argument("--mtp-config", required=True, help="extract_mtp_weights.py's output mtp_config.json")
    parser.add_argument("--wikitext", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--bits", type=int, default=None, help="uniform bit-width; overridden by --component-recipe")
    parser.add_argument("--component-recipe", default=None, choices=list(COMPONENT_BIT_RECIPES))
    parser.add_argument("--calib-chunks", type=int, default=24)
    parser.add_argument("--calib-chunk-tokens", type=int, default=512)
    parser.add_argument("--min-expert-tokens", type=int, default=8)
    parser.add_argument("--moe-subbatch", type=int, default=12)
    parser.add_argument("--gptq-device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"], help="device for the base model's own forward pass")
    args = parser.parse_args()
    if args.component_recipe is None and args.bits is None:
        raise SystemExit("Either --bits or --component-recipe is required.")

    if args.component_recipe:
        cbits = COMPONENT_BIT_RECIPES[args.component_recipe]
        for key in ("mtp_attention", "mtp_moe_shared", "mtp_moe_routed_up", "mtp_moe_routed_down"):
            if key not in cbits:
                raise SystemExit(f"--component-recipe {args.component_recipe!r} has no {key!r} entry.")
        eh_bits = cbits.get("mtp_fusion", cbits["mtp_attention"])
        # quantize_dense_block/quantize_moe_block look up bits by plain
        # component-name keys ("attention"/"moe_shared"/...) into
        # COMPONENT_BIT_RECIPES[component_recipe] -- remap this recipe's
        # dedicated mtp_* tier onto those same plain keys under a throwaway
        # recipe name so both functions apply completely unmodified.
        COMPONENT_BIT_RECIPES["_mtp_view"] = {
            "attention": cbits["mtp_attention"],
            "moe_shared": cbits["mtp_moe_shared"],
            "moe_routed_up": cbits["mtp_moe_routed_up"],
            "moe_routed_down": cbits["mtp_moe_routed_down"],
        }
        block_component_recipe = "_mtp_view"
    else:
        eh_bits = args.bits
        block_component_recipe = None

    print(f"Loading base model {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=False, device_map=args.device
    )
    model.eval()

    with open(args.mtp_config) as f:
        mtp_cfg = json.load(f)
    pattern = mtp_cfg["mtp_layers_block_type"]
    if len(pattern) != 2:
        raise SystemExit(
            f"gptq_mtp.py only supports the k=1 two-block MTP head (attention+moe), got "
            f"mtp_layers_block_type={pattern} -- k>1 (chained draft depth) needs a different "
            f"calibration loop (feed layer1's own output back through eh_proj for the next block-pair)."
        )

    layers = build_mtp_layers(model.config, pattern).to(device=args.device, dtype=torch.bfloat16)
    layer0, layer1 = layers[0], layers[1]

    print(f"Loading extracted mtp.* weights from {args.mtp_weights} ...", flush=True)
    raw = load_file(args.mtp_weights)
    state_dict = {k[len("mtp.layers."):]: v for k, v in raw.items() if k.startswith("mtp.layers.")}
    # NemotronHExperts stores every expert's weight as one 3D nn.Parameter
    # (num_experts, out, in) -- the raw checkpoint has them as separate
    # per-expert tensors (the on-disk safetensors convention), same as
    # mlx_lm.models.nemotron_h.Model.sanitize()'s own expert-stacking loop
    # for the exact same reason, just stacking into HF's parameter shape
    # instead of mlx's switch_mlp.fc1/fc2.
    prefixes = {k.rsplit(".experts.", 1)[0] for k in state_dict if ".experts." in k}
    for prefix in prefixes:
        for name in ("up_proj", "down_proj"):
            first_key = f"{prefix}.experts.0.{name}.weight"
            if first_key not in state_dict:
                continue
            num_experts = layers[1].mixer.experts.num_experts
            stacked = torch.stack(
                [state_dict.pop(f"{prefix}.experts.{e}.{name}.weight") for e in range(num_experts)]
            )
            state_dict[f"{prefix}.experts.{name}"] = stacked
    missing, unexpected = layers.load_state_dict(state_dict, strict=True)
    layers = layers.to(device=args.device)

    print("Loading calibration corpus ...", flush=True)
    calib_ids = load_calibration_chunks(args.wikitext, tokenizer, args.calib_chunks, args.calib_chunk_tokens)

    dense_targets = {
        name: getattr(layer0.mixer, name)
        for name in dense_projection_names(pattern[0], layer0.mixer)
        if isinstance(getattr(layer0.mixer, name, None), nn.Linear)
    }
    dense_captured: dict[str, list[torch.Tensor]] = {name: [] for name in dense_targets}
    eh_proj_captured: list[torch.Tensor] = []

    def make_dense_hook(name):
        def hook(module, inputs):
            dense_captured[name].append(
                inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]).cpu()
            )

        return hook

    def eh_proj_hook(module, inputs):
        eh_proj_captured.append(
            inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]).cpu()
        )

    experts_module = layer1.mixer.experts
    num_experts = experts_module.num_experts
    moe_expert_inputs: dict[int, list[torch.Tensor]] = {e: [] for e in range(num_experts)}
    moe_shared_inputs: dict[str, list[torch.Tensor]] = {"up_proj": [], "down_proj": []}

    def experts_hook(module, args_, kwargs_):
        hidden_states = args_[0] if args_ else kwargs_["hidden_states"]
        top_k_index = args_[1] if len(args_) > 1 else kwargs_["top_k_index"]
        hidden_states = hidden_states.detach().to(torch.float32)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
        for expert_idx in expert_hit:
            expert_idx = expert_idx.item()
            _, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            moe_expert_inputs[expert_idx].append(hidden_states[token_idx].cpu())

    def make_shared_hook(name):
        def hook(module, inputs):
            moe_shared_inputs[name].append(
                inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]).cpu()
            )

        return hook

    handles = [layer0.eh_proj.register_forward_pre_hook(eh_proj_hook)]
    for name, module in dense_targets.items():
        handles.append(module.register_forward_pre_hook(make_dense_hook(name)))
    handles.append(experts_module.register_forward_pre_hook(experts_hook, with_kwargs=True))
    handles.append(layer1.mixer.shared_experts.up_proj.register_forward_pre_hook(make_shared_hook("up_proj")))
    handles.append(layer1.mixer.shared_experts.down_proj.register_forward_pre_hook(make_shared_hook("down_proj")))

    print(f"Running MTP calibration forward pass ({len(calib_ids)} chunks) ...", flush=True)
    with torch.no_grad():
        for n, ids in enumerate(calib_ids):
            ids = ids.to(args.device)
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
            # Real (hidden@p, token@p+1) pairs, matching mtp_step's own
            # fusion at inference -- drop the last position (no real "next
            # token" to pair it with).
            hidden_trunc = hidden[:, :-1]
            tokens_next = ids[:, 1:]
            e = layer0.enorm(model.model.embeddings(tokens_next))
            h = layer0.hnorm(hidden_trunc)
            x = layer0.eh_proj(torch.cat([e, h], dim=-1))
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            x = layer0(x, position_ids=position_ids, use_cache=False)
            layer1(x, position_ids=position_ids, use_cache=False)
            print(f"  calibration pass {n + 1}/{len(calib_ids)} done", flush=True)

    for h in handles:
        h.remove()

    dense_acts = {name: torch.cat(v, dim=0) for name, v in dense_captured.items() if v}
    expert_inputs = {e: torch.cat(v, dim=0) for e, v in moe_expert_inputs.items() if v}
    shared_inputs = {k: torch.cat(v, dim=0) for k, v in moe_shared_inputs.items()}
    eh_proj_act = torch.cat(eh_proj_captured, dim=0)

    print("Quantizing eh_proj ...", flush=True)
    result = gptq_nbit(
        layer0.eh_proj.weight.detach().to(torch.float32).cpu(), eh_proj_act,
        bits=eh_bits, group_size=args.group_size, device="cpu", scheme="affine",
    )
    layer0.eh_proj.weight.data.copy_(result["W_hat"].to(layer0.eh_proj.weight.dtype))

    print("Quantizing layer0 (attention) ...", flush=True)
    quantized0 = quantize_dense_block(
        layer0, pattern[0], dense_acts, args.bits, args.group_size,
        component_recipe=block_component_recipe,
    )
    print(f"  {quantized0}", flush=True)

    print("Quantizing layer1 (moe) ...", flush=True)
    stats1 = quantize_moe_block(
        layer1, expert_inputs, shared_inputs, args.min_expert_tokens, args.gptq_device,
        args.moe_subbatch, args.bits, args.group_size,
        component_recipe=block_component_recipe,
    )
    print(f"  quantized={stats1['quantized_experts']} skipped={stats1['skipped']} "
          f"up_bits={stats1['up_bits']} down_bits={stats1['down_bits']}", flush=True)

    print(f"Saving GPTQ-corrected mtp.* weights to {args.output} ...", flush=True)
    out_state = {}
    for k, v in layers.state_dict().items():
        if k.endswith(".mixer.experts.up_proj") or k.endswith(".mixer.experts.down_proj"):
            # Undo the load-time stacking -- inject_mtp_weights.py (via
            # mlx_lm's Model.sanitize()) expects the checkpoint's own
            # per-expert layout and re-stacks it itself on the MLX side.
            prefix = k[: -len(".up_proj")] if k.endswith(".up_proj") else k[: -len(".down_proj")]
            name = "up_proj" if k.endswith(".up_proj") else "down_proj"
            for e in range(v.shape[0]):
                out_state[f"mtp.layers.{prefix}.{e}.{name}.weight"] = v[e].contiguous()
        else:
            out_state[f"mtp.layers.{k}"] = v.contiguous()
    save_file(out_state, args.output)
    print(f"Saved {len(out_state)} tensors -> {args.output}")
    print("GPTQ_MTP_DONE")


if __name__ == "__main__":
    main()
