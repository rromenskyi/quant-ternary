"""Local, no-GPU, no-checkpoint-download correctness test for the MoE-expert
GPTQ calibration path in gemma4_26b_gptq_calibrate.py.

Builds a tiny real `Gemma4TextRouter` + `Gemma4TextExperts` (from the
actually-installed transformers==5.17.0 source, not a mock), registers the
same hook the production script registers on `layer.experts`, runs a handful
of random forward passes through router->experts to populate real routing
data, then calls the SAME `run_moe_layer_gptq` helper the production script
calls on a real 51.6GB checkpoint. Asserts:
  (a) it runs without crashing
  (b) corrected gate_up_proj/down_proj have the exact same shape as the originals
  (c) corrected weights are NOT bit-identical to the originals (GPTQ changed something)
  (d) router.* parameters are byte-identical before/after (never touched)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma4_26b_gptq_calibrate import moe_hook_factory, run_moe_layer_gptq  # noqa: E402

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig  # noqa: E402
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts, Gemma4TextRouter  # noqa: E402


def build_tiny_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        hidden_size_per_layer_input=0,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=8,
        hidden_activation="gelu_pytorch_tanh",
    )


def main() -> None:
    torch.manual_seed(0)
    config = build_tiny_config()

    router = Gemma4TextRouter(config)
    experts = Gemma4TextExperts(config)
    torch.nn.init.normal_(experts.gate_up_proj, std=0.02)
    torch.nn.init.normal_(experts.down_proj, std=0.02)
    router.eval()
    experts.eval()

    # Snapshot router params (byte-identical check target (d)).
    router_before = {k: v.detach().clone() for k, v in router.state_dict().items()}
    gate_up_before = experts.gate_up_proj.detach().clone()
    down_before = experts.down_proj.detach().clone()

    layer_idx = 0
    expert_rows: dict[tuple[int, int], list[torch.Tensor]] = {}
    row_counts: dict[tuple[int, int], int] = {}
    hook = moe_hook_factory(
        layer_idx=layer_idx,
        expert_rows=expert_rows,
        row_counts=row_counts,
        max_rows_per_expert=64,
        num_experts=experts.num_experts,
    )
    handle = experts.register_forward_pre_hook(hook)

    # Run several random "calibration examples" through router -> experts,
    # exactly mirroring the real forward pass's own call shape:
    #   _, top_k_weights, top_k_index = router(hidden_states_flat)
    #   experts(hidden_states_2, top_k_index, top_k_weights)
    num_examples = 12
    tokens_per_example = 20
    with torch.no_grad():
        for _ in range(num_examples):
            hidden_states_flat = torch.randn(tokens_per_example, config.hidden_size)
            _, top_k_weights, top_k_index = router(hidden_states_flat)
            hidden_states_2 = torch.randn(tokens_per_example, config.hidden_size)  # stand-in for pre_feedforward_layernorm_2 output
            experts(hidden_states_2, top_k_index, top_k_weights)

    handle.remove()

    total_captured = sum(row_counts.values())
    assert total_captured > 0, "hook captured no rows at all -- routing/bucketing logic is broken"
    hit_experts = {e for (_li, e) in expert_rows.keys()}
    print(f"captured {total_captured} rows across {len(hit_experts)}/{experts.num_experts} experts")

    rows_per_expert: dict[int, torch.Tensor] = {}
    for e in range(experts.num_experts):
        parts = expert_rows.get((layer_idx, e), [])
        rows_per_expert[e] = torch.cat(parts, dim=0) if parts else torch.zeros(0, experts.hidden_dim)

    # (a) runs without crashing
    result = run_moe_layer_gptq(
        experts_module=experts,
        rows_per_expert=rows_per_expert,
        bits=4,
        group_size=8,
        device="cpu",
    )

    gate_up_hat = result["gate_up_proj"]
    down_hat = result["down_proj"]

    # (b) exact same shape as originals
    assert gate_up_hat.shape == gate_up_before.shape, (gate_up_hat.shape, gate_up_before.shape)
    assert down_hat.shape == down_before.shape, (down_hat.shape, down_before.shape)
    print(f"gate_up_proj shape {tuple(gate_up_hat.shape)}, down_proj shape {tuple(down_hat.shape)} -- match originals")

    # (c) corrected weights are NOT bit-identical to the originals
    gate_up_changed = not torch.equal(gate_up_hat, gate_up_before.to(torch.bfloat16))
    down_changed = not torch.equal(down_hat, down_before.to(torch.bfloat16))
    assert gate_up_changed, "gate_up_proj is bit-identical to the original -- GPTQ correction did nothing"
    assert down_changed, "down_proj is bit-identical to the original -- GPTQ correction did nothing"
    print("gate_up_proj and down_proj both changed after GPTQ correction, as expected")

    # Only experts with at least one captured row should meaningfully change;
    # sanity-check at least the hit experts changed noticeably.
    for e in hit_experts:
        diff = (gate_up_hat[e].float() - gate_up_before[e].float()).abs().max().item()
        assert diff > 0, f"expert {e} was routed calibration tokens but its gate_up_proj did not change at all"

    # (d) router.* parameters are byte-identical before/after
    router_after = router.state_dict()
    assert set(router_after.keys()) == set(router_before.keys())
    for k in router_before:
        assert torch.equal(router_before[k], router_after[k]), f"router.{k} changed -- router must never be touched"
    print(f"router.* untouched: {sorted(router_before.keys())}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
