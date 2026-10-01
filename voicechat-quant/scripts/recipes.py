"""Named component-type bit recipes for the VoiceChat LLM (dense NemotronH 9B).

Selected with `gptq_llm.py --component-recipe NAME`, never by code branches.
The "jang" family is nemotron-extreme-quant's recipe (poc/mlx_convert_recipe.py
COMPONENT_BIT_RECIPES["jang"], FINDINGS §1.3: bits by component type,
uniformly across layers), mapped onto the dense Nano-9B-v2 module names:

| jang (30B-A3B MoE) component     | VoiceChat LLM module (MLX path)          |
|----------------------------------|------------------------------------------|
| attention q/k/v/o_proj           | stt_model.llm.layers.N.mixer.{q,k,v,o}_proj |
| mamba in_proj/out_proj           | stt_model.llm.layers.N.mixer.{in,out}_proj  |
| moe_routed_up (switch_mlp.fc1)   | stt_model.llm.layers.N.mixer.up_proj      |
| moe_routed_down (switch_mlp.fc2) | stt_model.llm.layers.N.mixer.down_proj    |
| moe_shared                       | (none: no MoE)                            |
| lm_head                          | stt_model.lm_head and stt_model.function_head |
| embeddings                       | stt_model.embed_tokens                    |

"jang-voicechat" keeps jang's asymmetric up 4 / down 3 split for the dense
MLP (it is the largest parameter pool here too, ~36%) and uses 6-bit for the
two 131k-vocab heads and the embeddings (jang has lm_head 8).
"""

COMPONENT_PATTERNS = {
    "attention": r"mixer\.(q|k|v|o)_proj$",
    "mamba": r"mixer\.(in|out)_proj$",
    "mlp_up": r"mixer\.up_proj$",
    "mlp_down": r"mixer\.down_proj$",
}

COMPONENT_BIT_RECIPES = {
    "jang-voicechat": {
        "attention": 8,
        "mamba": 6,
        "mlp_up": 4,
        "mlp_down": 3,
        "lm_head": 6,
        "embeddings": 6,
    },
    # the original MoE jang tiers mapped 1:1 (lm_head 8), for comparison
    "jang": {
        "attention": 8,
        "mamba": 6,
        "mlp_up": 4,
        "mlp_down": 3,
        "lm_head": 8,
        "embeddings": 6,
    },
}


def recipe_overrides(name: str) -> tuple[list[str], int, int]:
    """-> (--override strings, head_bits, embed_bits) for gptq_llm.py."""
    r = COMPONENT_BIT_RECIPES[name]
    overrides = [f"{COMPONENT_PATTERNS[c]}={r[c]}" for c in COMPONENT_PATTERNS if c in r]
    return overrides, r["lm_head"], r["embeddings"]
