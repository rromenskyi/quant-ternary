"""Shared helpers for the Gemma 4 splice scripts (E4B and 26B-A4B).

Leftover RTN: GPTQ calibration only corrects the Linears it hooks (attention,
MLP, experts, the vision/audio encoder layers). Anything else that is a real
quantizable module in mlx-lm but was never a calibration target (e.g. the
audio tower's relative_k_proj / output_proj / subsample input projection on
E4B, embed_vision.embedding_projection on 26B) used to be copied through as
bf16 -- found by gemma4_smoke_test.py on the published E4B JANG (14 bf16
audio tensors). These get plain RTN at --rtn-leftovers-bits instead, so
nothing large is left in bf16.

"Quantizable" is decided against mlx-lm's own module tree built from
config.json (lazily, no weights), not by tensor shape: a 2D raw parameter
that isn't an nn.Linear/Embedding would break at load if its weight were
packed. Exceptions that must stay float go through --keep-float REGEX
(e.g. the MoE router, deliberately unquantized; vision
patch_embedder.input_proj, whose forward casts pixels to
`input_proj.weight.dtype`, which becomes uint32 once quantized).

Dead KV-shared weights: on models with num_kv_shared_layers > 0 (E4B), the
checkpoint still stores k_proj / v_proj / k_norm for the shared layers, but
mlx-lm's sanitize() drops them at load. --drop-kv-shared-dead leaves them out
of the output instead of writing ~110MB of never-loaded bf16.
"""

from __future__ import annotations

import re
from typing import Iterable


def module_path_of(raw_key: str) -> str:
    """Raw checkpoint key (without '.weight') -> mlx-lm internal module path.
    The only rename sanitize() does for these is text's extra ".model" hop."""
    path = raw_key.removesuffix(".weight").removeprefix("model.")
    if path.startswith("language_model."):
        path = path.replace("language_model.", "language_model.model.", 1)
    return path


def quantizable_module_paths(config: dict) -> set[str]:
    """Module paths that mlx-lm would quantize (have to_quantized) in the
    model built from this config. Built lazily: parameters are never
    evaluated, so this costs no real memory even for the 26B."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.utils import _get_classes

    model_class, args_class = _get_classes(config)
    model = model_class(args_class.from_dict(config))
    leaves = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    return {name for name, m in leaves if hasattr(m, "to_quantized")}


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


def is_leftover(
    raw_key: str,
    shape: list[int],
    dtype: str,
    quantizable: set[str],
    min_elems: int,
    keep_float: list[re.Pattern],
) -> bool:
    if not raw_key.endswith(".weight") or len(shape) != 2:
        return False
    if dtype not in ("BF16", "F16", "F32"):
        return False
    if shape[0] * shape[1] < min_elems:
        return False
    if any(p.search(raw_key) for p in keep_float):
        return False
    return module_path_of(raw_key) in quantizable


KV_SHARED_DEAD = re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm)\.")


def is_dead_kv_shared(raw_key: str, config: dict) -> bool:
    text = config.get("text_config", config)
    n_shared = text.get("num_kv_shared_layers") or 0
    if n_shared <= 0:
        return False
    m = KV_SHARED_DEAD.match(raw_key)
    return bool(m) and int(m.group(1)) >= text["num_hidden_layers"] - n_shared


def add_leftover_args(parser) -> None:
    parser.add_argument(
        "--rtn-leftovers-bits", type=int, default=8,
        help="RTN bits for quantizable Linears GPTQ never calibrated (0 = leave them float)",
    )
    parser.add_argument(
        "--leftover-min-elems", type=int, default=100_000,
        help="only RTN leftovers with at least this many elements",
    )
    parser.add_argument(
        "--keep-float", action="append", default=[], metavar="REGEX",
        help="raw checkpoint keys matching this stay float (repeatable), e.g. 'router\\.proj'",
    )
    parser.add_argument(
        "--drop-kv-shared-dead", action="store_true",
        help="omit k_proj/v_proj/k_norm of KV-shared layers (mlx-lm drops them at load anyway)",
    )
