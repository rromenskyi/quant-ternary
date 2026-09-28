"""In-memory variant transforms for NemotronLabs VoiceChat 11B (mlx-audio fork).

Every speed/quality variant in this project is a composition of these
transforms, selected by CLI flags that `vc_eval.py` (and the pod pipeline)
pass through verbatim -- no per-variant special cases in the code.

  --scales-dtype bf16      cast float32 quantization scales/biases to bf16
  --act-dtype bf16         run perception / LLM activations in bf16 instead
                           of the float32 they inherit from the mel frontend
  --rtn PREFIX:BITS:GROUP  round-to-nearest quantize every float Linear (and
                           1x1 Conv1d, rewritten as Linear) under PREFIX;
                           repeatable, e.g. --rtn stt_model.perception:8:64
  --rtn-skip REGEX         never quantize modules whose path matches REGEX
                           (repeatable; e.g. a head that must stay float)
"""

from __future__ import annotations

import re

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map_with_path

DTYPES = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}


def add_variant_args(ap) -> None:
    ap.add_argument("--scales-dtype", choices=sorted(DTYPES), default=None)
    ap.add_argument("--act-dtype", choices=sorted(DTYPES), default=None)
    ap.add_argument("--rtn", action="append", default=[], metavar="PREFIX:BITS:GROUP")
    ap.add_argument("--rtn-skip", action="append", default=[], metavar="REGEX")
    ap.add_argument("--tts-mog-gather", action="store_true",
                    help="exact: compute only the sampled mixture's proj_mus rows")
    ap.add_argument("--compile-tts-codes", action="store_true",
                    help="mx.compile the per-frame TTS code generation")
    ap.add_argument(
        "--rtn-mode", default="affine", help="mx.quantize mode for --rtn (affine|mxfp4|...)"
    )


class PointwiseConvAsLinear(nn.Module):
    """A kernel-size-1 Conv1d expressed as a Linear so it can be quantized.

    nn.Conv1d weight is (out, 1, in) in MLX layout; Linear wants (out, in).
    Input/output layout (B, T, C) is the same for both.
    """

    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        w = conv.weight
        self.linear = nn.Linear(w.shape[2], w.shape[0], bias="bias" in conv)
        self.linear.weight = w.reshape(w.shape[0], w.shape[2])
        if "bias" in conv:
            self.linear.bias = conv.bias

    def __call__(self, x):
        return self.linear(x)


def _rewrite_pointwise_convs(model: nn.Module, prefix: str) -> int:
    count = 0
    for path, module in list(model.named_modules()):
        if not path.startswith(prefix):
            continue
        for name, child in list(module.children().items()):
            if (
                isinstance(child, nn.Conv1d)
                and child.weight.shape[1] == 1
                and getattr(child, "groups", 1) == 1
                and child.weight.shape[2] > 1
            ):
                setattr(module, name, PointwiseConvAsLinear(child))
                count += 1
    return count


def cast_quant_scales(model: nn.Module, dtype) -> int:
    count = 0

    def fn(path, value):
        nonlocal count
        if path.endswith((".scales", ".biases")) and value.dtype == mx.float32:
            count += 1
            return value.astype(dtype)
        return value

    model.update(tree_map_with_path(fn, model.parameters()))
    return count


def apply_rtn(model: nn.Module, specs: list[str], skips: list[str], mode: str) -> dict:
    report = {}
    skip_res = [re.compile(s) for s in skips]
    for spec in specs:
        prefix, bits, group = spec.rsplit(":", 2)
        bits, group = int(bits), int(group)
        rewritten = _rewrite_pointwise_convs(model, prefix)
        quantized = []

        def pred(path, module):
            if not path.startswith(prefix) or any(r.search(path) for r in skip_res):
                return False
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                return False
            if module.weight.shape[-1] % group:
                return False
            quantized.append(path)
            return True

        nn.quantize(model, group_size=group, bits=bits, class_predicate=pred, mode=mode)
        report[spec] = {"modules": len(quantized), "pointwise_convs_rewritten": rewritten}
    return report


def apply_variant(model, args) -> dict:
    report = {}
    if args.scales_dtype:
        report["scales_cast"] = cast_quant_scales(model, DTYPES[args.scales_dtype])
    if args.rtn:
        report["rtn"] = apply_rtn(model, args.rtn, args.rtn_skip, args.rtn_mode)
    patch_tts_fast(args.tts_mog_gather, args.compile_tts_codes)
    mx.eval(model.parameters())
    report["param_gb"] = round(
        sum(v.size * v.itemsize for _, v in tree_flatten(model.parameters())) / 1e9, 3
    )
    return report


def _selected_means(proj, inputs, idx, num_predictions, low_rank):
    """proj(inputs) reshaped to (.., P, R) and gathered at idx -- but computing
    only the R rows of the selected mixture instead of all P*R outputs."""
    b, l, h = inputs.shape
    flat_idx = idx.reshape(-1)
    if isinstance(proj, nn.QuantizedLinear):
        w = proj.weight.reshape(num_predictions, low_rank, -1)[flat_idx]
        s = proj.scales.reshape(num_predictions, low_rank, -1)[flat_idx]
        bi = proj.biases.reshape(num_predictions, low_rank, -1)[flat_idx]
        w = mx.dequantize(
            w.reshape(-1, w.shape[-1]), s.reshape(-1, s.shape[-1]), bi.reshape(-1, bi.shape[-1]),
            group_size=proj.group_size, bits=proj.bits,
        ).reshape(b * l, low_rank, h)
    else:
        w = proj.weight.reshape(num_predictions, low_rank, h)[flat_idx]
    x = inputs.reshape(b * l, h, 1).astype(w.dtype)
    return (w @ x).reshape(b, l, low_rank)


FAST = {"mog_gather": False, "compile_codes": False}
_installed = False


def patch_tts_fast(mog_gather: bool, compile_codes: bool) -> None:
    """Exact rewrites of the TTS code generator (no weight changes).

    mog_gather: MoGHead.infer computes proj_mus for all 1024 mixtures (a
      1152 x 65536 mat-vec, 150 MB of bf16 read 8x per frame) and then keeps
      one; compute only the selected mixture's 64 rows instead.
    compile_codes: mx.compile the whole per-frame code generation
      (8 MoG iterations x 31 RVQ steps of small kernels), with the random
      state threaded through so the sampled codes are unchanged.
    """
    global _installed
    FAST.update(mog_gather=mog_gather, compile_codes=compile_codes)
    if _installed:
        return
    _installed = True
    from mlx_audio.lm.sample_utils import apply_top_p
    from mlx_audio.sts.models.nemotron_voicechat import tts as ttsmod

    orig_infer = ttsmod.MoGHead.infer
    if True:

        def infer(self, inputs, *, guidance_scale, top_p):
            if not FAST["mog_gather"]:
                return orig_infer(self, inputs, guidance_scale=guidance_scale, top_p=top_p)
            for layer in self.mlp_stack:
                inputs = layer(inputs)
            if guidance_scale > 0:
                conditional, unconditional = mx.split(inputs, 2, axis=0)
                inputs = conditional + guidance_scale * (conditional - unconditional)
            logits = self.proj_logits(inputs)
            log_probabilities = mx.log(mx.softmax(logits, axis=-1))
            if 0 < top_p < 1:
                log_probabilities = apply_top_p(log_probabilities, top_p)
            mixture_indices = mx.random.categorical(log_probabilities)
            selected_means = _selected_means(
                self.proj_mus, inputs, mixture_indices, self.num_predictions, self.low_rank
            ).astype(inputs.dtype)
            selected_projection = self.low_mat[mixture_indices]
            means = mx.einsum("btol,btl->bto", selected_projection, selected_means)
            residual = self.proj_else(inputs)
            log_stds = mx.maximum(self.proj_logs(inputs), self.min_log_std)
            return means * mx.exp(log_stds) + residual, log_stds

        ttsmod.MoGHead.infer = infer

    if True:
        orig = ttsmod.EARTTSModel._generate_codes
        compiled = {}

        def generate_codes(self, hidden, *, guidance_enabled):
            if not FAST["compile_codes"]:
                return orig(self, hidden, guidance_enabled=guidance_enabled)
            key = (id(self), guidance_enabled, FAST["mog_gather"])
            if key not in compiled:
                fn = lambda h: orig(self, h, guidance_enabled=guidance_enabled)  # noqa: E731
                compiled[key] = mx.compile(fn, inputs=[mx.random.state], outputs=[mx.random.state])
            return compiled[key](hidden)

        ttsmod.EARTTSModel._generate_codes = generate_codes


def patch_activation_dtype(dtype) -> None:
    """Cast the two float32 entry points (mel -> conformer, fused -> LLM).

    Class-level, so it also covers the system-prompt prefill that runs inside
    the session constructor (the LLM caches must not start out float32).
    """
    if dtype is None:
        return
    dt = DTYPES[dtype]
    from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState
    from mlx_audio.sts.models.nemotron_voicechat.streaming import (
        VoiceChatStreamingSession,
    )

    push = ConformerStreamingState.push
    ConformerStreamingState.push = lambda self, mel, **k: push(self, mel.astype(dt), **k)
    lang = VoiceChatStreamingSession._language_step
    VoiceChatStreamingSession._language_step = lambda self, x: lang(self, x.astype(dt))
