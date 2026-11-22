"""gptq_inkling2.py's layer-by-layer driver (step, later_passes) against
the transformers text model's own forward, float weights, no calibration.

    python test_driver.py --model /tmp/tiny-mini
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForImageTextToText

import gptq_inkling2 as g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    g.DEVICE = "cpu"
    lm = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().model.language_model
    cfg = lm.config
    ids = torch.randint(0, 200000, (1, 64), generator=torch.Generator().manual_seed(0))
    kwargs = g.layer_kwargs(lm, ids)
    with torch.no_grad():
        want = lm(input_ids=ids, use_cache=False).last_hidden_state
        h = lm.embed_tokens(ids)
        state, pre = (h, h.new_zeros(*h.shape[:-1], 0, h.shape[-1])), None
        loop = getattr(cfg, "loop_layer_index_start", None)
        for i, layer in enumerate(lm.layers):
            if i == loop:
                pre = [state]
            state = g.step(layer, state, kwargs[i])
            if loop is not None and i == cfg.loop_layer_index_end - 1:
                state = g.later_passes(lm, [state], pre, kwargs, cfg)[0]
        h, bs = state
        got = lm.norm(lm.final_input(bs, h)[0])
    diff = (got - want).abs().max().item()
    print(f"driver vs forward: max |diff| {diff:.3g}")
    if diff > 1e-4:
        raise SystemExit("the driver does not run the model as its forward does")
    print("DRIVER_OK")


if __name__ == "__main__":
    main()
