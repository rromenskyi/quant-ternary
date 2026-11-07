"""End-to-end smoke test for gptq_dory.py on a tiny random checkpoint, CPU
only: catches control-flow bugs (the input/recurrent/output staging, the
loop-resolution step, resume) before ever touching a real checkpoint or a
pod. Not a numerical-correctness check (that's crosscheck_reference.py).

    python smoke_test_gptq.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, ".")
from dory_reference import DoryConfig, DoryModel

VOCAB, HID, INTER = 97, 32, 48
CFG = dict(
    vocab_size=VOCAB, hidden_size=HID, intermediate_size=INTER, num_hidden_layers=16,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=6,
    n_input_layers=4, n_recurrent_layers=8, n_output_layers=4, n_recurrent_loops=2,
)


class _FakeTok:
    def __call__(self, text, return_tensors, truncation, max_length):
        ids = [b % VOCAB for b in text.encode()[:max_length]]
        ids += [1] * (max_length - len(ids))
        return {"input_ids": torch.tensor([ids[:max_length]])}


def build_snapshot(root: Path):
    torch.manual_seed(0)
    model = DoryModel(DoryConfig(**CFG))
    state = {}
    for k, v in model.state_dict().items():
        if "inv_freq" in k:
            continue
        key = "lm_head.logit_scale" if k == "logit_scale" else f"backbone.{k}"
        state[key] = v.to(torch.bfloat16).contiguous()
    root.mkdir(parents=True, exist_ok=True)
    save_file(state, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps({**CFG, "rope_parameters": {}}))


def main():
    work_root = Path(tempfile.mkdtemp(prefix="dory_smoke_"))
    snap, calib, work = work_root / "snap", work_root / "calib.txt", work_root / "work"
    build_snapshot(snap)
    calib.write_text("the quick brown fox jumps over the lazy dog " * 50)

    import gptq_dory

    gptq_dory.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: _FakeTok())

    argv = ["gptq_dory.py", "--model", str(snap), "--work", str(work), "--recipe", "attn=8,mlp=6",
            "--calib", f"text:{calib}", "--device", "cpu", "--calib-chunks", "2",
            "--calib-chunk-tokens", "16", "--group-size", "8"]

    # Full run.
    sys.argv = argv
    gptq_dory.main()
    n_layers = CFG["num_hidden_layers"]
    files = sorted((work / "layers").glob("*.safetensors"))
    assert len(files) == n_layers, f"expected {n_layers} layer files, got {len(files)}"
    print(f"full run OK: {len(files)} layer files")

    # Resume: wipe progress + the last 3 layers' files, rerun, confirm it
    # picks up where it left off instead of silently redoing (or skipping)
    # work, across the input/recurrent boundary included.
    shutil.rmtree(work)
    build_snapshot(snap)  # fresh float weights (the first run quantized snap's in place)
    sys.argv = argv
    gptq_dory.main()
    done_full = json.load(open(work / "progress.json"))["done"]
    assert done_full == n_layers - 1

    for f in sorted((work / "layers").glob("*.safetensors"))[-3:]:
        f.unlink()
    json.dump({"done": n_layers - 4, "layers": n_layers,
               "made_with": json.load(open(work / "progress.json"))["made_with"]},
              open(work / "progress.json", "w"))
    sys.argv = argv
    gptq_dory.main()
    files = sorted((work / "layers").glob("*.safetensors"))
    assert len(files) == n_layers, f"resume: expected {n_layers} layer files, got {len(files)}"
    print(f"resume OK: {len(files)} layer files after truncate-and-resume")

    shutil.rmtree(work_root)
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
