"""Repack a reference F16 GGUF's tensors into a mixed-precision file:
Q8_0 for Mamba in/out_proj + mlp.down_proj (majority of params, tolerates
8-bit fine per quality testing), TQ2_0 for attention.o_proj (only
TQ2_0-eligible tensor — hidden_size=3136 isn't divisible by 256, blocking
TQ2_0 for everything else), Q4_0 fallback for mlp.up_proj/attn q/k/v
(in_features=3136, not divisible by 256 so TQ2_0/TQ1_0 are unavailable —
same constraint NVIDIA's own official Q4_K_M GGUF release presumably hits,
hence their K-quant choice over a pure ternary/binary format), and F16 kept
for everything else (embeddings, lm_head, norms, SSM scalar params).
"""
import numpy as np
import gguf
from gguf.quants import quantize
from gguf.constants import GGMLQuantizationType, GGUFValueType

SRC = "/root/nano4b_ref_f16.gguf"
DST = "/root/nano4b-mixed.gguf"

Q8_SUFFIXES = ("ssm_in.weight", "ssm_out.weight", "ffn_down.weight")
Q8_EXACT_NAMES = ("token_embd.weight", "output.weight")
TQ2_SUFFIXES = ("attn_output.weight",)
Q4_SUFFIXES = ("ffn_up.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight")


def classify(name: str):
    if name in Q8_EXACT_NAMES:
        return GGMLQuantizationType.Q8_0
    if any(name.endswith(s) for s in TQ2_SUFFIXES):
        return GGMLQuantizationType.TQ2_0
    if any(name.endswith(s) for s in Q8_SUFFIXES):
        return GGMLQuantizationType.Q8_0
    if any(name.endswith(s) for s in Q4_SUFFIXES):
        return GGMLQuantizationType.Q4_0
    return None  # keep original dtype


EXPERT_KEY_SUFFIXES = (
    "expert_count", "expert_used_count", "expert_group_count", "expert_group_used_count",
    "expert_feed_forward_length", "expert_shared_feed_forward_length", "expert_shared_count",
    "expert_weights_norm", "expert_weights_scale",
)


def main():
    reader = gguf.GGUFReader(SRC)
    old_arch = reader.fields["general.architecture"].contents()
    # convert_hf_to_gguf.py's has_moe_params check is `"num_experts_per_tok" in llm_config`,
    # which is always true for NemotronHConfig (class-level default, not conditional on the
    # original checkpoint actually having MoE) — mistags every dense nemotron_h model as
    # nemotron_h_moe. The tensor names it produces are correct either way (confirmed by
    # inspection); only the architecture string + expert_* hparam KVs are wrong, and the
    # dense "nemotron_h" loader expects a plain (non-namespaced) "nemotron_h.*" hparam
    # prefix — this model genuinely has zero MoE tensors, so force it back.
    new_arch = "nemotron_h" if old_arch == "nemotron_h_moe" else old_arch
    writer = gguf.GGUFWriter(DST, arch=new_arch)

    skip_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for key, field in reader.fields.items():
        if key in skip_keys:
            continue
        if any(key.endswith(s) for s in EXPERT_KEY_SUFFIXES):
            continue
        if key.startswith(f"{old_arch}.") and new_arch != old_arch:
            key = f"{new_arch}." + key[len(old_arch) + 1:]
        main_type = field.types[0]
        if main_type == GGUFValueType.ARRAY:
            writer.add_array(key, field.contents())
        else:
            writer.add_key_value(key, field.contents(), main_type)

    total_bytes = 0
    orig_bytes = 0
    counts = {"Q8_0": 0, "TQ2_0": 0, "Q4_0": 0, "F16": 0}
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data  # numpy view, shape=(out,in) row-major, dtype matches tensor_type
        orig_bytes += tensor.n_bytes
        target = classify(name)
        if target is None:
            writer.add_tensor(name, data)
            total_bytes += tensor.n_bytes
            counts["F16"] += 1
            continue

        arr_f32 = data.astype(np.float32) if data.dtype != np.float32 else data
        try:
            packed = quantize(arr_f32, target)
        except Exception as e:
            print(f"  WARN: {name} failed {target.name} ({e}), falling back to Q4_0")
            packed = quantize(arr_f32, GGMLQuantizationType.Q4_0)
            target = GGMLQuantizationType.Q4_0
        writer.add_tensor(name, packed, raw_dtype=target)
        total_bytes += packed.nbytes
        counts[target.name] += 1
        print(f"  {name}: {tuple(arr_f32.shape)} -> {target.name}, {tensor.n_bytes} -> {packed.nbytes} bytes")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(f"\nTensor type counts: {counts}")
    print(f"Original (F16) total: {orig_bytes/1e9:.3f} GB")
    print(f"Packed total: {total_bytes/1e9:.3f} GB")
    print(f"Reduction: {orig_bytes/total_bytes:.2f}x")


if __name__ == "__main__":
    main()
