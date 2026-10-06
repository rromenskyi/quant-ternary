"""convert_mlx.pack against mx.quantize: the packing the exact-code rewrite
relies on (docs/GPTQ_EXACT_CODES.md).

    python test_pack.py   # needs mlx
"""

import mlx.core as mx
import numpy as np

from convert_mlx import pack


def test_pack_matches_mlx() -> None:
    mx.random.seed(0)
    for bits in (2, 3, 4, 5, 6, 8):
        w = mx.random.normal((16, 256)).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=64, bits=bits)
        d = mx.dequantize(q, s, b, group_size=64, bits=bits)
        S = np.repeat(np.array(s.astype(mx.float32)), 64, axis=-1)
        B = np.repeat(np.array(b.astype(mx.float32)), 64, axis=-1)
        codes = np.rint((np.array(d.astype(mx.float32)) - B) / S).astype(np.int64)
        assert np.array_equal(np.array(q), pack(codes, bits)), bits


if __name__ == "__main__":
    test_pack_matches_mlx()
    print("pack matches mx.quantize for 2, 3, 4, 5, 6, 8 bits")
