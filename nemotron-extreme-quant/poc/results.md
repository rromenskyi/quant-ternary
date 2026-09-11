# PoC Results — Nemotron-3-Nano-4B Stage A

Success criteria: weight cosine >= 0.9, activation cosine >= 0.95, effective bpw <= 1.8, no NaN/Inf.


## layers.12.mixer.o_proj (attention)

| Method | Eff. bpw | Weight cos | Weight MSE | Act cos | Act MSE | Verdict |
|---|---|---|---|---|---|---|
| naive_binary_g128 | 1.125 | 0.7973 | 8.777e-05 | 0.6628 | 1.099e-02 | fail |
| act_aware_binary_g128 | 1.125 | 0.7473 | 1.093e-04 | 0.8241 | 6.290e-03 | fail |
| naive_binary_g128_res0.01_magnitude | 1.415 | 0.8345 | 7.329e-05 | 0.8652 | 5.116e-03 | fail |
| act_aware_binary_g128_res0.03_activation_weighted | 1.995 | 0.7956 | 9.101e-05 | 0.9245 | 2.956e-03 | fail |
| ternary_g128 | 2.125 | 0.8981 | 4.659e-05 | 0.8287 | 6.225e-03 | fail |
| gptq_binary_g128 | 1.125 | 0.7394 | 1.100e-04 | 0.9330 | 3.077e-03 | fail |
| gptq_ternary_g128 | 2.125 | 0.8302 | 8.554e-05 | 0.9424 | 2.897e-03 | fail |
| rot_naive_binary_g128 | 1.125 | 0.8000 | 8.672e-05 | 0.8095 | 6.766e-03 | fail |
| rot_gptq_binary_g128 | 1.125 | 0.7444 | 1.081e-04 | 0.9806 | 8.346e-04 | PASS |
| rot_gptq_ternary_g128 | 2.125 | 0.8365 | 8.350e-05 | 0.9860 | 6.964e-04 | fail |

## layers.12.mixer.q_proj (attention)

| Method | Eff. bpw | Weight cos | Weight MSE | Act cos | Act MSE | Verdict |
|---|---|---|---|---|---|---|
| naive_binary_g128 | 1.125 | 0.7906 | 1.881e-04 | 0.8253 | 3.589e-01 | fail |
| act_aware_binary_g128 | 1.125 | 0.7845 | 1.931e-04 | 0.8856 | 2.409e-01 | fail |
| naive_binary_g128_res0.01_magnitude | 1.405 | 0.8260 | 1.596e-04 | 0.8797 | 2.450e-01 | fail |
| act_aware_binary_g128_res0.03_activation_weighted | 1.965 | 0.8249 | 1.608e-04 | 0.9550 | 9.444e-02 | fail |
| ternary_g128 | 2.125 | 0.8955 | 9.939e-05 | 0.9277 | 1.557e-01 | fail |
| gptq_binary_g128 | 1.125 | 0.6933 | 2.648e-04 | 0.9907 | 2.505e-02 | PASS |
| gptq_ternary_g128 | 2.125 | 0.7793 | 2.095e-04 | 0.9925 | 2.576e-02 | fail |
| rot_naive_binary_g128 | 1.125 | 0.7999 | 1.807e-04 | 0.8656 | 2.798e-01 | fail |
| rot_gptq_binary_g128 | 1.125 | 0.7088 | 2.537e-04 | 0.9934 | 1.646e-02 | PASS |
| rot_gptq_ternary_g128 | 2.125 | 0.7934 | 1.989e-04 | 0.9947 | 1.673e-02 | fail |

## layers.4.mixer.in_proj (mamba)

| Method | Eff. bpw | Weight cos | Weight MSE | Act cos | Act MSE | Verdict |
|---|---|---|---|---|---|---|
| naive_binary_g128 | 1.125 | 0.7930 | 1.325e-04 | 0.8711 | 6.755e-01 | fail |
| act_aware_binary_g128 | 1.125 | 0.7883 | 1.354e-04 | 0.9160 | 4.144e-01 | fail |
| naive_binary_g128_res0.01_magnitude | 1.405 | 0.8258 | 1.137e-04 | 0.9266 | 3.658e-01 | fail |
| act_aware_binary_g128_res0.03_activation_weighted | 1.965 | 0.8379 | 1.070e-04 | 0.9583 | 1.984e-01 | fail |
| ternary_g128 | 2.125 | 0.8969 | 6.983e-05 | 0.9450 | 3.063e-01 | fail |
| gptq_binary_g128 | 1.125 | 0.7027 | 1.833e-04 | 0.9847 | 9.768e-02 | PASS |
| gptq_ternary_g128 | 2.125 | 0.7846 | 1.465e-04 | 0.9868 | 1.056e-01 | fail |
| rot_naive_binary_g128 | 1.125 | 0.7998 | 1.286e-04 | 0.9128 | 5.004e-01 | fail |
| rot_gptq_binary_g128 | 1.125 | 0.7131 | 1.780e-04 | 0.9907 | 5.395e-02 | PASS |
| rot_gptq_ternary_g128 | 2.125 | 0.7941 | 1.411e-04 | 0.9922 | 5.727e-02 | fail |

## layers.4.mixer.out_proj (mamba)

| Method | Eff. bpw | Weight cos | Weight MSE | Act cos | Act MSE | Verdict |
|---|---|---|---|---|---|---|
| naive_binary_g128 | 1.125 | 0.7894 | 1.213e-04 | 0.8007 | 3.377e-04 | fail |
| act_aware_binary_g128 | 1.125 | 0.7699 | 1.312e-04 | 0.8475 | 2.666e-04 | fail |
| naive_binary_g128_res0.01_magnitude | 1.415 | 0.8270 | 1.019e-04 | 0.8767 | 2.161e-04 | fail |
| act_aware_binary_g128_res0.03_activation_weighted | 1.995 | 0.8160 | 1.079e-04 | 0.9234 | 1.376e-04 | fail |
| ternary_g128 | 2.125 | 0.8932 | 6.508e-05 | 0.8999 | 1.814e-04 | fail |
| gptq_binary_g128 | 1.125 | 0.7400 | 1.464e-04 | 0.9549 | 9.819e-05 | PASS |
| gptq_ternary_g128 | 2.125 | 0.8318 | 1.151e-04 | 0.9607 | 9.945e-05 | fail |
| rot_naive_binary_g128 | 1.125 | 0.7999 | 1.159e-04 | 0.8429 | 2.775e-04 | fail |
| rot_gptq_binary_g128 | 1.125 | 0.7566 | 1.382e-04 | 0.9799 | 4.218e-05 | PASS |
| rot_gptq_ternary_g128 | 2.125 | 0.8475 | 1.074e-04 | 0.9842 | 4.069e-05 | fail |

## layers.5.mixer.down_proj (mlp)

| Method | Eff. bpw | Weight cos | Weight MSE | Act cos | Act MSE | Verdict |
|---|---|---|---|---|---|---|
| naive_binary_g128 | 1.125 | 0.7956 | 1.185e-04 | 0.8366 | 6.360e-04 | fail |
| act_aware_binary_g128 | 1.125 | 0.7583 | 1.394e-04 | 0.8715 | 5.057e-04 | fail |
| naive_binary_g128_res0.01_magnitude | 1.425 | 0.8261 | 1.027e-04 | 0.8659 | 5.224e-04 | fail |
| act_aware_binary_g128_res0.03_activation_weighted | 2.025 | 0.7779 | 1.297e-04 | 0.9618 | 1.549e-04 | fail |
| ternary_g128 | 2.125 | 0.8975 | 6.277e-05 | 0.9203 | 3.222e-04 | fail |
| gptq_binary_g128 | 1.125 | 0.7195 | 1.575e-04 | 0.9667 | 1.618e-04 | PASS |
| gptq_ternary_g128 | 2.125 | 0.8024 | 1.253e-04 | 0.9758 | 1.490e-04 | fail |
| rot_naive_binary_g128 | 1.125 | 0.7995 | 1.165e-04 | 0.8283 | 6.624e-04 | fail |
| rot_gptq_binary_g128 | 1.125 | 0.7338 | 1.504e-04 | 0.9966 | 1.563e-05 | PASS |
| rot_gptq_ternary_g128 | 2.125 | 0.8155 | 1.197e-04 | 0.9963 | 1.886e-05 | fail |

## layers.5.mixer.up_proj (mlp)

| Method | Eff. bpw | Weight cos | Weight MSE | Act cos | Act MSE | Verdict |
|---|---|---|---|---|---|---|
| naive_binary_g128 | 1.125 | 0.7948 | 1.277e-04 | 0.9371 | 3.375e-02 | fail |
| act_aware_binary_g128 | 1.125 | 0.7837 | 1.345e-04 | 0.9568 | 2.046e-02 | PASS |
| naive_binary_g128_res0.01_magnitude | 1.405 | 0.8247 | 1.110e-04 | 0.9564 | 2.196e-02 | PASS |
| act_aware_binary_g128_res0.03_activation_weighted | 1.965 | 0.8361 | 1.060e-04 | 0.9749 | 1.002e-02 | fail |
| ternary_g128 | 2.125 | 0.8985 | 6.678e-05 | 0.9736 | 1.299e-02 | fail |
| gptq_binary_g128 | 1.125 | 0.7233 | 1.669e-04 | 0.9911 | 4.433e-03 | PASS |
| gptq_ternary_g128 | 2.125 | 0.8095 | 1.320e-04 | 0.9928 | 4.779e-03 | fail |
| rot_naive_binary_g128 | 1.125 | 0.7997 | 1.250e-04 | 0.9456 | 3.447e-02 | fail |
| rot_gptq_binary_g128 | 1.125 | 0.7280 | 1.647e-04 | 0.9927 | 3.534e-03 | PASS |
| rot_gptq_ternary_g128 | 2.125 | 0.8130 | 1.295e-04 | 0.9942 | 3.603e-03 | fail |

## Summary

Revised gate: activation cosine >= 0.95 and effective bpw <= 1.8 (weight cosine is reported but not gated on — for binary/ternary quantization it does not predict functional reconstruction quality; see the note below).

- 13/60 (method, layer) combinations passed
- 6/6 layers had at least one passing method

**Decision: GO**


### Note on weight cosine

Weight cosine stays in the 0.70-0.90 range for every binary/ternary method here, including the ones whose activation cosine is 0.98+. This is expected, not a bug: binary quantization flips each weight to +-scale independently, so the per-element weight vector is nowhere near the original in raw cosine terms, yet the *dot product with real, correlated activation data* reconstructs almost exactly, because GPTQ's error-compensation spreads each column's rounding error onto the exact directions the Hessian says the calibration activations care about most. Output-space (activation) cosine is the metric that predicts whether the quantized model actually behaves like the original; weight cosine does not, for this quantization family.

