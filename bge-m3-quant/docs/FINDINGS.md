# Findings: BGE-M3 -> GGUF (and MLX for LLMTray)

BAAI/bge-m3 at `5617a9f` (the revision LLMTray's registry pins too):
XLM-RoBERTa large, 568 M, `pytorch_model.bin` in fp32, CLS pooling
(`1_Pooling/config.json`), 8192-token context, 1024 dimensions. Dense
embeddings only here: the sparse and ColBERT heads (`sparse_linear.pt`,
`colbert_linear.pt`) have no place in a GGUF or in LLMTray's runtime.

Everything ran on k2 (CPU, no GPU): llama.cpp's prebuilt `b12751`
ubuntu-x64 binaries, the same tag's `convert_hf_to_gguf.py`.

## 1. GGUF builds (published as roman220220/bge-m3-GGUF, 2026-11-25)

Conversion: metadata equal to ggml-org/bge-m3-Q8_0-GGUF (pooling CLS,
context 8192) but for one flag the newer converter writes
(`tokenizer.ggml.clean_spaces`). Our Q8_0's tensors are ggml-org's: the
vectors agree at cosine 1.000000; the files differ by 32 bytes.

Measured through `llama-server --embedding` (`/v1/embeddings`) against
transformers fp32 (CLS, L2-normalized), on 12,039 texts: Tatoeba bitext
(5 languages x 1,000 pairs), 1,000 stdlib code pairs, 39 long modules
(5-6.5k tokens; one over 8192 left out). Cosine mean / min by task:

| build | MB | sentences | code | long | bitext top-1 | code top-1 | top-10 overlap |
|---|---|---|---|---|---|---|---|
| reference | 2271 | — | — | — | 0.9583 | 0.4970 | — |
| F16 | 1158 | 0.99999 / 0.9999 | 0.99999 / 0.9996 | 0.99994 / 0.9995 | 0.9583 | 0.4970 | 0.9963 |
| Q8_0 | 635 | 0.99958 / 0.9979 | 0.99931 / 0.9978 | 0.99828 / 0.9967 | 0.9587 | 0.4960 | 0.9712 |
| Q6_K | 499 | 0.99657 / 0.9836 | 0.99480 / 0.9870 | 0.99030 / 0.9874 | 0.9579 | 0.5000 | 0.9265 |
| Q5_K_M | 468 | 0.99385 / 0.9736 | 0.98928 / 0.9808 | 0.97942 / 0.9741 | 0.9579 | 0.4960 | 0.8999 |
| Q4_K_M | 438 | 0.98280 / 0.9278 | 0.97214 / 0.9495 | 0.94611 / 0.9339 | 0.9588 | 0.4890 | 0.8404 |

Ours vs ggml-org's Q8_0 on all texts: cosine mean 1.0000000, min
0.9999999. Long inputs lose the most: Q4_K_M 0.946 on them against 0.983
on sentences.

Tatoeba bitext is easy for bge-m3 (95.8 %): its top-1 barely moves even
at Q4_K_M, while the rankings do (top-10 overlap 84 %). The cosine and
the overlap are the numbers that separate the builds. Tokenizer:
llama.cpp's ids equal transformers' on 519 texts (every 25th and every
long one).

## 2. Things found on the way

- **Trailing whitespace.** BAAI's `tokenizer.json` turns trailing
  whitespace into an extra token (id 6, `▁`) before `</s>`;
  mlx-community's copy (the one LLMTray pins) has a WhitespaceSplit
  pre-tokenizer and doesn't. On the Tatoeba lines (ending in `\n`) 10,047
  of 12,045 texts tokenized differently; stripped, 0. The eval set strips
  every text; the card tells users to strip.
- **Inputs over 8192 tokens.** transformers truncates; llama-server
  answers HTTP 500. One of the 40 long modules hit 8192 and killed the
  first evaluation runs (they kept nothing: eval_gguf.py now saves its
  progress every 20 batches and leaves such rows out).
- **Two builds, one file name.** ours and ggml-org's Q8_0 are both
  `bge-m3-q8_0.gguf`: their caches collided until the name carried the
  folder.
- **Load on a shared node.** Six llama-servers and three MLX processes
  at once took k2 to 1 GB free: MLX's buffer cache grew to 6 GB a process
  on the CPU. eval_mlx.py now caps it (`--cache-gb`, default 1).
- llama.cpp `b12751` has no `llama-embedding` binary: embeddings through
  `llama-server`.

## 3. MLX (for LLMTray), published as roman220220/bge-m3-mlx-{fp16,8bit,6bit}

LLMTray's runtime (`runtime/llmtray_embed`, xlm-roberta family) loads
HF-layout `model.safetensors` with one `{"group_size", "bits"}`
quantization in `config.json` and quantizes the modules that have
`.scales`. mlx_build.py: the linear layers and the word embeddings at N
bits, group 64; position / token-type embeddings and norms fp16;
LLMTray's pinned `tokenizer.json` (sha256 `5df1f55d…`, mlx-community's,
no trailing-whitespace token).

Measured through LLMTray's own `Embedder` (eval_mlx.py: its loader,
tokenizer, batching, CLS pooling, normalization), fp16 compute, on the
Mac's GPU (Apple M5): 2-2.5 min a build for all 12,039 texts. Results in
`results/results-mlx.json`.

| build | MB | sentences | code | long | bitext top-1 | code top-1 | top-10 overlap |
|---|---|---|---|---|---|---|---|
| fp16 | 1133 | 1.00000 / 1.0000 | 1.00000 / 0.9995 | 0.99998 / 0.9999 | 0.9584 | 0.4970 | 0.9983 |
| 8-bit | 610 | 0.99986 / 0.9993 | 0.99975 / 0.9990 | 0.99921 / 0.9983 | 0.9585 | 0.4980 | 0.9845 |
| 6-bit | 471 | 0.99792 / 0.9691 | 0.99642 / 0.9933 | 0.99257 / 0.9894 | 0.9584 | 0.5000 | 0.9443 |

MLX's affine 8-bit (a scale and a bias per 64 values) is closer to the
original than GGUF Q8_0 (a scale per 32): 98.5 % against 97.1 % top-10
overlap, 0.9992 against 0.9983 on long texts, 26 MB smaller.

On k2's CPU, mlx[cpu] ran single-threaded: a full run of one build did
not finish in 3.5 hours (and kured's kernel-update reboot of k2 at 01:26
ended them all). The Mac did all three in 7 minutes.
