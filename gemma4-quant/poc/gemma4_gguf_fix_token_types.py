"""In-place fix of a Gemma 4 GGUF's tokenizer.ggml.token_type array so the
model's control tokens are tagged CONTROL (3) instead of NORMAL (1) /
USER_DEFINED (4).

Why: convert_hf_to_gguf.py tagged Gemma 4's control tokens (`<|turn>`,
`<turn|>`, `<|channel>`, `<channel|>`, `<|think|>`, the `<|tool*>` family,
image/audio/video markers) as NORMAL/USER_DEFINED -- it took the type from
the base vocab and ignored the `special: true` flag on the matching
added_tokens. At decode time NORMAL tokens are emitted as literal text, so
these leak into chat output ("<|channel>thought..."); CONTROL tokens are
suppressed. A correctly-tagged GGUF (stock q3km) stays clean.

This does a surgical BYTE patch: token_type is a contiguous INT32 array in
the GGUF metadata, so we locate each targeted element's absolute file
offset (via GGUFReader's memmap views) and overwrite just those 4-byte
ints with 3 (CONTROL). Tensors and all other metadata are untouched, so
this is instant and lossless -- no 12.5GB rewrite, no requantization.

Usage:
    python3 gemma4_gguf_fix_token_types.py --file model.gguf --special-ids 46,47,...
(--special-ids = every id whose added_tokens entry has special:true and
isn't already CONTROL; pass them explicitly so the fix is auditable.)
"""

import argparse

import gguf
import numpy as np

CONTROL = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="GGUF to patch in place")
    ap.add_argument("--special-ids", help="comma-separated token ids -> CONTROL")
    ap.add_argument("--from-tokenizer", help="tokenizer.json path: auto-derive ids from added_tokens with special=true (excludes <pad>/<eos>/<bos>/<unk>/<mask> which are already CONTROL)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.from_tokenizer:
        import json
        tj = json.load(open(args.from_tokenizer))
        skip = {"<pad>", "<eos>", "<bos>", "<unk>", "<mask>"}
        force_ids = sorted(
            t["id"] for t in tj.get("added_tokens", [])
            if t.get("special") and t["content"] not in skip
        )
        print(f"derived {len(force_ids)} special ids from {args.from_tokenizer}")
    elif args.special_ids:
        force_ids = sorted({int(x) for x in args.special_ids.split(",") if x.strip()})
    else:
        ap.error("pass --special-ids or --from-tokenizer")

    reader = gguf.GGUFReader(args.file)
    tt = reader.get_field("tokenizer.ggml.token_type")
    toks = reader.get_field("tokenizer.ggml.tokens")

    base_ptr = reader.data.ctypes.data  # memmap base address

    def elem_offset(field, i):
        view = field.parts[field.data[i]]
        return view.ctypes.data - base_ptr

    def tok_str(i):
        return bytes(toks.parts[toks.data[i]]).decode("utf-8", "replace")

    # Sanity: element stride should be 4 bytes (int32), contiguous.
    off0 = elem_offset(tt, 0)
    off1 = elem_offset(tt, 1)
    assert off1 - off0 == 4, f"token_type not int32-contiguous: stride={off1-off0}"

    plan = []
    for i in force_ids:
        cur = int(np.asarray(tt.parts[tt.data[i]]).item())
        plan.append((i, tok_str(i), cur, elem_offset(tt, i)))

    print("Planned token_type -> CONTROL(3):")
    for i, s, cur, off in plan:
        print(f"  id={i:6} {s!r:16} {cur} -> 3   @byte {off}")

    if args.dry_run:
        print("dry-run, no write")
        return

    with open(args.file, "r+b") as f:
        for i, s, cur, off in plan:
            f.seek(off)
            f.write(int(CONTROL).to_bytes(4, "little"))
    print("GEMMA4_GGUF_TOKENFIX_DONE (in-place)")


if __name__ == "__main__":
    main()
