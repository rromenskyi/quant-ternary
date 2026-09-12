"""Patch a pip-installed mlx_lm's server.py to add --kv-bits/--kv-group-size/
--quantized-kv-start flags, threading them into the stream_generate(...) call
used to serve each request. Upstream mlx_lm.generate already supports KV-cache
quantization end to end (mlx_lm/generate.py's stream_generate accepts these
params directly) -- server.py just never exposed them as CLI flags.

Idempotent: safe to run against an already-patched server.py, or after a
fresh `pip install mlx-lm` that wiped out a previous patch (re-applies).

Usage:
    python poc/patch_mlx_server_kv.py [path-to-server.py]
    # with no argument, patches whatever mlx_lm is importable in the current env
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "--kv-bits"

IMPORT_OLD = """from ._version import __version__
from .generate import (
    BatchGenerator,
    SequenceStateMachine,
    stream_generate,
)"""
IMPORT_NEW = """from ._version import __version__
from .generate import (
    DEFAULT_QUANTIZED_KV_START,
    BatchGenerator,
    SequenceStateMachine,
    stream_generate,
)"""

CALL_OLD = """                prompt_progress_callback=progress,
                prefill_step_size=self.cli_args.prefill_step_size,
            ):"""
CALL_NEW = """                prompt_progress_callback=progress,
                prefill_step_size=self.cli_args.prefill_step_size,
                kv_bits=self.cli_args.kv_bits,
                kv_group_size=self.cli_args.kv_group_size,
                quantized_kv_start=self.cli_args.quantized_kv_start,
            ):"""

ARGS_OLD = """    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )"""
ARGS_NEW = """    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        help="Number of bits for KV cache quantization. Defaults to no quantization.",
        default=None,
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        help="Group size for KV cache quantization.",
        default=64,
    )
    parser.add_argument(
        "--quantized-kv-start",
        help="When --kv-bits is set, start quantizing the KV cache "
        "from this step onwards.",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
    )"""


def find_server_py() -> Path:
    import mlx_lm

    return Path(mlx_lm.__file__).parent / "server.py"


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else find_server_py()
    text = target.read_text()

    if MARKER in text:
        print(f"{target}: already patched, nothing to do")
        return

    for old, new, label in (
        (IMPORT_OLD, IMPORT_NEW, "import"),
        (CALL_OLD, CALL_NEW, "stream_generate call"),
        (ARGS_OLD, ARGS_NEW, "argparse flags"),
    ):
        if old not in text:
            raise SystemExit(
                f"{target}: expected snippet for '{label}' not found -- "
                f"mlx_lm's server.py has likely changed shape upstream, patch needs updating"
            )
        text = text.replace(old, new, 1)

    target.write_text(text)
    print(f"{target}: patched (added --kv-bits/--kv-group-size/--quantized-kv-start)")


if __name__ == "__main__":
    main()
