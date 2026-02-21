"""Backward-compatible entrypoint for live sign-to-speech mode.

This file is intentionally tiny and stable.
It bootstraps `src/` into `sys.path` so both terminal and IDE runs work.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    """Ensure `<repo>/src` is importable when launching this file directly."""
    src_dir = Path(__file__).resolve().parent
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def main() -> None:
    _bootstrap_src_path()
    from signifyai.realtime import RealtimeConfig, run_realtime

    run_realtime(RealtimeConfig())


if __name__ == "__main__":
    main()
