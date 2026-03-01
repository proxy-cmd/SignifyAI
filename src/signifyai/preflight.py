from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import (
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEMPORAL_LABELS_PATH,
    DEFAULT_TEMPORAL_METADATA_PATH,
    DEFAULT_TEMPORAL_MODEL_PATH,
)
from .doctor import print_results, run_doctor


def _required_paths_for_mode(mode: str) -> list[Path]:
    mode = mode.lower().strip()
    if mode == "rules":
        return []
    if mode == "ml":
        return [DEFAULT_MODEL_PATH, DEFAULT_LABELS_PATH]
    if mode == "temporal":
        return [DEFAULT_TEMPORAL_MODEL_PATH, DEFAULT_TEMPORAL_LABELS_PATH, DEFAULT_TEMPORAL_METADATA_PATH]
    # hybrid
    return [
        DEFAULT_MODEL_PATH,
        DEFAULT_LABELS_PATH,
        DEFAULT_TEMPORAL_MODEL_PATH,
        DEFAULT_TEMPORAL_LABELS_PATH,
        DEFAULT_TEMPORAL_METADATA_PATH,
    ]


def _missing_paths(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if not p.exists()]


def run_preflight(mode: str = "hybrid", camera_index: int = 0, skip_camera: bool = False) -> int:
    """
    Preflight check before demo/deployment:
    - environment/mediapipe/camera/tts health via doctor
    - expected model files for selected mode
    """
    print("SignifyAI Preflight")
    print("=" * 60)
    code = print_results(run_doctor(camera_index=camera_index, check_camera=not skip_camera))

    required = _required_paths_for_mode(mode)
    missing = _missing_paths(required)
    if required:
        print("-" * 60)
        print(f"Mode: {mode}")
        if missing:
            print("[FAIL] Missing model artifacts:")
            for p in missing:
                print(f"  - {p}")
            code = 1
        else:
            print("[OK] Required model artifacts found.")
    else:
        print("-" * 60)
        print("Mode: rules (no model files required)")

    print("=" * 60)
    if code == 0:
        print("Preflight passed.")
    else:
        print("Preflight failed.")
    return code
