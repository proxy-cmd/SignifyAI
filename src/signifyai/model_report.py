from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_DEEP_LABELS_PATH,
    DEFAULT_DEEP_METADATA_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_TEMPORAL_LABELS_PATH,
    DEFAULT_TEMPORAL_METADATA_PATH,
    PATHS,
)


@dataclass
class ModelReportConfig:
    out_path: Path = PATHS.data_processed / "model_report.md"
    frame_metadata: Path = DEFAULT_METADATA_PATH
    deep_metadata: Path = DEFAULT_DEEP_METADATA_PATH
    temporal_metadata: Path = DEFAULT_TEMPORAL_METADATA_PATH
    frame_labels: Path = DEFAULT_LABELS_PATH
    deep_labels: Path = DEFAULT_DEEP_LABELS_PATH
    temporal_labels: Path = DEFAULT_TEMPORAL_LABELS_PATH


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _load_labels_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return len(raw)
    except Exception:
        return 0
    return 0


def _fmt(val: Any, default: str = "n/a") -> str:
    if val is None:
        return default
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _section(title: str, metadata_path: Path, labels_path: Path, payload: dict[str, Any] | None) -> list[str]:
    lines = [f"## {title}", ""]
    if payload is None:
        lines.append(f"- status: MISSING (`{metadata_path}`)")
        lines.append(f"- labels_count: {_load_labels_count(labels_path)}")
        lines.append("")
        return lines

    lines.append(f"- status: OK (`{metadata_path}`)")
    lines.append(f"- labels_count: {_load_labels_count(labels_path)}")
    lines.append(f"- model_type: {_fmt(payload.get('model_type'))}")
    lines.append(f"- accuracy: {_fmt(payload.get('accuracy'))}")
    lines.append(f"- f1_macro: {_fmt(payload.get('f1_macro'))}")
    lines.append(f"- dataset_path: {_fmt(payload.get('dataset_path'))}")
    lines.append(f"- num_samples: {_fmt(payload.get('num_samples'))}")
    lines.append(f"- num_features: {_fmt(payload.get('num_features'))}")
    lines.append("")
    return lines


def build_model_report(cfg: ModelReportConfig) -> Path:
    frame = _load_json(cfg.frame_metadata)
    deep = _load_json(cfg.deep_metadata)
    temporal = _load_json(cfg.temporal_metadata)

    lines: list[str] = []
    lines.append("# SignifyAI Model Report")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.extend(_section("Frame Model (AutoML)", cfg.frame_metadata, cfg.frame_labels, frame))
    lines.extend(_section("Deep Model (TensorFlow)", cfg.deep_metadata, cfg.deep_labels, deep))
    lines.extend(_section("Temporal Model", cfg.temporal_metadata, cfg.temporal_labels, temporal))

    if frame is None and deep is None and temporal is None:
        lines.append("## Notes")
        lines.append("")
        lines.append("- No model metadata files found yet. Train models first.")
        lines.append("")

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.write_text("\n".join(lines), encoding="utf-8")
    return cfg.out_path
