from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_SEQUENCE_DATASET_PATH,
    DEFAULT_TEMPORAL_MODEL_PATH,
    DEFAULT_TEMPORAL_LABELS_PATH,
    DEFAULT_TEMPORAL_METADATA_PATH,
)
from .sequence_dataset import build_sequence_dataset_from_frames
from .temporal_model import TemporalTrainConfig, run_temporal_training
from .train import TrainConfig, run_training


@dataclass
class ProductionTrainConfig:
    frame_dataset_csv: Path = DEFAULT_DATASET_PATH
    frame_model_path: Path = DEFAULT_MODEL_PATH
    frame_labels_path: Path = DEFAULT_LABELS_PATH
    frame_metadata_path: Path = DEFAULT_METADATA_PATH
    sequence_dataset_npz: Path = DEFAULT_SEQUENCE_DATASET_PATH
    sequence_len: int = 24
    sequence_stride: int = 4
    temporal_model_path: Path = DEFAULT_TEMPORAL_MODEL_PATH
    temporal_labels_path: Path = DEFAULT_TEMPORAL_LABELS_PATH
    temporal_metadata_path: Path = DEFAULT_TEMPORAL_METADATA_PATH
    summary_path: Path = Path("models/production_train_summary.json")


def run_production_training(cfg: ProductionTrainConfig) -> Path:
    # 1) Train frame-level model with AutoML.
    frame_acc = run_training(
        TrainConfig(
            dataset_csv=cfg.frame_dataset_csv,
            model_path=cfg.frame_model_path,
            labels_path=cfg.frame_labels_path,
            metadata_path=cfg.frame_metadata_path,
            automl=True,
        )
    )

    # 2) Build sequence dataset from frame CSV.
    windows_total, windows_saved = build_sequence_dataset_from_frames(
        frame_csv=cfg.frame_dataset_csv,
        out_npz=cfg.sequence_dataset_npz,
        seq_len=cfg.sequence_len,
        stride=cfg.sequence_stride,
        per_label_limit=0,
    )

    # 3) Train temporal model.
    temporal_acc = run_temporal_training(
        TemporalTrainConfig(
            dataset_npz=cfg.sequence_dataset_npz,
            model_path=cfg.temporal_model_path,
            labels_path=cfg.temporal_labels_path,
            metadata_path=cfg.temporal_metadata_path,
        )
    )

    summary = {
        "frame_accuracy": frame_acc,
        "temporal_accuracy": temporal_acc,
        "sequence_windows_total": windows_total,
        "sequence_windows_saved": windows_saved,
        "frame_model_path": str(cfg.frame_model_path),
        "temporal_model_path": str(cfg.temporal_model_path),
    }
    cfg.summary_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return cfg.summary_path

