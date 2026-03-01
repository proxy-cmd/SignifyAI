from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_DEEP_LABELS_PATH,
    DEFAULT_DEEP_METADATA_PATH,
    DEFAULT_DEEP_MODEL_PATH,
    DEFAULT_DEEP_PREPROCESS_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SEQUENCE_DATASET_PATH,
    DEFAULT_TEMPORAL_LABELS_PATH,
    DEFAULT_TEMPORAL_METADATA_PATH,
    DEFAULT_TEMPORAL_MODEL_PATH,
)
from .sequence_dataset import build_sequence_dataset_from_frames
from .temporal_model import TemporalTrainConfig, run_temporal_training
from .train import TrainConfig, run_training


@dataclass
class TrainAllConfig:
    dataset_csv: Path = DEFAULT_DATASET_PATH
    frame_model_path: Path = DEFAULT_MODEL_PATH
    frame_labels_path: Path = DEFAULT_LABELS_PATH
    frame_metadata_path: Path = DEFAULT_METADATA_PATH
    deep_model_path: Path = DEFAULT_DEEP_MODEL_PATH
    deep_labels_path: Path = DEFAULT_DEEP_LABELS_PATH
    deep_metadata_path: Path = DEFAULT_DEEP_METADATA_PATH
    deep_preprocess_path: Path = DEFAULT_DEEP_PREPROCESS_PATH
    sequence_dataset_npz: Path = DEFAULT_SEQUENCE_DATASET_PATH
    seq_len: int = 24
    seq_stride: int = 4
    temporal_model_path: Path = DEFAULT_TEMPORAL_MODEL_PATH
    temporal_labels_path: Path = DEFAULT_TEMPORAL_LABELS_PATH
    temporal_metadata_path: Path = DEFAULT_TEMPORAL_METADATA_PATH
    summary_path: Path = Path("models/train_all_summary.json")
    frame_min_samples_per_label: int = 5
    deep_min_samples_per_label: int = 6
    deep_epochs: int = 140
    deep_batch_size: int = 64
    deep_patience: int = 18


def run_train_all(cfg: TrainAllConfig) -> Path:
    from .deep_model import DeepTrainConfig, run_deep_training

    # 1) Train frame-level AutoML model.
    frame_acc = run_training(
        TrainConfig(
            dataset_csv=cfg.dataset_csv,
            model_path=cfg.frame_model_path,
            labels_path=cfg.frame_labels_path,
            metadata_path=cfg.frame_metadata_path,
            automl=True,
            min_samples_per_label=cfg.frame_min_samples_per_label,
        )
    )

    # 2) Train deep model.
    deep_result = run_deep_training(
        DeepTrainConfig(
            dataset_csv=cfg.dataset_csv,
            model_path=cfg.deep_model_path,
            labels_path=cfg.deep_labels_path,
            metadata_path=cfg.deep_metadata_path,
            preprocess_path=cfg.deep_preprocess_path,
            epochs=cfg.deep_epochs,
            batch_size=cfg.deep_batch_size,
            patience=cfg.deep_patience,
            min_samples_per_label=cfg.deep_min_samples_per_label,
        )
    )

    # 3) Build temporal dataset from frame-level dataset.
    windows_total, windows_saved = build_sequence_dataset_from_frames(
        frame_csv=cfg.dataset_csv,
        out_npz=cfg.sequence_dataset_npz,
        seq_len=cfg.seq_len,
        stride=cfg.seq_stride,
        per_label_limit=0,
    )

    # 4) Train temporal model.
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
        "deep_accuracy": deep_result.accuracy,
        "deep_f1_macro": deep_result.f1_macro,
        "deep_epochs_trained": deep_result.epochs_trained,
        "deep_dropped_labels": deep_result.dropped_labels,
        "temporal_accuracy": temporal_acc,
        "sequence_windows_total": windows_total,
        "sequence_windows_saved": windows_saved,
        "dataset_csv": str(cfg.dataset_csv),
        "frame_model_path": str(cfg.frame_model_path),
        "deep_model_path": str(cfg.deep_model_path),
        "temporal_model_path": str(cfg.temporal_model_path),
    }
    cfg.summary_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return cfg.summary_path
