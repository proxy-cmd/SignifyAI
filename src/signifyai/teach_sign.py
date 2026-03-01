from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from .collect import CollectConfig, run_collection
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
    PATHS,
)
from .phrase_map import set_phrase
from .sequence_dataset import build_sequence_dataset_from_frames
from .temporal_model import TemporalTrainConfig, run_temporal_training
from .train import TrainConfig, run_training


@dataclass
class TeachSignConfig:
    label: str
    phrase_text: str | None = None
    samples: int = 180
    camera_index: int = 0
    width: int = 960
    height: int = 720
    dataset_csv: Path = DEFAULT_DATASET_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    labels_path: Path = DEFAULT_LABELS_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    min_samples_per_label: int = 5
    run_deep: bool = False
    deep_model_path: Path = DEFAULT_DEEP_MODEL_PATH
    deep_labels_path: Path = DEFAULT_DEEP_LABELS_PATH
    deep_metadata_path: Path = DEFAULT_DEEP_METADATA_PATH
    deep_preprocess_path: Path = DEFAULT_DEEP_PREPROCESS_PATH
    deep_epochs: int = 80
    deep_batch_size: int = 64
    deep_patience: int = 12
    run_temporal: bool = False
    sequence_dataset_npz: Path = DEFAULT_SEQUENCE_DATASET_PATH
    seq_len: int = 16
    seq_stride: int = 4
    temporal_model_path: Path = DEFAULT_TEMPORAL_MODEL_PATH
    temporal_labels_path: Path = DEFAULT_TEMPORAL_LABELS_PATH
    temporal_metadata_path: Path = DEFAULT_TEMPORAL_METADATA_PATH
    summary_path: Path = PATHS.data_processed / "teach_sign_summary.json"


def normalize_label(label: str) -> str:
    out = str(label).strip().lower().replace(" ", "_")
    if not out:
        raise ValueError("Label cannot be empty.")
    return out


def run_teach_sign(cfg: TeachSignConfig) -> Path:
    label = normalize_label(cfg.label)
    if cfg.phrase_text and cfg.phrase_text.strip():
        set_phrase(label, cfg.phrase_text.strip())

    saved = run_collection(
        CollectConfig(
            label=label,
            samples=int(cfg.samples),
            camera_index=cfg.camera_index,
            width=cfg.width,
            height=cfg.height,
            out_csv=cfg.dataset_csv,
            auto_mode=True,
            capture_interval_sec=0.30,
            min_hand_frames=2,
            min_feature_delta=0.010,
            flush_every=20,
        )
    )

    frame_accuracy = run_training(
        TrainConfig(
            dataset_csv=cfg.dataset_csv,
            model_path=cfg.model_path,
            labels_path=cfg.labels_path,
            metadata_path=cfg.metadata_path,
            automl=True,
            min_samples_per_label=cfg.min_samples_per_label,
        )
    )

    deep_summary: dict[str, float | str] | None = None
    if cfg.run_deep:
        try:
            from .deep_model import DeepTrainConfig, run_deep_training

            deep_res = run_deep_training(
                DeepTrainConfig(
                    dataset_csv=cfg.dataset_csv,
                    model_path=cfg.deep_model_path,
                    labels_path=cfg.deep_labels_path,
                    metadata_path=cfg.deep_metadata_path,
                    preprocess_path=cfg.deep_preprocess_path,
                    epochs=cfg.deep_epochs,
                    batch_size=cfg.deep_batch_size,
                    patience=cfg.deep_patience,
                    min_samples_per_label=max(2, cfg.min_samples_per_label),
                )
            )
            deep_summary = {
                "accuracy": float(deep_res.accuracy),
                "f1_macro": float(deep_res.f1_macro),
                "epochs_trained": int(deep_res.epochs_trained),
            }
        except Exception as ex:
            deep_summary = {"error": str(ex)}

    temporal_summary: dict[str, float | int] | None = None
    if cfg.run_temporal:
        windows_total, windows_saved = build_sequence_dataset_from_frames(
            frame_csv=cfg.dataset_csv,
            out_npz=cfg.sequence_dataset_npz,
            seq_len=max(4, int(cfg.seq_len)),
            stride=max(1, int(cfg.seq_stride)),
            per_label_limit=0,
        )
        temporal_acc = run_temporal_training(
            TemporalTrainConfig(
                dataset_npz=cfg.sequence_dataset_npz,
                model_path=cfg.temporal_model_path,
                labels_path=cfg.temporal_labels_path,
                metadata_path=cfg.temporal_metadata_path,
            )
        )
        temporal_summary = {
            "accuracy": float(temporal_acc),
            "windows_total": int(windows_total),
            "windows_saved": int(windows_saved),
        }

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "phrase_text": cfg.phrase_text or "",
        "samples_saved": int(saved),
        "frame_accuracy": float(frame_accuracy),
        "deep": deep_summary,
        "temporal": temporal_summary,
        "dataset_csv": str(cfg.dataset_csv),
        "model_path": str(cfg.model_path),
    }
    cfg.summary_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return cfg.summary_path

