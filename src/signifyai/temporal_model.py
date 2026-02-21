from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_TEMPORAL_LABELS_PATH,
    DEFAULT_TEMPORAL_METADATA_PATH,
    DEFAULT_TEMPORAL_MODEL_PATH,
)
from .sequence_dataset import load_sequence_dataset


@dataclass
class TemporalTrainResult:
    accuracy: float
    f1_macro: float
    report: str
    labels: list[str]


@dataclass
class TemporalTrainConfig:
    dataset_npz: Path
    model_path: Path = DEFAULT_TEMPORAL_MODEL_PATH
    labels_path: Path = DEFAULT_TEMPORAL_LABELS_PATH
    metadata_path: Path = DEFAULT_TEMPORAL_METADATA_PATH


def _flatten_seq(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D sequence array [N,T,F], got {x.shape}")
    return x.reshape((x.shape[0], x.shape[1] * x.shape[2])).astype(np.float32)


def train_temporal_model(x_seq: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple[object, TemporalTrainResult]:
    labels = sorted(np.unique(y).tolist())
    if len(labels) < 2:
        raise ValueError("Need at least 2 labels for temporal training")
    x = _flatten_seq(x_seq)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = ExtraTreesClassifier(
        n_estimators=650,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="macro"))
    report = classification_report(y_test, preds, zero_division=0)
    result = TemporalTrainResult(accuracy=acc, f1_macro=f1, report=report, labels=labels)
    return model, result


def run_temporal_training(cfg: TemporalTrainConfig) -> float:
    ds = load_sequence_dataset(cfg.dataset_npz)
    model, result = train_temporal_model(ds.x, ds.y)

    print(f"Temporal training complete. Accuracy: {result.accuracy:.4f}")
    print(f"Temporal Macro F1: {result.f1_macro:.4f}")
    print("Temporal classification report:")
    print(result.report)

    save_temporal_model(model, result.labels, cfg.model_path, cfg.labels_path, cfg.metadata_path, ds.seq_len)
    print(f"Saved temporal model: {cfg.model_path}")
    print(f"Saved temporal labels: {cfg.labels_path}")
    print(f"Saved temporal metadata: {cfg.metadata_path}")
    return result.accuracy


def save_temporal_model(
    model: object,
    labels: list[str],
    model_path: Path,
    labels_path: Path,
    metadata_path: Path,
    seq_len: int,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    metadata = {
        "model_type": "temporal_extratrees",
        "labels": labels,
        "seq_len": int(seq_len),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_temporal_model(
    model_path: Path = DEFAULT_TEMPORAL_MODEL_PATH,
    labels_path: Path = DEFAULT_TEMPORAL_LABELS_PATH,
    metadata_path: Path = DEFAULT_TEMPORAL_METADATA_PATH,
) -> tuple[object, list[str], int]:
    if not model_path.exists():
        raise FileNotFoundError(f"Temporal model file not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Temporal labels file not found: {labels_path}")

    model = joblib.load(model_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    seq_len = 24
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            seq_len = int(meta.get("seq_len", 24))
        except Exception:
            seq_len = 24
    return model, labels, seq_len

