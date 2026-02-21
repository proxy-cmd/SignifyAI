from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_DATASET_PATH, DEFAULT_LABELS_PATH, DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH
from .dataset import load_dataset
from .modeling import save_model, train_model


@dataclass
class TrainConfig:
    dataset_csv: Path = DEFAULT_DATASET_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    labels_path: Path = DEFAULT_LABELS_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    calibrate_probs: bool = True


def run_training(cfg: TrainConfig) -> float:
    ds = load_dataset(cfg.dataset_csv)
    model, result = train_model(ds.x, ds.y, calibrate_probs=cfg.calibrate_probs)

    print(f"Training complete. Accuracy: {result.accuracy:.4f}")
    print(f"Macro F1: {result.f1_macro:.4f}")
    print("Classification report:")
    print(result.report)

    save_model(model, result.labels, cfg.model_path, cfg.labels_path)
    print(f"Saved model to: {cfg.model_path}")
    print(f"Saved labels to: {cfg.labels_path}")
    cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    class_counts = {label: int((ds.y == label).sum()) for label in result.labels}
    metadata = {
        "accuracy": result.accuracy,
        "f1_macro": result.f1_macro,
        "num_samples": int(ds.x.shape[0]),
        "num_features": int(ds.x.shape[1]),
        "class_counts": class_counts,
        "labels": result.labels,
        "calibrated_probabilities": cfg.calibrate_probs,
    }
    cfg.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to: {cfg.metadata_path}")

    return result.accuracy
