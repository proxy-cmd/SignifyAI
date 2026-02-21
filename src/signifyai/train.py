from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_DATASET_PATH, DEFAULT_LABELS_PATH, DEFAULT_MODEL_PATH
from .dataset import load_dataset
from .modeling import save_model, train_model


@dataclass
class TrainConfig:
    dataset_csv: Path = DEFAULT_DATASET_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    labels_path: Path = DEFAULT_LABELS_PATH


def run_training(cfg: TrainConfig) -> float:
    ds = load_dataset(cfg.dataset_csv)
    model, result = train_model(ds.x, ds.y)

    print(f"Training complete. Accuracy: {result.accuracy:.4f}")
    print("Classification report:")
    print(result.report)

    save_model(model, result.labels, cfg.model_path, cfg.labels_path)
    print(f"Saved model to: {cfg.model_path}")
    print(f"Saved labels to: {cfg.labels_path}")

    return result.accuracy
