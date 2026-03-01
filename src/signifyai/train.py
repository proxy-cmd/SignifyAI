from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .automl import save_automl_outputs, train_automl
from .config import (
    DEFAULT_CONFUSION_CSV_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
)
from .dataset import load_dataset
from .modeling import save_model, train_model


@dataclass
class TrainConfig:
    dataset_csv: Path = DEFAULT_DATASET_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    labels_path: Path = DEFAULT_LABELS_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    calibrate_probs: bool = True
    automl: bool = False
    confusion_csv_path: Path = DEFAULT_CONFUSION_CSV_PATH
    min_samples_per_label: int = 5


def run_training(cfg: TrainConfig) -> float:
    ds = load_dataset(cfg.dataset_csv)
    labels, counts = np.unique(ds.y, return_counts=True)
    dropped_labels: list[str] = []
    if cfg.min_samples_per_label > 1:
        keep = np.asarray(counts >= int(cfg.min_samples_per_label))
        dropped_labels = labels[~keep].astype(str).tolist()
        if np.any(~keep):
            keep_labels = set(labels[keep].astype(str).tolist())
            mask = np.asarray([label in keep_labels for label in ds.y], dtype=bool)
            ds = type(ds)(x=ds.x[mask], y=ds.y[mask])

    unique_after = sorted(np.unique(ds.y).tolist())
    if len(unique_after) < 2:
        details = f"Need at least 2 labels after filtering (min_samples_per_label={cfg.min_samples_per_label})."
        if dropped_labels:
            details += f" Dropped: {', '.join(dropped_labels)}."
        raise ValueError(details)

    if cfg.automl:
        model, result, confusion = train_automl(ds.x, ds.y)
        print(f"AutoML complete. Best model: {result.best_name}")
        print(f"CV Macro F1: {result.cv_f1_macro:.4f}")
        print(f"Test Accuracy: {result.test_accuracy:.4f}")
        print(f"Test Macro F1: {result.test_f1_macro:.4f}")
        print("Classification report:")
        print(result.report)

        save_automl_outputs(
            model=model,
            result=result,
            confusion=confusion,
            model_path=cfg.model_path,
            labels_path=cfg.labels_path,
            metadata_path=cfg.metadata_path,
            confusion_csv=cfg.confusion_csv_path,
        )
        try:
            meta = json.loads(cfg.metadata_path.read_text(encoding="utf-8"))
            meta["min_samples_per_label"] = int(cfg.min_samples_per_label)
            meta["dropped_labels"] = dropped_labels
            cfg.metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass
        print(f"Saved model to: {cfg.model_path}")
        print(f"Saved labels to: {cfg.labels_path}")
        print(f"Saved metadata to: {cfg.metadata_path}")
        print(f"Saved confusion matrix CSV to: {cfg.confusion_csv_path}")
        if dropped_labels:
            print(f"Dropped low-sample labels: {', '.join(dropped_labels)}")
        return result.test_accuracy

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
        "label_thresholds": result.label_thresholds,
        "min_samples_per_label": int(cfg.min_samples_per_label),
        "dropped_labels": dropped_labels,
    }
    cfg.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to: {cfg.metadata_path}")

    return result.accuracy
