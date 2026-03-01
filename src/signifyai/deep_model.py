from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import tensorflow as tf
    _TF_IMPORT_ERROR: Exception | None = None
except Exception as ex:
    tf = None  # type: ignore[assignment]
    _TF_IMPORT_ERROR = ex

from .config import (
    DEFAULT_DEEP_LABELS_PATH,
    DEFAULT_DEEP_METADATA_PATH,
    DEFAULT_DEEP_MODEL_PATH,
    DEFAULT_DEEP_PREPROCESS_PATH,
)
from .dataset import load_dataset


def _require_tensorflow() -> Any:
    if tf is None:
        raise RuntimeError(
            f"TensorFlow is required for deep training but could not be imported: {_TF_IMPORT_ERROR}"
        )
    return tf


@dataclass
class DeepTrainConfig:
    dataset_csv: Path
    model_path: Path = DEFAULT_DEEP_MODEL_PATH
    labels_path: Path = DEFAULT_DEEP_LABELS_PATH
    metadata_path: Path = DEFAULT_DEEP_METADATA_PATH
    preprocess_path: Path = DEFAULT_DEEP_PREPROCESS_PATH
    epochs: int = 140
    batch_size: int = 64
    patience: int = 18
    min_samples_per_label: int = 6
    seed: int = 42


@dataclass
class DeepTrainResult:
    accuracy: float
    f1_macro: float
    report: str
    labels: list[str]
    dropped_labels: list[str]
    epochs_trained: int


def _prepare_dataset(
    x: np.ndarray,
    y: np.ndarray,
    min_samples_per_label: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    labels, counts = np.unique(y, return_counts=True)
    dropped: list[str] = []
    if min_samples_per_label > 1:
        keep = counts >= int(min_samples_per_label)
        dropped = labels[~keep].astype(str).tolist()
        keep_labels = set(labels[keep].astype(str).tolist())
        mask = np.asarray([lbl in keep_labels for lbl in y], dtype=bool)
        x = x[mask]
        y = y[mask]

    final_labels = sorted(np.unique(y).tolist())
    if len(final_labels) < 2:
        raise ValueError(
            f"Need at least 2 labels after filtering (min_samples_per_label={min_samples_per_label}). "
            f"Dropped: {', '.join(dropped) if dropped else '(none)'}"
        )
    return x, y, final_labels, dropped


def _build_mlp(num_features: int, num_classes: int, seed: int) -> Any:
    tf_mod = _require_tensorflow()
    tf_mod.keras.utils.set_random_seed(seed)
    model = tf_mod.keras.Sequential(
        [
            tf_mod.keras.layers.Input(shape=(num_features,)),
            tf_mod.keras.layers.BatchNormalization(),
            tf_mod.keras.layers.Dense(512, activation="relu"),
            tf_mod.keras.layers.Dropout(0.30),
            tf_mod.keras.layers.Dense(256, activation="relu"),
            tf_mod.keras.layers.Dropout(0.25),
            tf_mod.keras.layers.Dense(128, activation="relu"),
            tf_mod.keras.layers.Dropout(0.20),
            tf_mod.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf_mod.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_deep_training(cfg: DeepTrainConfig) -> DeepTrainResult:
    _require_tensorflow()
    ds = load_dataset(cfg.dataset_csv)
    x, y, labels, dropped = _prepare_dataset(ds.x, ds.y, cfg.min_samples_per_label)

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y_idx = np.asarray([label_to_idx[lbl] for lbl in y], dtype=np.int32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_idx,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=y_idx,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    model = _build_mlp(num_features=x_train_scaled.shape[1], num_classes=len(labels), seed=cfg.seed)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=max(6, int(cfg.patience)),
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, int(cfg.patience // 3)),
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        x_train_scaled,
        y_train,
        epochs=max(8, int(cfg.epochs)),
        batch_size=max(8, int(cfg.batch_size)),
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0,
    )
    epochs_trained = int(len(history.history.get("loss", [])))

    probs = model.predict(x_test_scaled, verbose=0)
    preds = np.argmax(probs, axis=1).astype(np.int32)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="macro"))

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    y_test_labels = np.asarray([idx_to_label[int(v)] for v in y_test], dtype=str)
    pred_labels = np.asarray([idx_to_label[int(v)] for v in preds], dtype=str)
    report = classification_report(y_test_labels, pred_labels, zero_division=0)

    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.labels_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.preprocess_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(cfg.model_path)
    cfg.labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    joblib.dump({"scaler": scaler, "labels": labels}, cfg.preprocess_path)

    meta = {
        "model_type": "tensorflow_mlp",
        "accuracy": acc,
        "f1_macro": f1,
        "epochs_trained": epochs_trained,
        "labels": labels,
        "dropped_labels": dropped,
        "dataset_path": str(cfg.dataset_csv),
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "batch_size": int(cfg.batch_size),
        "max_epochs": int(cfg.epochs),
        "patience": int(cfg.patience),
        "seed": int(cfg.seed),
        "preprocess_path": str(cfg.preprocess_path),
    }
    cfg.metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return DeepTrainResult(
        accuracy=acc,
        f1_macro=f1,
        report=report,
        labels=labels,
        dropped_labels=dropped,
        epochs_trained=epochs_trained,
    )
