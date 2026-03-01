from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import (
    DEFAULT_DEEP_LABELS_PATH,
    DEFAULT_DEEP_METADATA_PATH,
    DEFAULT_DEEP_MODEL_PATH,
    DEFAULT_DEEP_PREPROCESS_PATH,
)


@dataclass
class DeepRuntimeBundle:
    model: Any
    scaler: Any
    labels: list[str]
    metadata: dict[str, Any]


def _load_tensorflow() -> Any:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as ex:
        raise RuntimeError(f"TensorFlow runtime unavailable: {ex}")
    return tf


def load_deep_runtime(
    model_path: Path = DEFAULT_DEEP_MODEL_PATH,
    labels_path: Path = DEFAULT_DEEP_LABELS_PATH,
    preprocess_path: Path = DEFAULT_DEEP_PREPROCESS_PATH,
    metadata_path: Path = DEFAULT_DEEP_METADATA_PATH,
) -> DeepRuntimeBundle:
    if not model_path.exists():
        raise FileNotFoundError(f"Deep model file not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Deep labels file not found: {labels_path}")
    if not preprocess_path.exists():
        raise FileNotFoundError(f"Deep preprocess file not found: {preprocess_path}")

    tf = _load_tensorflow()
    model = tf.keras.models.load_model(model_path)

    labels_raw = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels_raw, list) or not labels_raw:
        raise ValueError(f"Invalid deep labels file: {labels_path}")
    labels = [str(x) for x in labels_raw]

    prep = joblib.load(preprocess_path)
    scaler = prep.get("scaler") if isinstance(prep, dict) else None
    if scaler is None:
        raise ValueError(f"Missing scaler in preprocess file: {preprocess_path}")

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                metadata = raw
        except Exception:
            metadata = {}

    return DeepRuntimeBundle(model=model, scaler=scaler, labels=labels, metadata=metadata)


def predict_deep(bundle: DeepRuntimeBundle, features: np.ndarray) -> tuple[str | None, float, float]:
    x = np.asarray(features, dtype=np.float32).reshape(1, -1)
    x_scaled = bundle.scaler.transform(x).astype(np.float32)
    probs = np.asarray(bundle.model.predict(x_scaled, verbose=0), dtype=np.float32)
    if probs.ndim != 2 or probs.shape[0] != 1:
        return None, 0.0, 0.0

    p = probs[0]
    if p.size == 0:
        return None, 0.0, 0.0

    top = np.argsort(p)[::-1]
    i = int(top[0])
    j = int(top[1]) if len(top) > 1 else i
    if i < 0 or i >= len(bundle.labels):
        return None, 0.0, 0.0
    label = str(bundle.labels[i])
    conf = float(p[i])
    margin = float(p[i] - p[j]) if len(top) > 1 else 1.0
    return label, conf, margin
