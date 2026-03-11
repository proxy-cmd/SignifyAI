from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from ..data.dataset_version import load_split_arrays


@dataclass
class SequenceTrainConfig:
    version_dir: Path
    model_name: str
    out_dir: Path = Path("data/models")


@dataclass
class SequenceEvalResult:
    accuracy: float
    report: str
    samples: int


class SequenceModelPipeline:
    """Train and evaluate the baseline sequence classifier."""

    def train_sequence_model(self, cfg: SequenceTrainConfig) -> dict[str, Any]:
        X_train, y_train = load_split_arrays(cfg.version_dir, "train")
        X_val, y_val = load_split_arrays(cfg.version_dir, "val")

        self._validate_training_data(X_train, y_train)

        model = LogisticRegression(max_iter=2000, n_jobs=1, multi_class="auto")
        model.fit(X_train, y_train)

        val_accuracy = self._compute_accuracy(model, X_val, y_val)

        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        model_path = cfg.out_dir / f"{cfg.model_name}.joblib"
        meta_path = cfg.out_dir / f"{cfg.model_name}.json"

        joblib.dump(model, model_path)
        metadata = {
            "model_name": cfg.model_name,
            "version_dir": str(cfg.version_dir),
            "val_accuracy": val_accuracy,
            "classes": [str(x) for x in getattr(model, "classes_", [])],
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "model_path": str(model_path),
            "meta_path": str(meta_path),
            "val_accuracy": val_accuracy,
        }

    def evaluate(self, version_dir: Path, model_name: str, out_dir: Path = Path("data/models")) -> SequenceEvalResult:
        model = joblib.load(out_dir / f"{model_name}.joblib")
        X_test, y_test = load_split_arrays(version_dir, "test")

        if X_test.shape[0] == 0:
            return SequenceEvalResult(accuracy=0.0, report="No test samples", samples=0)

        y_pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred)
        return SequenceEvalResult(accuracy=accuracy, report=report, samples=int(X_test.shape[0]))

    @staticmethod
    def _validate_training_data(X_train: np.ndarray, y_train: np.ndarray) -> None:
        if X_train.shape[0] < 2:
            raise ValueError("Not enough training samples. Build dataset and record more clips.")

        unique_labels = set(y_train.tolist())
        if len(unique_labels) < 2:
            raise ValueError("Need at least 2 intent labels in training split.")

    @staticmethod
    def _compute_accuracy(model: Any, X: np.ndarray, y: np.ndarray) -> float:
        if X.shape[0] == 0:
            return 0.0
        pred = model.predict(X)
        return float(accuracy_score(y, pred))


def load_runtime_model(model_name: str, out_dir: Path = Path("data/models")) -> Any | None:
    model_path = out_dir / f"{model_name}.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def predict_sequence_model(model: Any, sequence_matrix: np.ndarray) -> tuple[str, float]:
    if model is None:
        return "unknown", 0.0

    flat = sequence_matrix.reshape(1, -1)
    probabilities = model.predict_proba(flat)[0]
    best_index = int(np.argmax(probabilities))
    label = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    return label, confidence
