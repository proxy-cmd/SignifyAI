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
class TrainCfg:
    version_dir: Path
    model_name: str
    out_dir: Path = Path("data/models")


@dataclass
class EvalRes:
    accuracy: float
    report: str
    samples: int


class SeqModel:
    """Train and evaluate the baseline sequence classifier."""

    def train(self, cfg: TrainCfg) -> dict[str, Any]:
        x_train, y_train = load_split_arrays(cfg.version_dir, "train")
        x_val, y_val = load_split_arrays(cfg.version_dir, "val")

        self.validate_train_data(x_train, y_train)

        model = LogisticRegression(max_iter=2000, n_jobs=1)
        model.fit(x_train, y_train)

        val_accuracy = self.compute_accuracy(model, x_val, y_val)

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

    # Backward-compatible API
    def train_sequence_model(self, cfg: TrainCfg) -> dict[str, Any]:
        return self.train(cfg)

    def eval(self, version_dir: Path, model_name: str, out_dir: Path = Path("data/models")) -> EvalRes:
        model = joblib.load(out_dir / f"{model_name}.joblib")
        x_test, y_test = load_split_arrays(version_dir, "test")

        if x_test.shape[0] == 0:
            return EvalRes(accuracy=0.0, report="No test samples", samples=0)

        y_pred = model.predict(x_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report = str(classification_report(y_test, y_pred))
        return EvalRes(accuracy=accuracy, report=report, samples=int(x_test.shape[0]))

    # Backward-compatible API
    def evaluate(self, version_dir: Path, model_name: str, out_dir: Path = Path("data/models")) -> EvalRes:
        return self.eval(version_dir=version_dir, model_name=model_name, out_dir=out_dir)

    @staticmethod
    def validate_train_data(x_train: np.ndarray, y_train: np.ndarray) -> None:
        if x_train.shape[0] < 2:
            raise ValueError("Not enough training samples. Build dataset and record more clips.")

        unique_labels = set(y_train.tolist())
        if len(unique_labels) < 2:
            raise ValueError("Need at least 2 intent labels in training split.")

    @staticmethod
    def compute_accuracy(model: Any, x: np.ndarray, y: np.ndarray) -> float:
        if x.shape[0] == 0:
            return 0.0
        pred = model.predict(x)
        return float(accuracy_score(y, pred))

    # Backward-compatible names.
    _validate_training_data = validate_train_data
    _compute_accuracy = compute_accuracy


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


# Backward-compatible type names used by existing imports.
SequenceTrainConfig = TrainCfg
SequenceEvalResult = EvalRes
SequenceModelPipeline = SeqModel
