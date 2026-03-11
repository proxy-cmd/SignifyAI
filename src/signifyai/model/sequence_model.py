from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

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
    def train_sequence_model(self, cfg: SequenceTrainConfig) -> dict:
        X_train, y_train = load_split_arrays(cfg.version_dir, "train")
        X_val, y_val = load_split_arrays(cfg.version_dir, "val")
        if X_train.shape[0] < 2:
            raise ValueError("Not enough training samples. Build dataset and record more clips.")
        if len(set(y_train.tolist())) < 2:
            raise ValueError("Need at least 2 intent labels in training split.")

        clf = LogisticRegression(max_iter=2000, n_jobs=1, multi_class="auto")
        clf.fit(X_train, y_train)

        val_acc = 0.0
        if X_val.shape[0] > 0:
            pred = clf.predict(X_val)
            val_acc = float(accuracy_score(y_val, pred))

        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        model_path = cfg.out_dir / f"{cfg.model_name}.joblib"
        meta_path = cfg.out_dir / f"{cfg.model_name}.json"
        joblib.dump(clf, model_path)
        meta = {
            "model_name": cfg.model_name,
            "version_dir": str(cfg.version_dir),
            "val_accuracy": val_acc,
            "classes": [str(x) for x in getattr(clf, "classes_", [])],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"model_path": str(model_path), "meta_path": str(meta_path), "val_accuracy": val_acc}

    def evaluate(self, version_dir: Path, model_name: str, out_dir: Path = Path("data/models")) -> SequenceEvalResult:
        model = joblib.load(out_dir / f"{model_name}.joblib")
        X_test, y_test = load_split_arrays(version_dir, "test")
        if X_test.shape[0] == 0:
            return SequenceEvalResult(accuracy=0.0, report="No test samples", samples=0)
        pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        report = classification_report(y_test, pred)
        return SequenceEvalResult(accuracy=acc, report=report, samples=int(X_test.shape[0]))


def load_runtime_model(model_name: str, out_dir: Path = Path("data/models")):
    path = out_dir / f"{model_name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def predict_sequence_model(model, seq_matrix: np.ndarray) -> tuple[str, float]:
    if model is None:
        return "unknown", 0.0
    vec = seq_matrix.reshape(1, -1)
    probs = model.predict_proba(vec)[0]
    idx = int(np.argmax(probs))
    lbl = str(model.classes_[idx])
    conf = float(probs[idx])
    return lbl, conf
