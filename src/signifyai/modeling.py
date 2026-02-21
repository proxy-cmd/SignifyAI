from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


@dataclass
class TrainResult:
    accuracy: float
    report: str
    labels: list[str]


def train_model(x: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple[RandomForestClassifier, TrainResult]:
    unique_labels = sorted(np.unique(y).tolist())
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 labels for training")

    stratify = y if len(y) >= len(unique_labels) * 3 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_split=2,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, zero_division=0)

    return clf, TrainResult(accuracy=acc, report=report, labels=unique_labels)


def save_model(model, labels: list[str], model_path: Path, labels_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def load_model(model_path: Path, labels_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    model = joblib.load(model_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return model, labels
