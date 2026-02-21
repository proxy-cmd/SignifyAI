from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    accuracy: float
    f1_macro: float
    report: str
    labels: list[str]


def _build_base_ensemble(random_state: int = 42):
    rf = RandomForestClassifier(
        n_estimators=450,
        max_depth=None,
        min_samples_split=2,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    et = ExtraTreesClassifier(
        n_estimators=450,
        max_depth=None,
        min_samples_split=2,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("lr", lr)],
        voting="soft",
        weights=[3, 3, 2],
        n_jobs=-1,
    )


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    calibrate_probs: bool = True,
) -> tuple[object, TrainResult]:
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

    clf = _build_base_ensemble(random_state=random_state)

    # Probability calibration tends to improve thresholding quality in realtime mode.
    if calibrate_probs and len(x_train) >= max(60, len(unique_labels) * 10):
        clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3)

    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="macro"))
    report = classification_report(y_test, preds, zero_division=0)

    return clf, TrainResult(accuracy=acc, f1_macro=f1, report=report, labels=unique_labels)


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
