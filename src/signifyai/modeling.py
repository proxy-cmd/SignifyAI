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
    label_thresholds: dict[str, float]


def derive_label_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: np.ndarray,
    default_threshold: float = 0.60,
) -> dict[str, float]:
    """Derive per-label confidence thresholds to reduce false positives in realtime."""
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probs, got shape {probs.shape}")
    if probs.shape[1] != len(classes):
        raise ValueError(
            f"Probability/class mismatch: probs has {probs.shape[1]} cols, classes has {len(classes)} labels"
        )
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D")
    if y_true.shape[0] != probs.shape[0]:
        raise ValueError("y_true/probs row count mismatch")

    grid = np.linspace(0.35, 0.90, num=23, dtype=np.float32)
    thresholds: dict[str, float] = {}

    for i, label in enumerate(classes.astype(str).tolist()):
        scores = probs[:, i].astype(np.float32)
        positives = (y_true == label)
        support = int(np.sum(positives))
        if support < 3:
            thresholds[label] = round(float(default_threshold), 4)
            continue

        best_f1 = -1.0
        best_t = float(default_threshold)
        for t in grid:
            pred_pos = scores >= float(t)
            tp = int(np.sum(pred_pos & positives))
            fp = int(np.sum(pred_pos & (~positives)))
            fn = int(np.sum((~pred_pos) & positives))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision < 0.55:
                continue
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = (2.0 * precision * recall) / (precision + recall)
            if f1 > best_f1 or (abs(f1 - best_f1) < 1e-9 and float(t) > best_t):
                best_f1 = f1
                best_t = float(t)

        if best_f1 < 0:
            positive_scores = scores[positives]
            if positive_scores.size > 0:
                q = float(np.quantile(positive_scores, 0.30))
                best_t = float(np.clip(q, 0.45, 0.90))
            else:
                best_t = float(default_threshold)

        thresholds[label] = round(best_t, 4)

    return thresholds


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
    probs = clf.predict_proba(x_test)
    classes = np.asarray(getattr(clf, "classes_", unique_labels), dtype=str)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="macro"))
    report = classification_report(y_test, preds, zero_division=0)
    label_thresholds = derive_label_thresholds(
        y_true=y_test.astype(str),
        probs=np.asarray(probs, dtype=np.float32),
        classes=classes,
        default_threshold=0.60,
    )

    return clf, TrainResult(
        accuracy=acc,
        f1_macro=f1,
        report=report,
        labels=unique_labels,
        label_thresholds=label_thresholds,
    )


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
