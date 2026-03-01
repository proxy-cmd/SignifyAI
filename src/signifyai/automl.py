from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import derive_label_thresholds


@dataclass
class AutoMLResult:
    best_name: str
    cv_f1_macro: float
    test_accuracy: float
    test_f1_macro: float
    labels: list[str]
    report: str
    label_thresholds: dict[str, float]


def _candidate_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "rf_300": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "rf_500": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "et_400": ExtraTreesClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "et_700": ExtraTreesClassifier(
            n_estimators=700,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def _augment_landmark_features(x: np.ndarray, y: np.ndarray, jitter_std: float = 0.007, copies: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if copies <= 0:
        return x, y
    aug_x = [x]
    aug_y = [y]
    for _ in range(copies):
        noise = np.random.normal(0.0, jitter_std, size=x.shape).astype(np.float32)
        aug_x.append((x + noise).astype(np.float32))
        aug_y.append(y.copy())
    return np.vstack(aug_x), np.concatenate(aug_y)


def train_automl(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    use_augmentation: bool = True,
) -> tuple[Any, AutoMLResult, np.ndarray]:
    labels = sorted(np.unique(y).tolist())
    if len(labels) < 2:
        raise ValueError("Need at least 2 labels for AutoML training")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    if use_augmentation and len(x_train) >= 200:
        x_train_aug, y_train_aug = _augment_landmark_features(x_train, y_train, copies=1)
    else:
        x_train_aug, y_train_aug = x_train, y_train

    models = _candidate_models(random_state=random_state)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)

    best_name = ""
    best_model = None
    best_cv = -1.0
    for name, model in models.items():
        scores = cross_val_score(model, x_train_aug, y_train_aug, cv=cv, scoring="f1_macro", n_jobs=-1)
        score = float(np.mean(scores))
        if score > best_cv:
            best_cv = score
            best_name = name
            best_model = model

    assert best_model is not None
    best_model.fit(x_train_aug, y_train_aug)
    preds = best_model.predict(x_test)
    probs_test = best_model.predict_proba(x_test)
    classes = np.asarray(getattr(best_model, "classes_", labels), dtype=str)
    test_acc = float(accuracy_score(y_test, preds))
    test_f1 = float(f1_score(y_test, preds, average="macro"))
    report = classification_report(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=labels)
    label_thresholds = derive_label_thresholds(
        y_true=y_test.astype(str),
        probs=np.asarray(probs_test, dtype=np.float32),
        classes=classes,
        default_threshold=0.60,
    )

    result = AutoMLResult(
        best_name=best_name,
        cv_f1_macro=best_cv,
        test_accuracy=test_acc,
        test_f1_macro=test_f1,
        labels=labels,
        report=report,
        label_thresholds=label_thresholds,
    )
    return best_model, result, cm


def save_automl_outputs(
    model: Any,
    result: AutoMLResult,
    confusion: np.ndarray,
    model_path: Path,
    labels_path: Path,
    metadata_path: Path,
    confusion_csv: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_csv.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    labels_path.write_text(json.dumps(result.labels, indent=2), encoding="utf-8")
    metadata = {
        "automl": True,
        "best_model": result.best_name,
        "cv_f1_macro": result.cv_f1_macro,
        "test_accuracy": result.test_accuracy,
        "test_f1_macro": result.test_f1_macro,
        "labels": result.labels,
        "label_thresholds": result.label_thresholds,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Save confusion matrix as CSV with header row/column.
    lines = []
    lines.append("," + ",".join(result.labels))
    for i, label in enumerate(result.labels):
        row = ",".join(str(int(v)) for v in confusion[i].tolist())
        lines.append(f"{label},{row}")
    confusion_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
