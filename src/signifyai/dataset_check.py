from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import FEATURE_SIZE


@dataclass
class DatasetCheckResult:
    ok: bool
    rows: int
    labels: int
    min_count: int
    max_count: int
    detail: str


def run_dataset_check(dataset_csv: Path, min_samples_per_label: int = 5) -> DatasetCheckResult:
    if not dataset_csv.exists():
        return DatasetCheckResult(
            ok=False,
            rows=0,
            labels=0,
            min_count=0,
            max_count=0,
            detail=f"Dataset file not found: {dataset_csv}",
        )

    df = pd.read_csv(dataset_csv)
    if "label" not in df.columns:
        return DatasetCheckResult(
            ok=False,
            rows=int(len(df)),
            labels=0,
            min_count=0,
            max_count=0,
            detail="Missing required column: label",
        )

    required_features = {f"f_{i:03d}" for i in range(FEATURE_SIZE)}
    present_features = {c for c in df.columns if c.startswith("f_")}
    missing_features = sorted(required_features - present_features)
    if missing_features:
        preview = ", ".join(missing_features[:5])
        suffix = " ..." if len(missing_features) > 5 else ""
        return DatasetCheckResult(
            ok=False,
            rows=int(len(df)),
            labels=int(df["label"].nunique()),
            min_count=0,
            max_count=0,
            detail=f"Missing feature columns ({len(missing_features)}): {preview}{suffix}",
        )

    label_counts = df["label"].value_counts()
    min_count = int(label_counts.min()) if len(label_counts) > 0 else 0
    max_count = int(label_counts.max()) if len(label_counts) > 0 else 0
    low_labels = label_counts[label_counts < int(min_samples_per_label)]

    if len(label_counts) < 2:
        return DatasetCheckResult(
            ok=False,
            rows=int(len(df)),
            labels=int(len(label_counts)),
            min_count=min_count,
            max_count=max_count,
            detail="Need at least 2 labels for training",
        )

    if len(low_labels) > 0:
        low_preview = ", ".join([f"{k}:{int(v)}" for k, v in low_labels.head(5).items()])
        suffix = " ..." if len(low_labels) > 5 else ""
        return DatasetCheckResult(
            ok=True,
            rows=int(len(df)),
            labels=int(len(label_counts)),
            min_count=min_count,
            max_count=max_count,
            detail=(
                f"Dataset usable. Labels under {min_samples_per_label} will be dropped during training: "
                f"{low_preview}{suffix}"
            ),
        )

    return DatasetCheckResult(
        ok=True,
        rows=int(len(df)),
        labels=int(len(label_counts)),
        min_count=min_count,
        max_count=max_count,
        detail="Dataset looks ready for training",
    )
