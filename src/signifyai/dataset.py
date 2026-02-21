from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import FEATURE_SIZE


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray


def make_feature_columns() -> list[str]:
    return [f"f_{i:03d}" for i in range(FEATURE_SIZE)]


def save_records(records: Iterable[tuple[np.ndarray, str]], out_csv: Path) -> int:
    rows = []
    cols = make_feature_columns()

    for features, label in records:
        if features.shape != (FEATURE_SIZE,):
            raise ValueError(f"Feature shape mismatch: expected {(FEATURE_SIZE,)}, got {features.shape}")
        row = {col: float(features[i]) for i, col in enumerate(cols)}
        row["label"] = label
        rows.append(row)

    if not rows:
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        df = pd.concat([prev, df], ignore_index=True)

    df.to_csv(out_csv, index=False)
    return len(rows)


def load_dataset(csv_path: Path) -> Dataset:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = make_feature_columns()

    missing = [c for c in cols + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    x = df[cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=str)
    return Dataset(x=x, y=y)
