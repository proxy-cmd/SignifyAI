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
    cols_all = cols + ["label"]
    df = pd.DataFrame(rows, columns=cols_all)
    write_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
    df.to_csv(out_csv, mode="a", index=False, header=write_header)
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
