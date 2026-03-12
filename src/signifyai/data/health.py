from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def clip_length_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    lengths: list[int] = []
    for row in rows:
        npz_path = Path(str(row.get("npz_path", "")))
        if not npz_path.exists():
            continue
        try:
            payload = np.load(npz_path)
            seq = payload["sequence"]
            if seq.ndim == 2:
                lengths.append(int(seq.shape[0]))
        except Exception:
            continue

    if not lengths:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    arr = np.asarray(lengths, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def analyze_dataset_version(version_dir: Path) -> dict[str, Any]:
    train_rows = load_jsonl(version_dir / "train.jsonl")
    val_rows = load_jsonl(version_dir / "val.jsonl")
    test_rows = load_jsonl(version_dir / "test.jsonl")
    all_rows = train_rows + val_rows + test_rows

    labels_all = Counter(str(r.get("intent_id", "unknown")) for r in all_rows)
    labels_train = Counter(str(r.get("intent_id", "unknown")) for r in train_rows)
    labels_val = Counter(str(r.get("intent_id", "unknown")) for r in val_rows)
    labels_test = Counter(str(r.get("intent_id", "unknown")) for r in test_rows)
    signers = Counter(str(r.get("signer_id", "unknown")) for r in all_rows)

    missing_npz = 0
    for row in all_rows:
        if not Path(str(row.get("npz_path", ""))).exists():
            missing_npz += 1

    warnings: list[str] = []
    if not version_dir.exists():
        warnings.append("Dataset version folder does not exist.")
    if len(train_rows) < 2:
        warnings.append("Training split has fewer than 2 clips.")
    if len(labels_train) < 2:
        warnings.append("Training split has fewer than 2 intent labels.")
    if len(val_rows) == 0:
        warnings.append("Validation split is empty.")
    if len(test_rows) == 0:
        warnings.append("Test split is empty.")
    if missing_npz > 0:
        warnings.append(f"{missing_npz} clip files are missing on disk.")
    if len(signers) < 2:
        warnings.append("Only one signer detected. Add more signers for robust model quality.")

    can_train = len(train_rows) >= 2 and len(labels_train) >= 2 and missing_npz == 0

    return {
        "version_dir": str(version_dir),
        "clips": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "total": len(all_rows),
        },
        "labels": {
            "all": dict(labels_all),
            "train": dict(labels_train),
            "val": dict(labels_val),
            "test": dict(labels_test),
        },
        "signers": dict(signers),
        "missing_npz": int(missing_npz),
        "clip_len": clip_length_stats(all_rows),
        "can_train": bool(can_train),
        "warnings": warnings,
    }
