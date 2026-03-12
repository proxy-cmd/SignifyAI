from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
from typing import Any

import numpy as np


@dataclass
class DsCfg:
    root: Path = Path("data/landmarks")
    out_root: Path = Path("data/landmarks/versions")
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42


class DsBuilder:
    """Build train/val/test jsonl files with signer-aware split."""

    def __init__(self, cfg: DsCfg) -> None:
        self.cfg = cfg

    def build(self, version: str) -> dict[str, Any]:
        random.seed(self.cfg.seed)

        rows = self.load_rows()
        by_signer = self.group_by_signer(rows)
        split_mode = "signer"
        split_signers: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

        if len(by_signer) >= 3:
            split_signers = self.split_signers(list(by_signer.keys()))
            split_rows = self.assign_split_rows(by_signer, split_signers)
        else:
            split_mode = "label_fallback"
            split_rows = self.split_rows_by_label(rows)

        out_dir = self.cfg.out_root / version
        out_dir.mkdir(parents=True, exist_ok=True)
        self.write_split_files(out_dir, split_rows)

        summary = {
            "version": version,
            "total_samples": len(rows),
            "train": len(split_rows["train"]),
            "val": len(split_rows["val"]),
            "test": len(split_rows["test"]),
            "split_mode": split_mode,
            "train_signers": sorted(split_signers["train"]),
            "val_signers": sorted(split_signers["val"]),
            "test_signers": sorted(split_signers["test"]),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # Backward-compatible API
    def build_dataset_version(self, version: str) -> dict[str, Any]:
        return self.build(version)

    def load_rows(self) -> list[dict[str, Any]]:
        raw_root = self.cfg.root / "raw"
        if not raw_root.exists():
            return []

        rows: list[dict[str, Any]] = []
        for session_dir in raw_root.iterdir():
            if not session_dir.is_dir():
                continue
            clip_index = session_dir / "clips.jsonl"
            if not clip_index.exists():
                continue
            lines = clip_index.read_text(encoding="utf-8").splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def group_by_signer(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            signer = str(row.get("signer_id", "anonymous"))
            out.setdefault(signer, []).append(row)
        return out

    def split_signers(self, signers: list[str]) -> dict[str, set[str]]:
        shuffled = signers[:]
        random.shuffle(shuffled)

        total = len(shuffled)
        train_count = int(round(total * self.cfg.train_ratio))
        val_count = int(round(total * self.cfg.val_ratio))

        train = set(shuffled[:train_count])
        val = set(shuffled[train_count : train_count + val_count])
        test = set(shuffled[train_count + val_count :])

        return {"train": train, "val": val, "test": test}

    @staticmethod
    def assign_split_rows(
        by_signer: dict[str, list[dict[str, Any]]],
        split_signers: dict[str, set[str]],
    ) -> dict[str, list[dict[str, Any]]]:
        split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for signer, rows in by_signer.items():
            if signer in split_signers["train"]:
                split_rows["train"].extend(rows)
                continue
            if signer in split_signers["val"]:
                split_rows["val"].extend(rows)
                continue
            split_rows["test"].extend(rows)
        return split_rows

    @staticmethod
    def write_split_files(out_dir: Path, split_rows: dict[str, list[dict[str, Any]]]) -> None:
        for split_name, rows in split_rows.items():
            path = out_dir / f"{split_name}.jsonl"
            payload = "\n".join(json.dumps(row) for row in rows)
            path.write_text(payload, encoding="utf-8")

    def split_rows_by_label(self, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            label = str(row.get("intent_id", "unknown"))
            by_label.setdefault(label, []).append(row)

        split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for _, label_rows in by_label.items():
            label_rows = label_rows[:]
            random.shuffle(label_rows)
            n = len(label_rows)
            if n < 3:
                split_rows["train"].extend(label_rows)
                continue

            n_train = int(round(n * self.cfg.train_ratio))
            n_val = int(round(n * self.cfg.val_ratio))
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            n_test = n - n_train - n_val

            split_rows["train"].extend(label_rows[:n_train])
            split_rows["val"].extend(label_rows[n_train : n_train + n_val])
            split_rows["test"].extend(label_rows[n_train + n_val : n_train + n_val + n_test])

        return split_rows

    # Backward-compatible method names.
    _load_rows = load_rows
    _group_by_signer = group_by_signer
    _split_signers = split_signers
    _split_rows = assign_split_rows
    _write_split_files = write_split_files
    _split_rows_by_label = split_rows_by_label


def normalize_sequence_length(sequence: np.ndarray, target_len: int) -> np.ndarray:
    steps = int(sequence.shape[0])
    if steps == target_len:
        return sequence.astype(np.float32)
    if steps <= 0:
        return np.zeros((target_len, sequence.shape[1]), dtype=np.float32)
    if steps > target_len:
        idx = np.linspace(0, steps - 1, num=target_len).astype(np.int32)
        return sequence[idx].astype(np.float32)

    # Pad with last frame when clip is shorter.
    pad_count = target_len - steps
    last = sequence[-1:, :].repeat(pad_count, axis=0)
    return np.concatenate([sequence, last], axis=0).astype(np.float32)


def load_split_arrays(version_dir: Path, split: str, target_len: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Load flattened sequence arrays (X) and labels (y) for a split."""
    split_path = version_dir / f"{split}.jsonl"
    if not split_path.exists():
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=object)

    rows = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    features: list[np.ndarray] = []
    labels: list[str] = []

    for row in rows:
        npz_path = Path(row["npz_path"])
        if not npz_path.exists():
            continue
        payload = np.load(npz_path)
        sequence = payload["sequence"].astype(np.float32)
        if sequence.ndim != 2:
            continue
        norm_seq = normalize_sequence_length(sequence, target_len=target_len)
        features.append(norm_seq.reshape(-1))
        labels.append(str(row["intent_id"]))

    if not features:
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=object)

    X = np.stack(features, axis=0)
    y = np.asarray(labels, dtype=object)
    return X, y


# Backward-compatible type names used by existing imports.
DatasetVersionConfig = DsCfg
DatasetVersionBuilder = DsBuilder
