from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import FEATURE_SIZE, DEFAULT_SEQUENCE_DATASET_PATH
from .dataset import load_dataset


@dataclass
class SequenceDataset:
    x: np.ndarray  # [N, T, F]
    y: np.ndarray  # [N]
    seq_len: int


def build_sequence_dataset_from_frames(
    frame_csv: Path,
    out_npz: Path = DEFAULT_SEQUENCE_DATASET_PATH,
    seq_len: int = 24,
    stride: int = 4,
    per_label_limit: int = 0,
) -> tuple[int, int]:
    if seq_len <= 1:
        raise ValueError("seq_len must be > 1")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    ds = load_dataset(frame_csv)
    labels = sorted(np.unique(ds.y).tolist())

    windows: list[np.ndarray] = []
    targets: list[str] = []
    total_candidates = 0

    for label in labels:
        idx = np.where(ds.y == label)[0]
        if len(idx) < seq_len:
            continue
        frames = ds.x[idx]
        count_for_label = 0
        for s in range(0, len(frames) - seq_len + 1, stride):
            seq = frames[s : s + seq_len]
            if seq.shape != (seq_len, FEATURE_SIZE):
                continue
            total_candidates += 1
            windows.append(seq.astype(np.float32))
            targets.append(label)
            count_for_label += 1
            if per_label_limit > 0 and count_for_label >= per_label_limit:
                break

    if not windows:
        raise ValueError("No valid sequences could be built from frame dataset.")

    x = np.stack(windows, axis=0).astype(np.float32)
    y = np.asarray(targets, dtype=str)
    save_sequence_dataset(x, y, out_npz, seq_len=seq_len)
    return total_candidates, int(x.shape[0])


def save_sequence_dataset(x: np.ndarray, y: np.ndarray, out_npz: Path, seq_len: int) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must be 3D [N,T,F], got shape {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same sample count")
    if x.shape[2] != FEATURE_SIZE:
        raise ValueError(f"Feature size mismatch. Expected {FEATURE_SIZE}, got {x.shape[2]}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        x=x.astype(np.float32),
        y=y.astype(str),
        seq_len=np.asarray([int(seq_len)], dtype=np.int32),
    )


def append_sequence_records(records: list[tuple[np.ndarray, str]], out_npz: Path, seq_len: int) -> int:
    if not records:
        return 0
    clips = []
    labels = []
    for clip, label in records:
        if clip.shape != (seq_len, FEATURE_SIZE):
            raise ValueError(f"Clip shape mismatch: expected {(seq_len, FEATURE_SIZE)}, got {clip.shape}")
        clips.append(clip.astype(np.float32))
        labels.append(label)
    new_x = np.stack(clips, axis=0).astype(np.float32)
    new_y = np.asarray(labels, dtype=str)

    if out_npz.exists():
        old = load_sequence_dataset(out_npz)
        if old.seq_len != seq_len:
            raise ValueError(f"Existing seq_len={old.seq_len}, cannot append seq_len={seq_len}")
        x = np.concatenate([old.x, new_x], axis=0)
        y = np.concatenate([old.y, new_y], axis=0)
    else:
        x = new_x
        y = new_y

    save_sequence_dataset(x, y, out_npz, seq_len=seq_len)
    return int(new_x.shape[0])


def load_sequence_dataset(npz_path: Path) -> SequenceDataset:
    if not npz_path.exists():
        raise FileNotFoundError(f"Sequence dataset not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(str)
    seq_len = int(data["seq_len"][0]) if "seq_len" in data else int(x.shape[1])

    if x.ndim != 3:
        raise ValueError(f"Invalid x shape in npz: {x.shape}")
    if x.shape[2] != FEATURE_SIZE:
        raise ValueError(f"Feature size mismatch in npz. Expected {FEATURE_SIZE}, got {x.shape[2]}")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("Invalid y shape or sample count mismatch in npz.")

    return SequenceDataset(x=x, y=y, seq_len=seq_len)

