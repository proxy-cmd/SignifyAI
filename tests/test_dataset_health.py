from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from signifyai.data.health import analyze_dataset_version


def write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "\n".join(json.dumps(r) for r in rows)
    path.write_text(payload, encoding="utf-8")


def make_clip(path: Path, frames: int = 6, dims: int = 10) -> None:
    seq = np.zeros((frames, dims), dtype=np.float32)
    ts = np.arange(frames, dtype=np.float32)
    np.savez_compressed(path, sequence=seq, timestamps=ts)


def test_health_blocks_single_label_training(tmp_path: Path):
    version = tmp_path / "v1"
    version.mkdir(parents=True)

    clip = tmp_path / "clip_0001.npz"
    make_clip(clip)
    row = {"intent_id": "hello", "signer_id": "s1", "npz_path": str(clip)}

    write_jsonl(version / "train.jsonl", [row, row])
    write_jsonl(version / "val.jsonl", [])
    write_jsonl(version / "test.jsonl", [])

    report = analyze_dataset_version(version)
    assert report["can_train"] is False
    assert any("fewer than 2 intent labels" in msg for msg in report["warnings"])


def test_health_allows_training_with_two_labels(tmp_path: Path):
    version = tmp_path / "v2"
    version.mkdir(parents=True)

    clip_a = tmp_path / "a.npz"
    clip_b = tmp_path / "b.npz"
    make_clip(clip_a)
    make_clip(clip_b)
    row_a = {"intent_id": "hello", "signer_id": "s1", "npz_path": str(clip_a)}
    row_b = {"intent_id": "help", "signer_id": "s1", "npz_path": str(clip_b)}

    write_jsonl(version / "train.jsonl", [row_a, row_b])
    write_jsonl(version / "val.jsonl", [])
    write_jsonl(version / "test.jsonl", [])

    report = analyze_dataset_version(version)
    assert report["can_train"] is True
