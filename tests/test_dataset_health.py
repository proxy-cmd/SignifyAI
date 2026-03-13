from pathlib import Path
import json

import numpy as np

from dataset.dataset_builder import check_dataset


def write_jsonl(path, rows):
    payload = "\n".join(json.dumps(r) for r in rows)
    path.write_text(payload, encoding="utf-8")


def make_clip(path, frames=6, dims=10):
    seq = np.zeros((frames, dims), dtype=np.float32)
    ts = np.arange(frames, dtype=np.float32)
    np.savez_compressed(path, sequence=seq, timestamps=ts)


def test_health_blocks_single_label_training(tmp_path):
    version = tmp_path / "v1"
    version.mkdir(parents=True)

    clip = tmp_path / "clip_0001.npz"
    make_clip(clip)
    row = {"intent_id": "hello", "signer_id": "s1", "npz_path": str(clip)}

    write_jsonl(version / "train.jsonl", [row, row])
    write_jsonl(version / "val.jsonl", [])
    write_jsonl(version / "test.jsonl", [])

    report = check_dataset(version)
    assert report["can_train"] is False
    assert any("fewer than 2 intent labels" in msg for msg in report["warnings"])


def test_health_allows_training_with_two_labels(tmp_path):
    version = tmp_path / "v2"
    version.mkdir(parents=True)

    clip_a = tmp_path / "a.npz"
    clip_b = tmp_path / "b.npz"
    clip_c = tmp_path / "c.npz"
    clip_d = tmp_path / "d.npz"
    make_clip(clip_a)
    make_clip(clip_b)
    make_clip(clip_c)
    make_clip(clip_d)
    row_a = {"intent_id": "hello", "signer_id": "s1", "npz_path": str(clip_a)}
    row_b = {"intent_id": "help", "signer_id": "s1", "npz_path": str(clip_b)}
    row_c = {"intent_id": "hello", "signer_id": "s1", "npz_path": str(clip_c)}
    row_d = {"intent_id": "help", "signer_id": "s1", "npz_path": str(clip_d)}

    write_jsonl(version / "train.jsonl", [row_a, row_b])
    write_jsonl(version / "val.jsonl", [row_c])
    write_jsonl(version / "test.jsonl", [row_d])

    report = check_dataset(version)
    assert report["can_train"] is True


def test_health_blocks_when_val_or_test_missing(tmp_path):
    version = tmp_path / "v3"
    version.mkdir(parents=True)

    clip_a = tmp_path / "a1.npz"
    clip_b = tmp_path / "b1.npz"
    make_clip(clip_a)
    make_clip(clip_b)
    row_a = {"intent_id": "hello", "signer_id": "s1", "npz_path": str(clip_a)}
    row_b = {"intent_id": "help", "signer_id": "s1", "npz_path": str(clip_b)}

    write_jsonl(version / "train.jsonl", [row_a, row_b])
    write_jsonl(version / "val.jsonl", [])
    write_jsonl(version / "test.jsonl", [])

    report = check_dataset(version)
    assert report["can_train"] is False
    assert any("Validation split is empty." in msg for msg in report["warnings"])
    assert any("Test split is empty." in msg for msg in report["warnings"])
