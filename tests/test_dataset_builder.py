from pathlib import Path
import json

import numpy as np

from dataset.dataset_builder import DataBuildCfg, DataBuilder


def make_one_clip(session_dir):
    clip_file = session_dir / "clip_0001.npz"
    seq = np.zeros((4, 10), dtype=np.float32)
    ts = np.arange(4)
    np.savez_compressed(clip_file, sequence=seq, timestamps=ts)
    return clip_file


def write_clip_row(session_dir, clip_file):
    row = {
        "session_id": "s1",
        "clip_id": "clip_0001",
        "intent_id": "hospital_help",
        "signer_id": "signer_a",
        "consent_raw_video": False,
        "npz_path": str(clip_file),
        "frames": 4,
        "quality": {"brightness_avg": 60.0, "blur_avg": 80.0, "hand_area_max": 0.02},
    }
    lines = json.dumps(row) + "\n"
    (session_dir / "clips.jsonl").write_text(lines, encoding="utf-8")


def test_ds_builder_signer_split(tmp_path):
    root = tmp_path / "landmarks"
    session_dir = root / "raw" / "s1"
    session_dir.mkdir(parents=True)
    clip_file = make_one_clip(session_dir)
    write_clip_row(session_dir, clip_file)

    out_root = root / "versions"
    ds = DataBuilder(DataBuildCfg(root=root, out_root=out_root))
    summary = ds.build("vtest")
    assert summary["total_samples"] == 1
    assert (out_root / "vtest" / "summary.json").exists()
