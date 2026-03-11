from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from signifyai.data.dataset_version import DsBuilder, DsCfg


def test_ds_builder_signer_split(tmp_path: Path):
    root = tmp_path / "landmarks"
    session_dir = root / "raw" / "s1"
    session_dir.mkdir(parents=True)
    npz = session_dir / "clip_0001.npz"
    np.savez_compressed(npz, sequence=np.zeros((4, 10), dtype=np.float32), timestamps=np.arange(4))
    clip_row = {
        "session_id": "s1",
        "clip_id": "clip_0001",
        "intent_id": "hospital_help",
        "signer_id": "signer_a",
        "consent_raw_video": False,
        "npz_path": str(npz),
        "frames": 4,
        "quality": {"brightness_avg": 60.0, "blur_avg": 80.0, "hand_area_max": 0.02},
    }
    (session_dir / "clips.jsonl").write_text(json.dumps(clip_row) + "\n", encoding="utf-8")

    out_root = root / "versions"
    ds = DsBuilder(DsCfg(root=root, out_root=out_root))
    summary = ds.build("vtest")
    assert summary["total_samples"] == 1
    assert (out_root / "vtest" / "summary.json").exists()
