import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from model.replay_eval import run_replay_eval


def _make_npz(path, frames):
    seq = np.asarray(frames, dtype=np.float32)
    ts = np.arange(seq.shape[0], dtype=np.int64)
    np.savez_compressed(path, sequence=seq, timestamps=ts)


def test_replay_eval_runs_on_saved_clips(tmp_path):
    version_dir = tmp_path / "versions" / "v1"
    model_dir = tmp_path / "models"
    raw_dir = tmp_path / "raw"
    version_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    a1 = raw_dir / "a1.npz"
    a2 = raw_dir / "a2.npz"
    b1 = raw_dir / "b1.npz"
    b2 = raw_dir / "b2.npz"
    _make_npz(a1, [[1.0, 1.0], [1.0, 1.0]])
    _make_npz(a2, [[0.9, 1.1], [1.0, 1.0]])
    _make_npz(b1, [[-1.0, -1.0], [-1.0, -1.0]])
    _make_npz(b2, [[-1.1, -0.9], [-1.0, -1.0]])

    rows = [
        {"intent_id": "a", "npz_path": str(a1)},
        {"intent_id": "a", "npz_path": str(a2)},
        {"intent_id": "b", "npz_path": str(b1)},
        {"intent_id": "b", "npz_path": str(b2)},
    ]
    (version_dir / "val.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (version_dir / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    x = np.asarray([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]], dtype=np.float32)
    y = np.asarray(["a", "b"], dtype=object)
    model = LogisticRegression().fit(x, y)
    bundle = {"name": "logreg", "model": model, "scale": False, "scaler": None}
    joblib.dump(bundle, model_dir / "toy.joblib")
    (model_dir / "toy.json").write_text(json.dumps({"seq_len": 2}, indent=2), encoding="utf-8")

    out = run_replay_eval(
        version_dir=version_dir,
        model_name="toy",
        out_dir=model_dir,
        splits=["val", "test"],
        min_conf=0.0,
    )
    assert out["model_name"] == "toy"
    assert out["seq_len"] == 2
    assert out["splits"]["val"]["samples"] == 4
    assert out["splits"]["test"]["samples"] == 4
    assert out["splits"]["val"]["accuracy"] >= 0.5
