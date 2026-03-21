from pathlib import Path
import json

import numpy as np

from model.sequence_model import SeqCfg, SeqTrainer


def _write_npz(path, rows):
    seq = np.asarray(rows, dtype=np.float32)
    ts = np.arange(seq.shape[0], dtype=np.int64)
    np.savez_compressed(path, sequence=seq, timestamps=ts)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_writes_lineage(tmp_path):
    version = tmp_path / "versions" / "v1"
    models = tmp_path / "models"
    version.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    a1 = tmp_path / "a1.npz"
    a2 = tmp_path / "a2.npz"
    b1 = tmp_path / "b1.npz"
    b2 = tmp_path / "b2.npz"
    _write_npz(a1, [[1.0, 1.0], [1.0, 1.0]])
    _write_npz(a2, [[1.1, 0.9], [1.0, 1.0]])
    _write_npz(b1, [[-1.0, -1.0], [-1.0, -1.0]])
    _write_npz(b2, [[-1.1, -0.9], [-1.0, -1.0]])

    train = [
        {"intent_id": "a", "signer_id": "s1", "npz_path": str(a1)},
        {"intent_id": "b", "signer_id": "s1", "npz_path": str(b1)},
    ]
    val = [{"intent_id": "a", "signer_id": "s1", "npz_path": str(a2)}]
    test = [{"intent_id": "b", "signer_id": "s1", "npz_path": str(b2)}]
    _write_jsonl(version / "train.jsonl", train)
    _write_jsonl(version / "val.jsonl", val)
    _write_jsonl(version / "test.jsonl", test)
    (version / "summary.json").write_text(json.dumps({"version": "v1"}), encoding="utf-8")

    trainer = SeqTrainer()
    out = trainer.train(SeqCfg(version_dir=version, model_name="toy", out_dir=models, seq_len=2, algo="logreg"))
    assert Path(out["meta_path"]).exists()

    meta = json.loads(Path(out["meta_path"]).read_text(encoding="utf-8"))
    lineage = meta.get("dataset_lineage", {})
    assert len(lineage.get("train_jsonl_sha1", "")) == 40
    assert len(lineage.get("val_jsonl_sha1", "")) == 40
    assert len(lineage.get("test_jsonl_sha1", "")) == 40
