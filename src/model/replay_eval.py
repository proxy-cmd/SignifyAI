from pathlib import Path
import json

import numpy as np

from dataset.dataset_builder import _norm_seq_len, _read_jsonl
from model.sequence_model import load_model_for_runtime, predict_seq


def _read_seq_len(out_dir, model_name, fallback=24):
    meta_path = Path(out_dir) / f"{model_name}.json"
    if not meta_path.exists():
        return int(fallback)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta.get("seq_len", fallback))
    except Exception:
        return int(fallback)


def _eval_split(rows, seq_len, model_bundle, min_conf):
    total = 0
    correct = 0
    unknown = 0
    confs = []

    for row in rows:
        npz_path = Path(str(row.get("npz_path", "")))
        if not npz_path.exists():
            continue
        try:
            payload = np.load(npz_path)
            seq = np.asarray(payload["sequence"], dtype=np.float32)
        except Exception:
            continue
        if seq.ndim != 2:
            continue

        total += 1
        seq_norm = _norm_seq_len(seq, target=int(seq_len))
        pred, conf = predict_seq(model_bundle, seq_norm)
        if float(conf) < float(min_conf):
            pred = "unknown"
            unknown += 1
        confs.append(float(conf))

        truth = str(row.get("intent_id", "unknown"))
        if str(pred) == truth:
            correct += 1

    acc = (float(correct) / float(total)) if total > 0 else 0.0
    unknown_rate = (float(unknown) / float(total)) if total > 0 else 0.0
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return {
        "samples": int(total),
        "accuracy": acc,
        "unknown_rate": unknown_rate,
        "avg_conf": avg_conf,
    }


def run_replay_eval(version_dir, model_name="custom", out_dir=Path("data/models"), splits=None, min_conf=0.0):
    version_dir = Path(version_dir)
    out_dir = Path(out_dir)
    if splits is None:
        splits = ["val", "test"]

    model = load_model_for_runtime(model_name, out_dir=out_dir)
    if model is None:
        raise FileNotFoundError(f"Model not found: {out_dir / f'{model_name}.joblib'}")

    seq_len = _read_seq_len(out_dir=out_dir, model_name=model_name, fallback=24)
    out = {
        "version_dir": str(version_dir),
        "model_name": str(model_name),
        "seq_len": int(seq_len),
        "min_conf": float(min_conf),
        "splits": {},
    }
    for split in splits:
        rows = _read_jsonl(version_dir / f"{split}.jsonl")
        out["splits"][str(split)] = _eval_split(rows, seq_len=seq_len, model_bundle=model, min_conf=min_conf)
    return out
