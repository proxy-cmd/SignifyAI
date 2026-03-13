from collections import Counter
from pathlib import Path
import json
import random

import numpy as np


class DataBuildCfg:
    def __init__(self, root=Path("data/landmarks"), out_root=Path("data/landmarks/versions"), train_ratio=0.7, val_ratio=0.15, seed=42):
        self.root = root
        self.out_root = out_root
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed


class DataBuilder:
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = DataBuildCfg()
        self.cfg = cfg

    def build(self, version):
        # build one dataset version folder: train/val/test + summary
        random.seed(self.cfg.seed)
        rows = self._load_rows()
        by_signer = self._by_signer(rows)
        split_mode = "signer"
        split_ids = {"train": set(), "val": set(), "test": set()}

        if len(by_signer) >= 3:
            split_ids = self._split_signers(list(by_signer.keys()))
            splits = self._assign_by_signer(by_signer, split_ids)
        else:
            split_mode = "label_fallback"
            splits = self._split_by_label(rows)

        out = self.cfg.out_root / version
        out.mkdir(parents=True, exist_ok=True)
        self._write_splits(out, splits)

        summary = {
            "version": version,
            "total_samples": len(rows),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "split_mode": split_mode,
            "train_signers": sorted(split_ids["train"]),
            "val_signers": sorted(split_ids["val"]),
            "test_signers": sorted(split_ids["test"]),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _load_rows(self):
        raw = self.cfg.root / "raw"
        if not raw.exists():
            return []
        rows = []
        for sess in raw.iterdir():
            if not sess.is_dir():
                continue
            idx = sess / "clips.jsonl"
            if not idx.exists():
                continue
            for line in idx.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _by_signer(rows):
        out = {}
        for row in rows:
            signer = str(row.get("signer_id", "anonymous"))
            out.setdefault(signer, []).append(row)
        return out

    def _split_signers(self, signer_ids):
        ids = signer_ids[:]
        random.shuffle(ids)
        total = len(ids)
        n_train = int(round(total * self.cfg.train_ratio))
        n_val = int(round(total * self.cfg.val_ratio))

        # keep at least one signer in each split when signer split is used
        if total >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)

            # reserve at least one signer for test
            if n_train + n_val >= total:
                n_val = max(1, total - n_train - 1)
            if n_train + n_val >= total:
                n_train = max(1, total - n_val - 1)

        return {
            "train": set(ids[:n_train]),
            "val": set(ids[n_train : n_train + n_val]),
            "test": set(ids[n_train + n_val :]),
        }

    @staticmethod
    def _assign_by_signer(by_signer, split_ids):
        out = {"train": [], "val": [], "test": []}
        for signer, rows in by_signer.items():
            if signer in split_ids["train"]:
                out["train"].extend(rows)
            elif signer in split_ids["val"]:
                out["val"].extend(rows)
            else:
                out["test"].extend(rows)
        return out

    @staticmethod
    def _write_splits(out_dir, splits):
        for split_name, rows in splits.items():
            payload = "\n".join(json.dumps(r) for r in rows)
            (out_dir / f"{split_name}.jsonl").write_text(payload, encoding="utf-8")

    def _split_by_label(self, rows):
        by_label = {}
        for row in rows:
            label = str(row.get("intent_id", "unknown"))
            by_label.setdefault(label, []).append(row)

        out = {"train": [], "val": [], "test": []}
        for label_rows in by_label.values():
            items = label_rows[:]
            random.shuffle(items)
            n = len(items)
            if n < 3:
                out["train"].extend(items)
                continue

            n_train = max(1, int(round(n * self.cfg.train_ratio)))
            n_val = max(1, int(round(n * self.cfg.val_ratio)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            n_test = n - n_train - n_val

            out["train"].extend(items[:n_train])
            out["val"].extend(items[n_train : n_train + n_val])
            out["test"].extend(items[n_train + n_val : n_train + n_val + n_test])
        return out


def _read_jsonl(path):
    # read json lines file into python dict list
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _norm_seq_len(seq, target):
    # make all clips same length for ML input
    steps = int(seq.shape[0])
    if steps == target:
        return seq.astype(np.float32)
    if steps <= 0:
        return np.zeros((target, seq.shape[1]), dtype=np.float32)
    if steps > target:
        idx = np.linspace(0, steps - 1, num=target).astype(np.int32)
        return seq[idx].astype(np.float32)
    pad = target - steps
    last = seq[-1:, :].repeat(pad, axis=0)
    return np.concatenate([seq, last], axis=0).astype(np.float32)


def load_split_xy(version_dir, split, seq_len=24):
    # load one split and return X, y arrays
    rows = _read_jsonl(version_dir / f"{split}.jsonl")
    x_list = []
    y_list = []
    for row in rows:
        npz = Path(str(row.get("npz_path", "")))
        if not npz.exists():
            continue
        payload = np.load(npz)
        seq = payload["sequence"].astype(np.float32)
        if seq.ndim != 2:
            continue
        norm = _norm_seq_len(seq, target=seq_len)
        x_list.append(norm.reshape(-1))
        y_list.append(str(row.get("intent_id", "unknown")))

    if not x_list:
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=object)
    return np.stack(x_list, axis=0), np.asarray(y_list, dtype=object)


def _clip_len_stats(rows):
    # clip-length stats shown in dataset health output
    lens = []
    for row in rows:
        npz = Path(str(row.get("npz_path", "")))
        if not npz.exists():
            continue
        try:
            payload = np.load(npz)
            seq = payload["sequence"]
            if seq.ndim == 2:
                lens.append(int(seq.shape[0]))
        except Exception:
            continue
    if not lens:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(lens, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def check_dataset(version_dir):
    # basic checks before training
    train = _read_jsonl(version_dir / "train.jsonl")
    val = _read_jsonl(version_dir / "val.jsonl")
    test = _read_jsonl(version_dir / "test.jsonl")
    all_rows = train + val + test

    labels_all = Counter()
    labels_train = Counter()
    labels_val = Counter()
    labels_test = Counter()
    signers = Counter()

    for row in all_rows:
        label = str(row.get("intent_id", "unknown"))
        signer = str(row.get("signer_id", "unknown"))
        labels_all[label] += 1
        signers[signer] += 1
    for row in train:
        label = str(row.get("intent_id", "unknown"))
        labels_train[label] += 1
    for row in val:
        label = str(row.get("intent_id", "unknown"))
        labels_val[label] += 1
    for row in test:
        label = str(row.get("intent_id", "unknown"))
        labels_test[label] += 1

    missing = 0
    for row in all_rows:
        if not Path(str(row.get("npz_path", ""))).exists():
            missing += 1

    warns = []
    if not version_dir.exists():
        warns.append("Dataset version folder does not exist.")
    if len(train) < 2:
        warns.append("Training split has fewer than 2 clips.")
    if len(labels_train) < 2:
        warns.append("Training split has fewer than 2 intent labels.")
    if len(val) == 0:
        warns.append("Validation split is empty.")
    if len(test) == 0:
        warns.append("Test split is empty.")
    if missing > 0:
        warns.append(f"{missing} clip files are missing on disk.")
    if len(signers) < 2:
        warns.append("Only one signer detected. Add more signers for robust model quality.")

    has_train = len(train) >= 2
    has_labels = len(labels_train) >= 2
    has_val = len(val) > 0
    has_test = len(test) > 0
    has_all_files = missing == 0
    can_train = has_train and has_labels and has_val and has_test and has_all_files
    return {
        "version_dir": str(version_dir),
        "clips": {"train": len(train), "val": len(val), "test": len(test), "total": len(all_rows)},
        "labels": {"all": dict(labels_all), "train": dict(labels_train), "val": dict(labels_val), "test": dict(labels_test)},
        "signers": dict(signers),
        "missing_npz": int(missing),
        "clip_len": _clip_len_stats(all_rows),
        "can_train": bool(can_train),
        "warnings": warns,
    }
