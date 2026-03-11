from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random

import numpy as np


@dataclass
class DatasetVersionConfig:
    root: Path = Path("data/landmarks")
    out_root: Path = Path("data/landmarks/versions")
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42


class DatasetVersionBuilder:
    def __init__(self, cfg: DatasetVersionConfig) -> None:
        self.cfg = cfg

    def build_dataset_version(self, version: str) -> dict:
        random.seed(self.cfg.seed)
        raw_root = self.cfg.root / "raw"
        samples: list[dict] = []
        if raw_root.exists():
            for sess in raw_root.iterdir():
                if not sess.is_dir():
                    continue
                clips = sess / "clips.jsonl"
                if not clips.exists():
                    continue
                for line in clips.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        samples.append(json.loads(line))

        by_signer: dict[str, list[dict]] = {}
        for s in samples:
            by_signer.setdefault(str(s.get("signer_id", "anonymous")), []).append(s)

        signers = list(by_signer.keys())
        random.shuffle(signers)
        n = len(signers)
        n_train = int(round(n * self.cfg.train_ratio))
        n_val = int(round(n * self.cfg.val_ratio))
        train_signers = set(signers[:n_train])
        val_signers = set(signers[n_train:n_train + n_val])
        test_signers = set(signers[n_train + n_val:])

        split = {"train": [], "val": [], "test": []}
        for signer, rows in by_signer.items():
            if signer in train_signers:
                split["train"].extend(rows)
            elif signer in val_signers:
                split["val"].extend(rows)
            else:
                split["test"].extend(rows)

        out_dir = self.cfg.out_root / version
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, rows in split.items():
            (out_dir / f"{key}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        summary = {
            "version": version,
            "total_samples": len(samples),
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
            "train_signers": sorted(train_signers),
            "val_signers": sorted(val_signers),
            "test_signers": sorted(test_signers),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def load_split_arrays(version_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = version_dir / f"{split}.jsonl"
    if not path.exists():
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=object)
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    xs: list[np.ndarray] = []
    ys: list[str] = []
    for row in rows:
        npz = Path(row["npz_path"])
        if not npz.exists():
            continue
        payload = np.load(npz)
        seq = payload["sequence"].astype(np.float32)
        xs.append(seq.reshape(-1))
        ys.append(str(row["intent_id"]))
    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=object)
    return np.stack(xs, axis=0), np.asarray(ys, dtype=object)
