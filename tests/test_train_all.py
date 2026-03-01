import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.config import FEATURE_SIZE
from signifyai.dataset import save_records
from signifyai.train_all import TrainAllConfig, run_train_all


class TrainAllTests(unittest.TestCase):
    def test_run_train_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.csv"
            records = []
            for _ in range(110):
                records.append((np.full((FEATURE_SIZE,), 0.15, dtype=np.float32), "a"))
            for _ in range(110):
                records.append((np.full((FEATURE_SIZE,), 0.85, dtype=np.float32), "b"))
            save_records(records, dataset)

            summary = run_train_all(
                TrainAllConfig(
                    dataset_csv=dataset,
                    frame_model_path=root / "frame.joblib",
                    frame_labels_path=root / "frame_labels.json",
                    frame_metadata_path=root / "frame_meta.json",
                    deep_model_path=root / "deep.keras",
                    deep_labels_path=root / "deep_labels.json",
                    deep_metadata_path=root / "deep_meta.json",
                    deep_preprocess_path=root / "deep_preprocess.joblib",
                    sequence_dataset_npz=root / "seq.npz",
                    seq_len=10,
                    seq_stride=2,
                    temporal_model_path=root / "temporal.joblib",
                    temporal_labels_path=root / "temporal_labels.json",
                    temporal_metadata_path=root / "temporal_meta.json",
                    summary_path=root / "summary.json",
                    deep_epochs=30,
                    deep_batch_size=32,
                    deep_patience=6,
                )
            )

            self.assertTrue(summary.exists())


if __name__ == "__main__":
    unittest.main()
