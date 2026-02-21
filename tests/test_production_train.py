import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.config import FEATURE_SIZE
from signifyai.dataset import save_records
from signifyai.production_train import ProductionTrainConfig, run_production_training


class ProductionTrainTests(unittest.TestCase):
    def test_run_production_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_csv = root / "dataset.csv"
            records = []
            for _ in range(90):
                records.append((np.full((FEATURE_SIZE,), 0.2, dtype=np.float32), "hello"))
            for _ in range(90):
                records.append((np.full((FEATURE_SIZE,), 0.8, dtype=np.float32), "thanks"))
            save_records(records, frame_csv)

            summary = run_production_training(
                ProductionTrainConfig(
                    frame_dataset_csv=frame_csv,
                    frame_model_path=root / "gesture_model.joblib",
                    frame_labels_path=root / "labels.json",
                    frame_metadata_path=root / "model_metadata.json",
                    sequence_dataset_npz=root / "sequence_dataset.npz",
                    sequence_len=10,
                    sequence_stride=2,
                    temporal_model_path=root / "temporal_model.joblib",
                    temporal_labels_path=root / "temporal_labels.json",
                    temporal_metadata_path=root / "temporal_metadata.json",
                    summary_path=root / "summary.json",
                )
            )
            self.assertTrue(summary.exists())


if __name__ == "__main__":
    unittest.main()

