import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.config import FEATURE_SIZE
from signifyai.dataset import save_records
from signifyai.dataset_check import run_dataset_check


class DatasetCheckTests(unittest.TestCase):
    def test_dataset_check_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.csv"
            records = []
            for _ in range(10):
                records.append((np.full((FEATURE_SIZE,), 0.2, dtype=np.float32), "a"))
            for _ in range(10):
                records.append((np.full((FEATURE_SIZE,), 0.8, dtype=np.float32), "b"))
            save_records(records, dataset)

            result = run_dataset_check(dataset, min_samples_per_label=5)
            self.assertTrue(result.ok)
            self.assertEqual(result.labels, 2)
            self.assertEqual(result.rows, 20)

    def test_dataset_check_low_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.csv"
            records = []
            for _ in range(5):
                records.append((np.full((FEATURE_SIZE,), 0.2, dtype=np.float32), "a"))
            for _ in range(1):
                records.append((np.full((FEATURE_SIZE,), 0.8, dtype=np.float32), "b"))
            save_records(records, dataset)

            result = run_dataset_check(dataset, min_samples_per_label=3)
            self.assertTrue(result.ok)
            self.assertIn("will be dropped", result.detail)


if __name__ == "__main__":
    unittest.main()
