import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.config import FEATURE_SIZE
from signifyai.dataset import load_dataset, save_records
from signifyai.modeling import train_model


class DataAndModelTests(unittest.TestCase):
    def test_save_and_load_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "dataset.csv"
            records = []
            for _ in range(5):
                records.append((np.ones((FEATURE_SIZE,), dtype=np.float32), "hello"))
            for _ in range(5):
                records.append((np.zeros((FEATURE_SIZE,), dtype=np.float32), "thanks"))

            count = save_records(records, csv_path)
            self.assertEqual(count, 10)

            ds = load_dataset(csv_path)
            self.assertEqual(ds.x.shape, (10, FEATURE_SIZE))
            self.assertEqual(ds.y.shape, (10,))

    def test_train_model(self):
        x = []
        y = []

        for _ in range(20):
            x.append(np.ones((FEATURE_SIZE,), dtype=np.float32))
            y.append("hello")

        for _ in range(20):
            x.append(np.zeros((FEATURE_SIZE,), dtype=np.float32))
            y.append("thanks")

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=str)

        model, result = train_model(x, y)
        self.assertIn("hello", result.labels)
        self.assertIn("thanks", result.labels)
        self.assertGreaterEqual(result.accuracy, 0.80)
        self.assertGreaterEqual(result.f1_macro, 0.80)
        self.assertTrue(hasattr(model, "predict"))


if __name__ == "__main__":
    unittest.main()
