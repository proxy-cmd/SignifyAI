import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.config import FEATURE_SIZE
from signifyai.dataset import load_dataset, save_records
from signifyai.modeling import derive_label_thresholds, train_model


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
        self.assertIn("hello", result.label_thresholds)
        self.assertIn("thanks", result.label_thresholds)
        self.assertTrue(hasattr(model, "predict"))

    def test_derive_label_thresholds(self):
        y_true = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=str)
        probs = np.asarray(
            [
                [0.91, 0.09],
                [0.84, 0.16],
                [0.72, 0.28],
                [0.18, 0.82],
                [0.12, 0.88],
                [0.35, 0.65],
            ],
            dtype=np.float32,
        )
        classes = np.asarray(["a", "b"], dtype=str)
        thresholds = derive_label_thresholds(y_true, probs, classes, default_threshold=0.6)
        self.assertEqual(sorted(thresholds.keys()), ["a", "b"])
        self.assertGreaterEqual(thresholds["a"], 0.35)
        self.assertLessEqual(thresholds["a"], 0.90)
        self.assertGreaterEqual(thresholds["b"], 0.35)
        self.assertLessEqual(thresholds["b"], 0.90)


if __name__ == "__main__":
    unittest.main()
