import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.automl import save_automl_outputs, train_automl


class AutoMLTests(unittest.TestCase):
    def test_train_automl_and_save_outputs(self):
        rng = np.random.default_rng(42)
        n = 240
        x_a = rng.normal(0.2, 0.05, size=(n, 12)).astype(np.float32)
        x_b = rng.normal(0.8, 0.05, size=(n, 12)).astype(np.float32)
        x = np.vstack([x_a, x_b]).astype(np.float32)
        y = np.array(["a"] * n + ["b"] * n)

        model, result, cm = train_automl(x, y, use_augmentation=False)
        self.assertIn(result.best_name, {"rf_300", "rf_500", "et_400", "et_700", "logreg"})
        self.assertGreaterEqual(result.test_accuracy, 0.80)
        self.assertEqual(cm.shape, (2, 2))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.joblib"
            labels_path = root / "labels.json"
            metadata_path = root / "metadata.json"
            confusion_csv = root / "confusion.csv"
            save_automl_outputs(
                model=model,
                result=result,
                confusion=cm,
                model_path=model_path,
                labels_path=labels_path,
                metadata_path=metadata_path,
                confusion_csv=confusion_csv,
            )
            self.assertTrue(model_path.exists())
            self.assertTrue(labels_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertTrue(confusion_csv.exists())


if __name__ == "__main__":
    unittest.main()

