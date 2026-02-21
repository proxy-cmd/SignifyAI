import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.config import FEATURE_SIZE
from signifyai.sequence_dataset import save_sequence_dataset
from signifyai.temporal_model import TemporalTrainConfig, load_temporal_model, run_temporal_training


class TemporalModelTests(unittest.TestCase):
    def test_train_and_load_temporal_model(self):
        rng = np.random.default_rng(7)
        n = 120
        seq_len = 12
        x_a = rng.normal(0.2, 0.03, size=(n, seq_len, FEATURE_SIZE)).astype(np.float32)
        x_b = rng.normal(0.8, 0.03, size=(n, seq_len, FEATURE_SIZE)).astype(np.float32)
        x = np.concatenate([x_a, x_b], axis=0).astype(np.float32)
        y = np.asarray(["hello"] * n + ["thanks"] * n, dtype=str)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq_npz = root / "seq.npz"
            model_path = root / "temporal.joblib"
            labels_path = root / "temporal_labels.json"
            metadata_path = root / "temporal_meta.json"
            save_sequence_dataset(x, y, seq_npz, seq_len=seq_len)

            acc = run_temporal_training(
                TemporalTrainConfig(
                    dataset_npz=seq_npz,
                    model_path=model_path,
                    labels_path=labels_path,
                    metadata_path=metadata_path,
                )
            )
            self.assertGreaterEqual(acc, 0.90)

            model, labels, loaded_seq_len = load_temporal_model(model_path, labels_path, metadata_path)
            self.assertTrue(hasattr(model, "predict_proba"))
            self.assertEqual(sorted(labels), ["hello", "thanks"])
            self.assertEqual(loaded_seq_len, seq_len)


if __name__ == "__main__":
    unittest.main()

