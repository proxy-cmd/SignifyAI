import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.deep_infer import DeepRuntimeBundle, predict_deep
from signifyai.realtime import _fuse_frame_models


class _FakeScaler:
    def transform(self, x):
        return np.asarray(x, dtype=np.float32)


class _FakeModel:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=np.float32)

    def predict(self, x, verbose=0):
        return self._probs


class DeepRuntimeTests(unittest.TestCase):
    def test_predict_deep_returns_label_confidence_and_margin(self):
        bundle = DeepRuntimeBundle(
            model=_FakeModel([[0.1, 0.8, 0.1]]),
            scaler=_FakeScaler(),
            labels=["a", "b", "c"],
            metadata={},
        )
        label, conf, margin = predict_deep(bundle, np.ones((126,), dtype=np.float32))
        self.assertEqual(label, "b")
        self.assertGreaterEqual(conf, 0.79)
        self.assertGreater(margin, 0.65)

    def test_fuse_prefers_agreement(self):
        label, conf, src = _fuse_frame_models(
            ml_label="HELLO",
            ml_conf=0.78,
            ml_margin=0.11,
            ml_threshold=0.60,
            ml_min_margin=0.08,
            deep_label="HELLO",
            deep_conf=0.81,
            deep_margin=0.14,
            deep_threshold=0.62,
            deep_min_margin=0.06,
        )
        self.assertEqual(label, "HELLO")
        self.assertEqual(src, "ML+DEEP")
        self.assertGreaterEqual(conf, 0.81)

    def test_fuse_blocks_close_disagreement(self):
        label, _conf, src = _fuse_frame_models(
            ml_label="YES",
            ml_conf=0.74,
            ml_margin=0.12,
            ml_threshold=0.60,
            ml_min_margin=0.08,
            deep_label="NO",
            deep_conf=0.76,
            deep_margin=0.10,
            deep_threshold=0.62,
            deep_min_margin=0.06,
        )
        self.assertIsNone(label)
        self.assertEqual(src, "ML_DEEP_DISAGREE")


if __name__ == "__main__":
    unittest.main()
