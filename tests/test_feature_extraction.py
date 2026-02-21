import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.config import FEATURE_SIZE
from signifyai.feature_extraction import normalize_features


class FeatureExtractionTests(unittest.TestCase):
    def test_normalize_shape_is_preserved(self):
        x = np.random.rand(FEATURE_SIZE).astype(np.float32)
        y = normalize_features(x)
        self.assertEqual(y.shape, (FEATURE_SIZE,))

    def test_zero_vector_stays_zero(self):
        x = np.zeros((FEATURE_SIZE,), dtype=np.float32)
        y = normalize_features(x)
        self.assertTrue(np.allclose(y, 0.0))


if __name__ == "__main__":
    unittest.main()
