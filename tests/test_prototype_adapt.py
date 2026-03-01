import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.config import FEATURE_SIZE
from signifyai.prototype_adapt import (
    PrototypeDB,
    append_prototypes,
    load_prototype_db,
    predict_prototype,
)


class PrototypeAdaptTests(unittest.TestCase):
    def test_append_and_load_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "proto.npz"
            v1 = np.full((FEATURE_SIZE,), 0.2, dtype=np.float32)
            v2 = np.full((FEATURE_SIZE,), 0.8, dtype=np.float32)
            added = append_prototypes([v1, v2], ["hello", "thanks"], path=db_path)
            self.assertEqual(added, 2)

            db = load_prototype_db(db_path)
            self.assertEqual(db.vectors.shape, (2, FEATURE_SIZE))
            self.assertEqual(sorted(db.labels.tolist()), ["hello", "thanks"])

    def test_predict_prototype_with_margin(self):
        v1 = np.zeros((FEATURE_SIZE,), dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros((FEATURE_SIZE,), dtype=np.float32)
        v2[1] = 1.0
        vecs = np.stack([v1, v2], axis=0)
        labels = np.asarray(["a", "b"], dtype=str)
        norms = vecs.copy()
        db = PrototypeDB(vectors=vecs, labels=labels, norms=norms)

        q = np.zeros((FEATURE_SIZE,), dtype=np.float32)
        q[0] = 1.0
        out = predict_prototype(q, db, min_similarity=0.5, min_margin=0.1)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.label, "a")

        # Ambiguous query should fail high margin requirement.
        q2 = np.zeros((FEATURE_SIZE,), dtype=np.float32)
        q2[0] = 1.0
        q2[1] = 1.0
        out2 = predict_prototype(q2, db, min_similarity=0.5, min_margin=0.9)
        self.assertIsNone(out2)


if __name__ == "__main__":
    unittest.main()
