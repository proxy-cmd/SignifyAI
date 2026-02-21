import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.config import FEATURE_SIZE
from signifyai.dataset import save_records
from signifyai.sequence_dataset import (
    append_sequence_records,
    build_sequence_dataset_from_frames,
    load_sequence_dataset,
)


class SequenceDatasetTests(unittest.TestCase):
    def test_build_from_frame_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_csv = root / "dataset.csv"
            seq_npz = root / "sequence.npz"

            records = []
            for _ in range(60):
                records.append((np.full((FEATURE_SIZE,), 0.2, dtype=np.float32), "hello"))
            for _ in range(60):
                records.append((np.full((FEATURE_SIZE,), 0.8, dtype=np.float32), "thanks"))
            save_records(records, frame_csv)

            total, saved = build_sequence_dataset_from_frames(frame_csv, seq_npz, seq_len=12, stride=3)
            self.assertGreaterEqual(total, saved)
            self.assertGreater(saved, 10)

            ds = load_sequence_dataset(seq_npz)
            self.assertEqual(ds.seq_len, 12)
            self.assertEqual(ds.x.shape[2], FEATURE_SIZE)

    def test_append_sequence_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = Path(tmp) / "seq.npz"
            seq_len = 8
            clip_a = np.full((seq_len, FEATURE_SIZE), 0.1, dtype=np.float32)
            clip_b = np.full((seq_len, FEATURE_SIZE), 0.9, dtype=np.float32)

            n1 = append_sequence_records([(clip_a, "a")], out_npz, seq_len=seq_len)
            n2 = append_sequence_records([(clip_b, "b")], out_npz, seq_len=seq_len)
            self.assertEqual(n1, 1)
            self.assertEqual(n2, 1)

            ds = load_sequence_dataset(out_npz)
            self.assertEqual(ds.x.shape[0], 2)
            self.assertEqual(sorted(np.unique(ds.y).tolist()), ["a", "b"])


if __name__ == "__main__":
    unittest.main()

