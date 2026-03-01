import sys
from collections import deque
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.realtime import _frame_motion_score, _weighted_label_vote


class RealtimeVotingTests(unittest.TestCase):
    def test_weighted_vote_prefers_high_confidence_label(self):
        labels = deque(["HELLO", "YES", "YES", "HELLO"], maxlen=4)
        confs = deque([0.92, 0.45, 0.42, 0.90], maxlen=4)
        label, conf = _weighted_label_vote(labels, confs)
        self.assertEqual(label, "HELLO")
        self.assertGreater(conf, 0.80)

    def test_weighted_vote_handles_empty(self):
        label, conf = _weighted_label_vote(deque(maxlen=4), deque(maxlen=4))
        self.assertEqual(label, "NO_HAND")
        self.assertEqual(conf, 0.0)

    def test_frame_motion_score_detects_change(self):
        frame_a = np.zeros((32, 32, 3), dtype=np.uint8)
        frame_b = np.full((32, 32, 3), 255, dtype=np.uint8)
        s0, gray_a = _frame_motion_score(None, frame_a)
        s1, _ = _frame_motion_score(gray_a, frame_b)
        self.assertGreater(s0, 100.0)
        self.assertGreater(s1, 10.0)


if __name__ == "__main__":
    unittest.main()
