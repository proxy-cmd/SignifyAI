import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.hand_tracking import DetectionResult
from signifyai.rules import RuleBasedInterpreter


def _blank_hand() -> np.ndarray:
    h = np.zeros((21, 3), dtype=np.float32)
    h[:, 0] = 0.5
    h[:, 1] = 0.6
    return h


class RulesTests(unittest.TestCase):
    def test_open_palm_maps_to_hello(self):
        hand = _blank_hand()
        # Make index/middle/ring/pinky extended upward.
        for tip, pip, mcp in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]:
            hand[mcp, 1] = 0.60
            hand[pip, 1] = 0.45
            hand[tip, 1] = 0.30

        det = DetectionResult(
            features=np.zeros((126,), dtype=np.float32),
            hand_count=1,
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            raw_hands=[hand],
            handedness=["right"],
        )

        r = RuleBasedInterpreter().predict(det)
        self.assertIsNotNone(r)
        self.assertEqual(r.label, "HELLO")

    def test_two_open_palms_maps_to_thank_you(self):
        hand = _blank_hand()
        for tip, pip, mcp in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]:
            hand[mcp, 1] = 0.60
            hand[pip, 1] = 0.45
            hand[tip, 1] = 0.30

        det = DetectionResult(
            features=np.zeros((126,), dtype=np.float32),
            hand_count=2,
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            raw_hands=[hand.copy(), hand.copy()],
            handedness=["left", "right"],
        )

        r = RuleBasedInterpreter().predict(det)
        self.assertIsNotNone(r)
        self.assertEqual(r.label, "THANK YOU")


if __name__ == "__main__":
    unittest.main()
