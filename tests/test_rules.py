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

    def test_thumbs_up_maps_to_yes(self):
        hand = _blank_hand()
        # Fold other fingers.
        for tip, pip, mcp in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]:
            hand[mcp, 1] = 0.60
            hand[pip, 1] = 0.62
            hand[tip, 1] = 0.66

        # Thumb up
        hand[2] = np.array([0.46, 0.58, 0.0], dtype=np.float32)  # thumb mcp
        hand[3] = np.array([0.43, 0.50, 0.0], dtype=np.float32)  # thumb ip
        hand[4] = np.array([0.40, 0.38, 0.0], dtype=np.float32)  # thumb tip

        det = DetectionResult(
            features=np.zeros((126,), dtype=np.float32),
            hand_count=1,
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            raw_hands=[hand],
            handedness=["right"],
        )
        r = RuleBasedInterpreter().predict(det)
        self.assertIsNotNone(r)
        self.assertEqual(r.label, "YES")

    def test_two_vs_peace_spread(self):
        base = _blank_hand()
        # index + middle raised only
        for tip, pip, mcp in [(8, 6, 5), (12, 10, 9)]:
            base[mcp, 1] = 0.60
            base[pip, 1] = 0.45
            base[tip, 1] = 0.30
        for tip, pip, mcp in [(16, 14, 13), (20, 18, 17)]:
            base[mcp, 1] = 0.60
            base[pip, 1] = 0.62
            base[tip, 1] = 0.66

        # Close fingers -> TWO
        hand_two = base.copy()
        hand_two[8, 0] = 0.49
        hand_two[12, 0] = 0.52

        det_two = DetectionResult(
            features=np.zeros((126,), dtype=np.float32),
            hand_count=1,
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            raw_hands=[hand_two],
            handedness=["right"],
        )
        r_two = RuleBasedInterpreter().predict(det_two)
        self.assertIsNotNone(r_two)
        self.assertEqual(r_two.label, "TWO")

        # Wide V -> PEACE
        hand_peace = base.copy()
        hand_peace[8, 0] = 0.42
        hand_peace[12, 0] = 0.58

        det_peace = DetectionResult(
            features=np.zeros((126,), dtype=np.float32),
            hand_count=1,
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            raw_hands=[hand_peace],
            handedness=["right"],
        )
        r_peace = RuleBasedInterpreter().predict(det_peace)
        self.assertIsNotNone(r_peace)
        self.assertEqual(r_peace.label, "PEACE")


if __name__ == "__main__":
    unittest.main()
