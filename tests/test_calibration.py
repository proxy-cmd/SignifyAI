import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.calibration import recommend_runtime_settings


class CalibrationTests(unittest.TestCase):
    def test_recommend_runtime_settings_low_fps(self):
        rec = recommend_runtime_settings(
            avg_fps=12.0,
            p20_brightness=42.0,
            p20_blur=40.0,
            p15_hand_area=0.010,
        )
        self.assertEqual(rec["infer_interval"], 3)
        self.assertLessEqual(float(rec["threshold"]), 0.58)
        self.assertGreaterEqual(float(rec["min_brightness"]), 30.0)

    def test_recommend_runtime_settings_high_fps(self):
        rec = recommend_runtime_settings(
            avg_fps=30.0,
            p20_brightness=70.0,
            p20_blur=120.0,
            p15_hand_area=0.030,
        )
        self.assertEqual(rec["infer_interval"], 1)
        self.assertGreaterEqual(float(rec["infer_scale"]), 0.75)
        self.assertGreaterEqual(float(rec["target_fps"]), 20.0)


if __name__ == "__main__":
    unittest.main()

