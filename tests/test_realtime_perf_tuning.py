import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.realtime import _should_force_sync_detection, _tune_infer_interval


class RealtimePerfTuningTests(unittest.TestCase):
    def test_hand_visible_forces_interval_one(self):
        out = _tune_infer_interval(
            infer_every=4,
            fps=12.0,
            perf_target=30.0,
            hand_count=1,
            max_interval=4,
        )
        self.assertEqual(out, 1)

    def test_no_hand_can_increase_interval_on_low_fps(self):
        out = _tune_infer_interval(
            infer_every=1,
            fps=10.0,
            perf_target=30.0,
            hand_count=0,
            max_interval=4,
        )
        self.assertEqual(out, 2)

    def test_no_hand_can_reduce_interval_on_high_fps(self):
        out = _tune_infer_interval(
            infer_every=3,
            fps=40.0,
            perf_target=30.0,
            hand_count=0,
            max_interval=4,
        )
        self.assertEqual(out, 2)

    def test_force_sync_detection_when_async_result_is_stale_and_motion_is_high(self):
        out = _should_force_sync_detection(
            run_inference=True,
            last_detection=None,
            motion_score=8.0,
            static_diff_threshold=1.8,
            frames_since_fresh_result=2,
        )
        self.assertTrue(out)

    def test_no_force_sync_when_worker_is_still_fresh(self):
        out = _should_force_sync_detection(
            run_inference=True,
            last_detection=None,
            motion_score=8.0,
            static_diff_threshold=1.8,
            frames_since_fresh_result=1,
        )
        self.assertFalse(out)


if __name__ == "__main__":
    unittest.main()
