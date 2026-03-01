import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.preflight import _required_paths_for_mode


class PreflightTests(unittest.TestCase):
    def test_rules_mode_requires_no_model_files(self):
        self.assertEqual(_required_paths_for_mode("rules"), [])

    def test_hybrid_mode_requires_frame_and_temporal_artifacts(self):
        req = _required_paths_for_mode("hybrid")
        self.assertGreaterEqual(len(req), 5)


if __name__ == "__main__":
    unittest.main()
