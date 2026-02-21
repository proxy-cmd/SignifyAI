import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.doctor import run_doctor


class DoctorTests(unittest.TestCase):
    def test_doctor_runs_without_camera_check(self):
        results = run_doctor(check_camera=False)
        names = {r.name for r in results}
        self.assertIn("python", names)
        self.assertIn("opencv", names)
        self.assertIn("mediapipe", names)


if __name__ == "__main__":
    unittest.main()

