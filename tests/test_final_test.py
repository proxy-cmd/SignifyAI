import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.final_test import _build_report_markdown


class FinalTestTests(unittest.TestCase):
    def test_build_report_markdown_contains_summary(self):
        payload = {
            "timestamp": "2026-03-01T20:00:00",
            "python": "3.12",
            "dataset_check": {"status": "PASS", "detail": "ok"},
            "summary": {"total": 10, "passed": 10, "failed": 0, "skipped": 0, "pass_rate": 100.0},
        }
        text = _build_report_markdown(payload)
        self.assertIn("SignifyAI Final Test Report", text)
        self.assertIn("PASS", text)
        self.assertIn("100.0%", text)


if __name__ == "__main__":
    unittest.main()

