import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.analytics import append_event
from signifyai.report import ReportConfig, build_session_report


class ReportTests(unittest.TestCase):
    def test_report_generates_markdown_with_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "session_log.csv"
            out_path = tmp_path / "session_report.md"

            append_event(log_path, label="HELLO", confidence=0.9, hand_count=1)
            append_event(log_path, label="HELLO", confidence=0.8, hand_count=1)
            append_event(log_path, label="THANK YOU", confidence=0.95, hand_count=2)

            report_path = build_session_report(ReportConfig(log_path=log_path, out_path=out_path))
            text = report_path.read_text(encoding="utf-8")

            self.assertIn("Total spoken events: **3**", text)
            self.assertIn("- HELLO: 2", text)
            self.assertIn("- THANK YOU: 1", text)


if __name__ == "__main__":
    unittest.main()

