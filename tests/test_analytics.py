import sys
from pathlib import Path
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.analytics import append_event


class AnalyticsTests(unittest.TestCase):
    def test_append_event_creates_and_appends_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "session.csv"
            append_event(log_path, label="hello", confidence=0.88, hand_count=1)
            append_event(log_path, label="thanks", confidence=0.91, hand_count=2)

            df = pd.read_csv(log_path)
            self.assertEqual(len(df), 2)
            self.assertEqual(df.iloc[0]["label"], "hello")
            self.assertEqual(df.iloc[1]["label"], "thanks")

    def test_append_event_sanitizes_formula_like_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "session.csv"
            append_event(log_path, label="=HYPERLINK(\"x\")", confidence=0.8, hand_count=1)
            df = pd.read_csv(log_path)
            self.assertTrue(str(df.iloc[0]["label"]).startswith("'="))


if __name__ == "__main__":
    unittest.main()
