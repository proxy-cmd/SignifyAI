import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.model_report import ModelReportConfig, build_model_report


class ModelReportTests(unittest.TestCase):
    def test_build_model_report_with_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "model_report.md"
            report = build_model_report(
                ModelReportConfig(
                    out_path=out,
                    frame_metadata=root / "frame_meta.json",
                    deep_metadata=root / "deep_meta.json",
                    temporal_metadata=root / "temp_meta.json",
                    frame_labels=root / "frame_labels.json",
                    deep_labels=root / "deep_labels.json",
                    temporal_labels=root / "temp_labels.json",
                )
            )
            self.assertEqual(report, out)
            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8")
            self.assertIn("status: MISSING", text)


if __name__ == "__main__":
    unittest.main()
