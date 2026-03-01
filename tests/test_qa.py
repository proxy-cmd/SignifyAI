import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.qa import QAConfig, run_validate_all


class QATests(unittest.TestCase):
    def test_run_validate_all_minimal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "qa.json"
            report = run_validate_all(
                QAConfig(
                    out_json=out,
                    include_pytest=False,
                    include_cli_help_checks=False,
                    include_release_bundle_check=False,
                )
            )
            self.assertTrue(report.exists())
            text = report.read_text(encoding="utf-8")
            self.assertIn('"summary"', text)


if __name__ == "__main__":
    unittest.main()
