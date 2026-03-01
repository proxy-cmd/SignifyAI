import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.safe_logging import csv_safe_text, redact_cli_args


class SafeLoggingTests(unittest.TestCase):
    def test_redact_cli_args_masks_secret_values(self):
        args = [
            "main.py",
            "--api-key",
            "abc123",
            "--token=xyz",
            "--mode",
            "hybrid",
            "OPENAI_API_KEY=secret-value",
        ]
        redacted = redact_cli_args(args)
        self.assertIn("***", redacted)
        self.assertIn("--token=***", redacted)
        self.assertIn("OPENAI_API_KEY=***", redacted)
        self.assertIn("hybrid", redacted)

    def test_csv_safe_text_blocks_formula_prefixes(self):
        self.assertEqual(csv_safe_text("=2+2"), "'=2+2")
        self.assertEqual(csv_safe_text("+cmd"), "'+cmd")
        self.assertEqual(csv_safe_text("-calc"), "'-calc")
        self.assertEqual(csv_safe_text("@sum"), "'@sum")
        self.assertEqual(csv_safe_text("hello"), "hello")


if __name__ == "__main__":
    unittest.main()

