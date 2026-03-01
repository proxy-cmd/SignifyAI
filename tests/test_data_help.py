import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.data_help import get_data_help_text


class DataHelpTests(unittest.TestCase):
    def test_data_help_contains_core_commands(self):
        text = get_data_help_text()
        self.assertIn("collect --label", text)
        self.assertIn("build-image-dataset", text)
        self.assertIn("build-video-dataset", text)
        self.assertIn("import-url", text)


if __name__ == "__main__":
    unittest.main()
