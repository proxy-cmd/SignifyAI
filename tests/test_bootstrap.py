import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.bootstrap import _ensure_free_space


class BootstrapTests(unittest.TestCase):
    def test_ensure_free_space_raises_for_huge_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                _ensure_free_space(min_free_gb=1_000_000.0, path=Path(tmp))


if __name__ == "__main__":
    unittest.main()

