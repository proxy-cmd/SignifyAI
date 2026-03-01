import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.image_dataset import resolve_class_root


class ImageDatasetTests(unittest.TestCase):
    def test_resolve_class_root_nested_archive_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "downloaded" / "ASL_Alphabet_Dataset" / "train"
            (nested / "hello").mkdir(parents=True, exist_ok=True)
            (nested / "thanks").mkdir(parents=True, exist_ok=True)
            (nested / "hello" / "a.jpg").write_bytes(b"fake")
            (nested / "thanks" / "b.jpg").write_bytes(b"fake")

            resolved = resolve_class_root(root)
            self.assertEqual(resolved, nested.resolve())


if __name__ == "__main__":
    unittest.main()
