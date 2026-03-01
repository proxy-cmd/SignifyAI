import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.video_dataset import resolve_video_class_root


class VideoDatasetTests(unittest.TestCase):
    def test_resolve_video_class_root_detects_nested_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "archive" / "dataset"
            (nested / "hello").mkdir(parents=True, exist_ok=True)
            (nested / "thanks").mkdir(parents=True, exist_ok=True)
            (nested / "hello" / "a.mp4").write_bytes(b"dummy")
            (nested / "thanks" / "b.mov").write_bytes(b"dummy")

            detected = resolve_video_class_root(root)
            self.assertEqual(detected, nested.resolve())


if __name__ == "__main__":
    unittest.main()
