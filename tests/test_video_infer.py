import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.video_infer import compress_labels


class VideoInferTests(unittest.TestCase):
    def test_compress_labels(self):
        inp = ["NO_HAND", "HELLO", "HELLO", "UNKNOWN", "THANK YOU", "THANK YOU", "NO_HAND", "YES"]
        out = compress_labels(inp)
        self.assertEqual(out, ["HELLO", "THANK YOU", "YES"])


if __name__ == "__main__":
    unittest.main()

