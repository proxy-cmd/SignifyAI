import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.tts import tts_rate_to_sapi_rate


class TTSUtilsTests(unittest.TestCase):
    def test_rate_mapping_clamped(self):
        self.assertEqual(tts_rate_to_sapi_rate(170), 0)
        self.assertLess(tts_rate_to_sapi_rate(120), 0)
        self.assertGreater(tts_rate_to_sapi_rate(220), 0)
        self.assertEqual(tts_rate_to_sapi_rate(999), 10)
        self.assertEqual(tts_rate_to_sapi_rate(1), -10)


if __name__ == "__main__":
    unittest.main()
