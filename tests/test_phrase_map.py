import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.phrase_map import get_phrase, load_phrase_map, set_phrase


class PhraseMapTests(unittest.TestCase):
    def test_set_and_get_phrase(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "phrases.json"
            set_phrase("watching_you", "I am watching you.", path=p)
            self.assertEqual(get_phrase("watching_you", path=p), "I am watching you.")
            m = load_phrase_map(p)
            self.assertIn("watching_you", m)


if __name__ == "__main__":
    unittest.main()

