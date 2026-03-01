import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.language import sentence_to_text, smooth_sentence


class LanguageTests(unittest.TestCase):
    def test_smooth_sentence_removes_adjacent_duplicates(self):
        toks = ["hello", "hello", "i", "i", "fine"]
        self.assertEqual(smooth_sentence(toks), ["hello", "I", "fine"])

    def test_sentence_to_text_formats_output(self):
        txt = sentence_to_text(["hello", "hello", "world"])
        self.assertEqual(txt, "Hello world.")

    def test_sentence_handles_common_single_sign_grammar(self):
        self.assertEqual(sentence_to_text(["help"]), "Please help.")
        self.assertEqual(sentence_to_text(["thank_you"]), "Thank you.")
        self.assertEqual(sentence_to_text(["i_love_you"]), "I love you.")

    def test_sentence_question_and_intro_formatting(self):
        txt = sentence_to_text(["hello", "i", "am", "proxy", "what", "is", "your", "name"])
        self.assertEqual(txt, "Hello, I am proxy what is your name?")


if __name__ == "__main__":
    unittest.main()
