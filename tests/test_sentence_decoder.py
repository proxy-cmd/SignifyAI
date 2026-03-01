import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.sentence_decoder import SentenceDecoder, SentenceDecoderConfig


class SentenceDecoderTests(unittest.TestCase):
    def test_rapid_stream_append_and_phrase_collapse(self):
        dec = SentenceDecoder(
            SentenceDecoderConfig(
                min_stable_frames=1,
                append_cooldown_sec=0.15,
                pause_speak_sec=1.0,
                max_tokens=10,
            )
        )
        t = 10.0
        dec.update(label="HELLO", stable_hits=1, hand_count=1, now_ts=t, auto_speak_enabled=False)
        dec.update(label="I", stable_hits=1, hand_count=1, now_ts=t + 0.2, auto_speak_enabled=False)
        dec.update(label="AM", stable_hits=1, hand_count=1, now_ts=t + 0.4, auto_speak_enabled=False)
        self.assertEqual(dec.tokens, ["HELLO", "I_AM"])

    def test_collapse_question_phrase(self):
        dec = SentenceDecoder(SentenceDecoderConfig(append_cooldown_sec=0.0))
        t = 1.0
        for i, tok in enumerate(["WHAT", "IS", "YOUR", "NAME"]):
            dec.update(
                label=tok,
                stable_hits=2,
                hand_count=1,
                now_ts=t + (i * 0.1),
                auto_speak_enabled=False,
            )
        self.assertEqual(dec.tokens, ["WHAT_IS_YOUR_NAME"])

    def test_auto_speaks_on_pause_and_clears_tokens(self):
        dec = SentenceDecoder(
            SentenceDecoderConfig(
                min_stable_frames=1,
                append_cooldown_sec=0.05,
                pause_speak_sec=0.5,
                no_hand_flush_frames=2,
            )
        )
        dec.update(label="HELLO", stable_hits=2, hand_count=1, now_ts=1.0, auto_speak_enabled=True)
        dec.update(label="YES", stable_hits=2, hand_count=1, now_ts=1.2, auto_speak_enabled=True)
        out = dec.update(label="NO_HAND", stable_hits=1, hand_count=0, now_ts=1.8, auto_speak_enabled=True)
        self.assertEqual(out.auto_spoken_text, "Hello yes.")
        self.assertEqual(dec.tokens, [])

    def test_manual_append_and_speak(self):
        dec = SentenceDecoder(SentenceDecoderConfig())
        self.assertTrue(dec.append_manual("THANK_YOU", now_ts=5.0))
        said = dec.speak_now(now_ts=6.0)
        self.assertEqual(said, "Thank you.")
        self.assertEqual(dec.tokens, [])


if __name__ == "__main__":
    unittest.main()
