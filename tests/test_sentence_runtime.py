import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.sentence_runtime import can_auto_append_token, should_auto_speak_sentence


class SentenceRuntimeTests(unittest.TestCase):
    def test_can_auto_append_token_basic(self):
        self.assertTrue(
            can_auto_append_token(
                label="HELLO",
                stable_hits=4,
                min_stable_frames=3,
                now_ts=10.0,
                last_append_ts=8.0,
                append_cooldown_sec=1.0,
                last_token="YES",
            )
        )

    def test_can_auto_append_token_rejects_duplicate_and_unknown(self):
        self.assertFalse(
            can_auto_append_token(
                label="UNKNOWN",
                stable_hits=4,
                min_stable_frames=3,
                now_ts=10.0,
                last_append_ts=0.0,
                append_cooldown_sec=1.0,
                last_token=None,
            )
        )
        self.assertFalse(
            can_auto_append_token(
                label="HELLO",
                stable_hits=4,
                min_stable_frames=3,
                now_ts=10.0,
                last_append_ts=8.0,
                append_cooldown_sec=1.0,
                last_token="hello",
            )
        )

    def test_should_auto_speak_sentence(self):
        self.assertTrue(
            should_auto_speak_sentence(
                tokens_count=3,
                now_ts=10.0,
                last_token_ts=6.0,
                last_sentence_speak_ts=0.0,
                pause_sec=2.5,
            )
        )
        self.assertFalse(
            should_auto_speak_sentence(
                tokens_count=3,
                now_ts=7.0,
                last_token_ts=6.0,
                last_sentence_speak_ts=0.0,
                pause_sec=2.5,
            )
        )


if __name__ == "__main__":
    unittest.main()

