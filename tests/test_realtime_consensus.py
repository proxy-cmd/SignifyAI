import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signifyai.realtime import _strict_consensus_decision


class RealtimeConsensusTests(unittest.TestCase):
    def test_consensus_picks_agreeing_label(self):
        out = _strict_consensus_decision(
            rule_label="HELLO",
            rule_conf=0.85,
            proto_label=None,
            proto_conf=0.0,
            ml_label="HELLO",
            ml_conf=0.81,
            temporal_label="NO",
            temporal_conf=0.60,
            override_conf=0.92,
        )
        self.assertEqual(out[0], "HELLO")
        self.assertEqual(out[2], "CONSENSUS")

    def test_disagreement_without_override_returns_unknown(self):
        out = _strict_consensus_decision(
            rule_label="HELLO",
            rule_conf=0.80,
            proto_label=None,
            proto_conf=0.0,
            ml_label="YES",
            ml_conf=0.79,
            temporal_label="NO",
            temporal_conf=0.78,
            override_conf=0.92,
        )
        self.assertEqual(out[0], "UNKNOWN")
        self.assertEqual(out[2], "CONSENSUS")

    def test_high_conf_override_is_allowed(self):
        out = _strict_consensus_decision(
            rule_label="HELLO",
            rule_conf=0.96,
            proto_label=None,
            proto_conf=0.0,
            ml_label="YES",
            ml_conf=0.60,
            temporal_label="NO",
            temporal_conf=0.55,
            override_conf=0.92,
        )
        self.assertEqual(out[0], "HELLO")
        self.assertIn("OVERRIDE", out[2])


if __name__ == "__main__":
    unittest.main()
