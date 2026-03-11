from __future__ import annotations

from signifyai.decoder.stability import StabilityConfig as StbCfg
from signifyai.decoder.stability import StabilityFilter as Stb
from signifyai.decoder.rules_intents import IntentHit


def test_stability_holds_label_after_window():
    f = Stb(StbCfg(window=3, min_confidence=0.5, hold_sec=0.0))
    for _ in range(3):
        lbl, conf, state = f.update(IntentHit("hospital_help", 0.8))
    assert lbl == "hospital_help"
    assert conf > 0.0
    assert state in {"stable", "pending"}
