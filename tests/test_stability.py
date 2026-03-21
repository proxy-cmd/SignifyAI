from core.stability import Hit, StableCfg, StableFilter


def test_holds_label():
    f = StableFilter(StableCfg(win=3, min_conf=0.5, hold_sec=0.0))
    lbl = "unknown"
    conf = 0.0
    state = "pending"
    for _ in range(3):
        lbl, conf, state = f.update(Hit("hospital_help", 0.8))
    assert lbl == "hospital_help"
    assert conf > 0.0
    assert state in {"stable", "pending"}
