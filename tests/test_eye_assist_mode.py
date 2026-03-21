from types import SimpleNamespace

from modes.eye_assist_mode import EyeAssistDecoder


def _state(ts_ms, ear=0.30, gaze_x=0.5, gaze_y=0.5, face=True):
    return SimpleNamespace(
        ts_ms=int(ts_ms),
        left_ear=float(ear),
        right_ear=float(ear),
        gaze_x=float(gaze_x),
        gaze_y=float(gaze_y),
        face_found=bool(face),
    )


def test_blink_emergency():
    dec = EyeAssistDecoder()

    assert dec.decode(_state(100, ear=0.30)) is None
    assert dec.decode(_state(150, ear=0.10)) is None
    hit = dec.decode(_state(980, ear=0.30))

    assert hit is not None
    assert hit.label == "emergency"


def test_single_blink_yes():
    dec = EyeAssistDecoder()

    assert dec.decode(_state(100, ear=0.30)) is None
    assert dec.decode(_state(150, ear=0.10)) is None
    assert dec.decode(_state(240, ear=0.30)) is None
    hit = dec.decode(_state(610, ear=0.30))

    assert hit is not None
    assert hit.label == "yes"


def test_triple_need_water():
    dec = EyeAssistDecoder()

    assert dec.decode(_state(100, ear=0.30)) is None
    assert dec.decode(_state(150, ear=0.10)) is None
    assert dec.decode(_state(240, ear=0.30)) is None
    assert dec.decode(_state(420, ear=0.10)) is None
    assert dec.decode(_state(520, ear=0.30)) is None
    assert dec.decode(_state(700, ear=0.10)) is None
    hit = dec.decode(_state(820, ear=0.30))

    assert hit is not None
    assert hit.label == "need_water"


def test_direction_holds():
    dec = EyeAssistDecoder()

    assert dec.decode(_state(100, gaze_x=0.2, gaze_y=0.5)) is None
    hit_left = dec.decode(_state(780, gaze_x=0.2, gaze_y=0.5))
    assert hit_left is not None and hit_left.label == "no"

    assert dec.decode(_state(1500, gaze_x=0.8, gaze_y=0.5)) is None
    hit_right = dec.decode(_state(2200, gaze_x=0.8, gaze_y=0.5))
    assert hit_right is not None and hit_right.label == "call_family"

    assert dec.decode(_state(3000, gaze_x=0.5, gaze_y=0.2)) is None
    hit_up = dec.decode(_state(3700, gaze_x=0.5, gaze_y=0.2))
    assert hit_up is not None and hit_up.label == "need_food"

    assert dec.decode(_state(4600, gaze_x=0.5, gaze_y=0.8)) is None
    hit_down = dec.decode(_state(5300, gaze_x=0.5, gaze_y=0.8))
    assert hit_down is None


def test_down_no_false_blink():
    dec = EyeAssistDecoder()

    assert dec.decode(_state(100, ear=0.205, gaze_y=0.85)) is None
    assert dec.decode(_state(400, ear=0.230, gaze_y=0.85)) is None
    # no blink edge should produce yes/emergency here
    out = dec.decode(_state(700, ear=0.230, gaze_y=0.85))
    assert out is None


def test_single_blink_yes_soft_close():
    dec = EyeAssistDecoder()
    assert dec.decode(_state(100, ear=0.30)) is None
    assert dec.decode(_state(150, ear=0.19)) is None
    assert dec.decode(_state(240, ear=0.30)) is None
    hit = dec.decode(_state(610, ear=0.30))
    assert hit is not None
    assert hit.label == "yes"
