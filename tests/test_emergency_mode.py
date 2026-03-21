from types import SimpleNamespace

import numpy as np

from modes.emergency_mode import AID_SIGNS, AidDecoder


TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


def _make_hand(open_fingers=None, thumb_open=False, thumb_pose="neutral"):
    open_fingers = set(open_fingers or [])
    hand = np.zeros((21, 3), dtype=np.float32)

    hand[0] = [0.50, 0.60, 0.0]  # wrist

    x_map = {
        "index": 0.44,
        "middle": 0.50,
        "ring": 0.56,
        "pinky": 0.62,
    }

    for name in ("index", "middle", "ring", "pinky"):
        x = x_map[name]
        hand[MCP[name]] = [x, 0.60, 0.0]
        if name in open_fingers:
            hand[PIP[name]] = [x, 0.50, 0.0]
            hand[TIP[name]] = [x, 0.40, 0.0]
        else:
            hand[PIP[name]] = [x, 0.62, 0.0]
            hand[TIP[name]] = [x, 0.64, 0.0]

    # Keep thumb state controllable for rule tests.
    hand[MCP["thumb"]] = [0.40, 0.58, 0.0]
    if thumb_open:
        if thumb_pose == "up":
            hand[PIP["thumb"]] = [0.44, 0.55, 0.0]
            hand[TIP["thumb"]] = [0.32, 0.48, 0.0]
        elif thumb_pose == "down":
            hand[PIP["thumb"]] = [0.44, 0.64, 0.0]
            hand[TIP["thumb"]] = [0.32, 0.76, 0.0]
        else:
            hand[PIP["thumb"]] = [0.44, 0.58, 0.0]
            hand[TIP["thumb"]] = [0.32, 0.58, 0.0]
    else:
        hand[PIP["thumb"]] = [0.46, 0.58, 0.0]
        hand[TIP["thumb"]] = [0.45, 0.58, 0.0]

    return hand


def _frame(hand):
    return SimpleNamespace(left=hand, right=None)


def test_aid_sign_list():
    labels = {s.label for s in AID_SIGNS}
    expected = {
        "need_water",
        "need_food",
        "need_toilet",
        "call_family",
        "hospital_help",
        "emergency",
        "severe_pain",
        "cannot_breathe",
        "bleeding",
        "head_injury",
        "chest_pain",
        "yes",
        "no",
    }
    assert expected.issubset(labels)
    assert "stop" not in labels
    assert "repeat" not in labels


def test_aid_yes_no():
    dec = AidDecoder()

    yes_hit = dec.decode(_frame(_make_hand(open_fingers=set(), thumb_open=True, thumb_pose="up")))
    no_hit = dec.decode(_frame(_make_hand(open_fingers=set(), thumb_open=True, thumb_pose="down")))

    assert yes_hit is not None and yes_hit.label == "yes"
    assert no_hit is not None and no_hit.label == "no"


def test_aid_medical_intents():
    dec = AidDecoder()

    pain_hit = dec.decode(_frame(_make_hand(open_fingers={"index", "pinky"})))
    breathe_hit = dec.decode(_frame(_make_hand(open_fingers={"index"}, thumb_open=True)))
    bleed_hit = dec.decode(_frame(_make_hand(open_fingers={"index", "ring"})))

    assert pain_hit is not None and pain_hit.label == "severe_pain"
    assert breathe_hit is not None and breathe_hit.label == "cannot_breathe"
    assert bleed_hit is not None and bleed_hit.label == "bleeding"
