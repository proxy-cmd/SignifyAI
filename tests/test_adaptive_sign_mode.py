from types import SimpleNamespace

import numpy as np

from modes.adaptive_sign_mode import AdaptiveSignDecoder, PrototypeStore


def _blank_hand():
    hand = np.zeros((21, 3), dtype=np.float32)
    # basic wrist + palm anchors
    hand[0] = [0.5, 0.8, 0.0]
    hand[5] = [0.45, 0.65, 0.0]
    hand[9] = [0.50, 0.64, 0.0]
    hand[13] = [0.55, 0.65, 0.0]
    hand[17] = [0.60, 0.67, 0.0]
    return hand


def _frame_with_left(hand):
    return SimpleNamespace(left=hand, right=None)


def test_store_roundtrip(tmp_path):
    path = tmp_path / "proto.json"
    store = PrototypeStore(path=path)
    vec = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    store.add("drink_water", vec)

    store2 = PrototypeStore(path=path)
    hit = store2.best_match(np.asarray([0.1, 0.2, 0.31], dtype=np.float32), max_dist=0.2)
    assert hit is not None
    assert hit.label == "drink_water"


def test_rule_one_index_only(tmp_path):
    dec = AdaptiveSignDecoder()
    dec.store = PrototypeStore(path=tmp_path / "proto.json")

    hand = _blank_hand()
    # Folded fingers baseline
    hand[6, 1] = 0.66
    hand[8, 1] = 0.72
    hand[10, 1] = 0.66
    hand[12, 1] = 0.72
    hand[14, 1] = 0.67
    hand[16, 1] = 0.73
    # Index up
    hand[6, 1] = 0.61
    hand[8, 1] = 0.52
    # Thumb folded
    hand[2] = [0.44, 0.74, 0.0]
    hand[3] = [0.45, 0.73, 0.0]
    hand[4] = [0.46, 0.72, 0.0]

    hit = dec.decode(_frame_with_left(hand))
    assert hit is not None
    assert hit.label == "one"


def test_teach_and_match(tmp_path):
    dec = AdaptiveSignDecoder()
    dec.store = PrototypeStore(path=tmp_path / "proto.json")

    hand = _blank_hand()
    # Create a distinct pose
    hand[8] = [0.47, 0.55, 0.0]
    hand[12] = [0.52, 0.56, 0.0]
    hand[16] = [0.56, 0.62, 0.0]
    hand[20] = [0.61, 0.68, 0.0]

    frame = _frame_with_left(hand)
    assert dec.teach(frame, "custom_wave") is True

    hit = dec.decode(frame)
    assert hit is not None
    assert hit.label == "custom_wave"


def test_teach_two_signs_both_match(tmp_path):
    dec = AdaptiveSignDecoder()
    dec.store = PrototypeStore(path=tmp_path / "proto.json")

    hand_a = _blank_hand()
    hand_a[8] = [0.42, 0.52, 0.0]
    hand_a[12] = [0.50, 0.59, 0.0]

    hand_b = _blank_hand()
    hand_b[8] = [0.62, 0.72, 0.0]
    hand_b[12] = [0.65, 0.74, 0.0]

    frame_a = _frame_with_left(hand_a)
    frame_b = _frame_with_left(hand_b)

    assert dec.teach(frame_a, "sign_a") is True
    assert dec.teach(frame_b, "sign_b") is True

    hit_a = dec.decode(frame_a)
    hit_b = dec.decode(frame_b)
    assert hit_a is not None and hit_a.label == "sign_a"
    assert hit_b is not None and hit_b.label == "sign_b"
