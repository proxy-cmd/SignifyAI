from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import LandmarkFrame
from .rules_intents import IntentHit

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


@dataclass
class DemoSign:
    label: str
    hint: str


DEMO_SIGNS: list[DemoSign] = [
    DemoSign("hello", "Open palm"),
    DemoSign("yes", "Thumb up"),
    DemoSign("no", "Thumb down"),
    DemoSign("one", "Only index up"),
    DemoSign("two", "Index + middle up"),
    DemoSign("three", "Index + middle + ring up"),
    DemoSign("four", "Four fingers up (no thumb)"),
    DemoSign("five", "All five fingers up"),
    DemoSign("stop", "Closed fist"),
]


class DemoIntentDecoder:
    @staticmethod
    def finger_states(hand: np.ndarray) -> dict[str, bool]:
        state: dict[str, bool] = {}
        for name in ("index", "middle", "ring", "pinky"):
            state[name] = bool(hand[TIP[name], 1] < hand[PIP[name], 1] - 0.01)

        thumb_tip = hand[TIP["thumb"]]
        thumb_ip = hand[PIP["thumb"]]
        state["thumb"] = bool(abs(float(thumb_tip[0] - thumb_ip[0])) > 0.03 or abs(float(thumb_tip[1] - thumb_ip[1])) > 0.03)
        return state

    @staticmethod
    def count_open(state: dict[str, bool]) -> int:
        return int(state["thumb"]) + int(state["index"]) + int(state["middle"]) + int(state["ring"]) + int(state["pinky"])

    def decode(self, frame: LandmarkFrame) -> IntentHit | None:
        hands = [hand for hand in (frame.left_hand, frame.right_hand) if hand is not None]
        if not hands:
            return None

        hand = hands[0]
        state = self.finger_states(hand)
        count = self.count_open(state)

        if count == 0:
            return IntentHit("stop", 0.90, source="demo")
        if count == 1 and state["index"] and not state["thumb"]:
            return IntentHit("one", 0.90, source="demo")
        if count == 2 and state["index"] and state["middle"] and not state["thumb"]:
            return IntentHit("two", 0.90, source="demo")
        if count == 3 and state["index"] and state["middle"] and state["ring"] and not state["thumb"]:
            return IntentHit("three", 0.90, source="demo")
        if count == 4 and state["index"] and state["middle"] and state["ring"] and state["pinky"] and not state["thumb"]:
            return IntentHit("four", 0.90, source="demo")
        if count == 5:
            return IntentHit("five", 0.92, source="demo")

        folded_others = not state["index"] and not state["middle"] and not state["ring"] and not state["pinky"]
        if folded_others and state["thumb"]:
            thumb_tip = hand[TIP["thumb"]]
            wrist = hand[0]
            if thumb_tip[1] < wrist[1] - 0.05:
                return IntentHit("yes", 0.92, source="demo")
            if thumb_tip[1] > wrist[1] + 0.05:
                return IntentHit("no", 0.92, source="demo")

        open_palm = state["index"] and state["middle"] and state["ring"] and state["pinky"]
        if open_palm:
            return IntentHit("hello", 0.85, source="demo")

        return None
