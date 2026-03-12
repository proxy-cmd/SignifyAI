from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import LandmarkFrame
from .rules_intents import IntentHit

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


@dataclass
class AidSign:
    label: str
    hint: str


AID_SIGNS: list[AidSign] = [
    AidSign("need_water", "Index finger only"),
    AidSign("need_food", "Index + middle"),
    AidSign("need_toilet", "Pinky finger only"),
    AidSign("call_family", "Three fingers (index+middle+ring)"),
    AidSign("hospital_help", "Open palm"),
    AidSign("emergency", "Closed fist"),
]


class AidIntentDecoder:
    @staticmethod
    def finger_states(hand: np.ndarray) -> dict[str, bool]:
        state: dict[str, bool] = {}
        for name in ("index", "middle", "ring", "pinky"):
            tip_y = float(hand[TIP[name], 1])
            pip_y = float(hand[PIP[name], 1])
            mcp_y = float(hand[MCP[name], 1])
            state[name] = bool(tip_y < (pip_y - 0.018) and pip_y < (mcp_y - 0.003))

        thumb_tip = hand[TIP["thumb"]]
        thumb_ip = hand[PIP["thumb"]]
        idx_mcp = hand[MCP["index"]]
        thumb_open = (abs(float(thumb_tip[0] - thumb_ip[0])) > 0.035) or (float(np.linalg.norm(thumb_tip[:2] - idx_mcp[:2])) > 0.14)
        state["thumb"] = bool(thumb_open)
        return state

    def decode(self, frame: LandmarkFrame) -> IntentHit | None:
        hands = [hand for hand in (frame.left_hand, frame.right_hand) if hand is not None]
        if not hands:
            return None

        hand = hands[0]
        s = self.finger_states(hand)

        # Priority order: specific signals before generic ones.
        if s["index"] and not s["middle"] and not s["ring"] and not s["pinky"]:
            return IntentHit("need_water", 0.92, source="aid")
        if s["index"] and s["middle"] and not s["ring"] and not s["pinky"]:
            return IntentHit("need_food", 0.92, source="aid")
        if s["pinky"] and not s["index"] and not s["middle"] and not s["ring"]:
            return IntentHit("need_toilet", 0.92, source="aid")
        if s["index"] and s["middle"] and s["ring"] and not s["pinky"]:
            return IntentHit("call_family", 0.90, source="aid")

        open_palm = s["thumb"] and s["index"] and s["middle"] and s["ring"] and s["pinky"]
        if open_palm:
            return IntentHit("hospital_help", 0.90, source="aid")

        fist = not s["index"] and not s["middle"] and not s["ring"] and not s["pinky"]
        if fist:
            return IntentHit("emergency", 0.90, source="aid")

        return None
