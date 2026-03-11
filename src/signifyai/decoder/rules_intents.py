from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from ..contracts import LandmarkFrame

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


@dataclass
class IntentHit:
    intent_id: str
    confidence: float
    source: str = "rules"


class RuleIntentDecoder:
    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _finger_states(self, hand: np.ndarray) -> dict[str, bool]:
        state: dict[str, bool] = {}
        for name in ("index", "middle", "ring", "pinky"):
            state[name] = bool(hand[TIP[name], 1] < hand[PIP[name], 1] < hand[MCP[name], 1] + 0.02)

        thumb_tip = hand[TIP["thumb"]]
        thumb_ip = hand[PIP["thumb"]]
        wrist = hand[0]
        palm = (hand[MCP["index"]] + hand[MCP["middle"]] + hand[MCP["ring"]] + hand[MCP["pinky"]]) / 4.0
        state["thumb"] = bool(
            self._dist(thumb_tip, palm) > self._dist(thumb_ip, palm) * 1.04
            and self._dist(thumb_tip, wrist) > self._dist(thumb_ip, wrist) * 1.02
        )
        return state

    @staticmethod
    def _time_intent() -> str:
        h = datetime.now().hour
        if h < 12:
            return "hello"
        return "thank_you"

    def decode(self, frame: LandmarkFrame) -> Optional[IntentHit]:
        if frame.hand_count == 0:
            return None

        hands: list[np.ndarray] = []
        if frame.left_hand is not None:
            hands.append(frame.left_hand)
        if frame.right_hand is not None:
            hands.append(frame.right_hand)

        if len(hands) >= 2:
            a = self._finger_states(hands[0])
            b = self._finger_states(hands[1])
            if (not any(a.values())) and (not any(b.values())):
                return IntentHit("emergency", 0.82)
            if a["index"] and b["index"]:
                return IntentHit("hospital_help", 0.84)
            return IntentHit("call_family", 0.70)

        hand = hands[0]
        s = self._finger_states(hand)

        if s["index"] and s["middle"] and not s["ring"] and not s["pinky"]:
            spread = self._dist(hand[TIP["index"]], hand[TIP["middle"]])
            if spread > 0.09:
                return IntentHit("thank_you", 0.80)
            return IntentHit("need_water", 0.78)

        if s["thumb"] and not s["index"] and not s["middle"] and not s["ring"] and not s["pinky"]:
            tip = hand[TIP["thumb"]]
            wrist = hand[0]
            if tip[1] < wrist[1] - 0.05:
                return IntentHit("yes", 0.86)
            if tip[1] > wrist[1] + 0.05:
                return IntentHit("no", 0.86)

        if s["thumb"] and s["index"] and s["middle"] and s["ring"] and s["pinky"]:
            return IntentHit(self._time_intent(), 0.83)

        if (not any(s.values())):
            return IntentHit("need_toilet", 0.72)

        return IntentHit("need_food", 0.66)
