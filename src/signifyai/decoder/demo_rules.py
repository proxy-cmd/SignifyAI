from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..contracts import LandmarkFrame
from .rules_intents import IntentHit

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


@dataclass
class DemoSign:
    label: str
    hint: str


DEMO_SIGNS: list[DemoSign] = [
    DemoSign("hello", "Open palm + wave"),
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
    def __init__(self) -> None:
        self.wrist_x_hist: deque[float] = deque(maxlen=14)

    @staticmethod
    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def finger_states(self, hand: np.ndarray) -> dict[str, bool]:
        state: dict[str, bool] = {}
        for name in ("index", "middle", "ring", "pinky"):
            tip_y = float(hand[TIP[name], 1])
            pip_y = float(hand[PIP[name], 1])
            mcp_y = float(hand[MCP[name], 1])
            state[name] = bool(tip_y < (pip_y - 0.018) and pip_y < (mcp_y - 0.003))

        thumb_tip = hand[TIP["thumb"]]
        thumb_ip = hand[PIP["thumb"]]
        idx_mcp = hand[MCP["index"]]
        thumb_open_dist = self.dist(thumb_tip, idx_mcp) > self.dist(thumb_ip, idx_mcp) * 1.08
        thumb_move = abs(float(thumb_tip[0] - thumb_ip[0])) > 0.035 or abs(float(thumb_tip[1] - thumb_ip[1])) > 0.035
        state["thumb"] = bool(thumb_open_dist or thumb_move)
        return state

    @staticmethod
    def count_open(state: dict[str, bool]) -> int:
        return int(state["thumb"]) + int(state["index"]) + int(state["middle"]) + int(state["ring"]) + int(state["pinky"])

    def is_wave(self, hand: np.ndarray, open_palm: bool) -> bool:
        if not open_palm:
            self.wrist_x_hist.clear()
            return False

        self.wrist_x_hist.append(float(hand[0, 0]))
        if len(self.wrist_x_hist) < 8:
            return False

        xs = np.asarray(self.wrist_x_hist, dtype=np.float32)
        amplitude = float(xs.max() - xs.min())
        dx = np.diff(xs)
        dirs = np.sign(dx)
        dirs[np.abs(dx) < 0.01] = 0
        dirs = dirs[dirs != 0]
        if len(dirs) < 3:
            return False
        turns = int(np.sum(np.abs(np.diff(dirs)) > 0))
        return amplitude > 0.10 and turns >= 2

    def decode(self, frame: LandmarkFrame) -> IntentHit | None:
        hands = [hand for hand in (frame.left_hand, frame.right_hand) if hand is not None]
        if not hands:
            self.wrist_x_hist.clear()
            return None

        hand = hands[0]
        state = self.finger_states(hand)
        count = self.count_open(state)

        folded_others = not state["index"] and not state["middle"] and not state["ring"] and not state["pinky"]
        if folded_others and state["thumb"]:
            thumb_tip = hand[TIP["thumb"]]
            wrist = hand[0]
            if thumb_tip[1] < wrist[1] - 0.05:
                return IntentHit("yes", 0.92, source="demo")
            if thumb_tip[1] > wrist[1] + 0.05:
                return IntentHit("no", 0.92, source="demo")

        open_palm = state["index"] and state["middle"] and state["ring"] and state["pinky"] and state["thumb"]
        if open_palm and self.is_wave(hand, open_palm=True):
            return IntentHit("hello", 0.94, source="demo")
        if open_palm:
            return IntentHit("five", 0.92, source="demo")

        if count <= 1 and not state["index"] and not state["middle"] and not state["ring"] and not state["pinky"]:
            return IntentHit("stop", 0.90, source="demo")

        if state["index"] and not state["middle"] and not state["ring"] and not state["pinky"]:
            return IntentHit("one", 0.90, source="demo")
        if state["index"] and state["middle"] and not state["ring"] and not state["pinky"]:
            return IntentHit("two", 0.90, source="demo")
        if state["index"] and state["middle"] and state["ring"] and not state["pinky"]:
            return IntentHit("three", 0.90, source="demo")
        if state["index"] and state["middle"] and state["ring"] and state["pinky"] and not state["thumb"]:
            return IntentHit("four", 0.90, source="demo")

        return None
