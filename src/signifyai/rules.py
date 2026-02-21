from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .hand_tracking import DetectionResult


TIP = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
PIP = {
    "thumb": 3,
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}
MCP = {
    "thumb": 2,
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}


@dataclass
class RulePrediction:
    label: str
    confidence: float


class RuleBasedInterpreter:
    """Heuristic gesture recognizer for presentation-grade prototype behavior."""

    def __init__(self, wave_window: int = 16) -> None:
        self.wrist_x_hist: deque[float] = deque(maxlen=wave_window)

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _finger_states(self, hand: np.ndarray) -> dict[str, bool]:
        states: dict[str, bool] = {}

        # Index/middle/ring/pinky: tip above pip and mcp means likely extended.
        for name in ("index", "middle", "ring", "pinky"):
            tip_y = hand[TIP[name], 1]
            pip_y = hand[PIP[name], 1]
            mcp_y = hand[MCP[name], 1]
            states[name] = bool(tip_y < pip_y < (mcp_y + 0.02))

        # Thumb: combine reach + side spread + vertical direction.
        thumb_tip = hand[TIP["thumb"]]
        thumb_ip = hand[PIP["thumb"]]
        index_mcp = hand[MCP["index"]]
        wrist = hand[0]
        palm_center = (hand[MCP["index"]] + hand[MCP["middle"]] + hand[MCP["ring"]] + hand[MCP["pinky"]]) / 4.0

        open_metric = self._dist(thumb_tip, palm_center)
        fold_metric = self._dist(thumb_ip, palm_center)
        reach_tip = self._dist(thumb_tip, wrist)
        reach_ip = self._dist(thumb_ip, wrist)
        side_open = abs(float(thumb_tip[0] - index_mcp[0])) > 0.07
        vertical_open = float(thumb_tip[1]) < float(thumb_ip[1] - 0.01)
        states["thumb"] = bool(
            (open_metric > (fold_metric * 1.05))
            and (reach_tip > (reach_ip * 1.03))
            and (side_open or vertical_open)
        )

        return states

    def _is_open_palm(self, states: dict[str, bool]) -> bool:
        return states["index"] and states["middle"] and states["ring"] and states["pinky"]

    def _is_fist(self, states: dict[str, bool]) -> bool:
        return not states["thumb"] and not states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]

    def _is_finger_folded(self, hand: np.ndarray, name: str) -> bool:
        return bool(hand[TIP[name], 1] > hand[PIP[name], 1] - 0.005)

    def _thumb_only_label(self, hand: np.ndarray, states: dict[str, bool]) -> Optional[RulePrediction]:
        others_folded = all(self._is_finger_folded(hand, n) for n in ("index", "middle", "ring", "pinky"))
        if not others_folded:
            return None

        thumb_tip = hand[TIP["thumb"]]
        thumb_ip = hand[PIP["thumb"]]
        wrist = hand[0]
        index_mcp = hand[MCP["index"]]
        pinky_mcp = hand[MCP["pinky"]]
        palm_center = (index_mcp + hand[MCP["middle"]] + hand[MCP["ring"]] + pinky_mcp) / 4.0
        thumb_len = self._dist(thumb_tip, thumb_ip)
        reach = self._dist(thumb_tip, wrist)
        if thumb_len < 0.03 or reach < 0.10:
            return None

        # Thumb should be notably away from palm for a true thumb-only sign.
        thumb_to_palm = self._dist(thumb_tip, palm_center)
        if thumb_to_palm < 0.085:
            return None

        # Reject side-pointing thumb (often confused near a fist/STOP).
        # Keep mostly vertical thumbs for YES/NO.
        dy = float(thumb_tip[1] - thumb_ip[1])
        dx = float(thumb_tip[0] - thumb_ip[0])
        if abs(dx) > abs(dy) * 1.25:
            return None

        # Strong vertical tests for thumbs up/down.
        if thumb_tip[1] < (wrist[1] - 0.06):
            return RulePrediction("YES", 0.90)
        if thumb_tip[1] > (wrist[1] + 0.06):
            return RulePrediction("NO", 0.90)

        # Fallback direction via thumb tip vs ip.
        if states["thumb"]:
            if thumb_tip[1] < thumb_ip[1]:
                return RulePrediction("YES", 0.82)
            if thumb_tip[1] > thumb_ip[1]:
                return RulePrediction("NO", 0.82)

        return None

    def _is_wave(self, hand: np.ndarray, is_open_palm: bool) -> bool:
        if not is_open_palm:
            self.wrist_x_hist.clear()
            return False

        wrist_x = float(hand[0, 0])
        self.wrist_x_hist.append(wrist_x)

        if len(self.wrist_x_hist) < 8:
            return False

        xs = np.asarray(self.wrist_x_hist, dtype=np.float32)
        amplitude = float(xs.max() - xs.min())

        dx = np.diff(xs)
        dirs = np.sign(dx)
        dirs[np.abs(dx) < 0.01] = 0
        dirs = dirs[dirs != 0]
        if len(dirs) < 4:
            return False

        turns = int(np.sum(np.abs(np.diff(dirs)) > 0))
        return amplitude > 0.12 and turns >= 2

    def _single_hand_rule(self, hand: np.ndarray) -> Optional[RulePrediction]:
        states = self._finger_states(hand)

        index_tip = hand[TIP["index"]]
        thumb_tip = hand[TIP["thumb"]]
        ok_dist = self._dist(index_tip, thumb_tip)

        open_palm = self._is_open_palm(states)

        # Priority: explicit wave hello
        if self._is_wave(hand, open_palm):
            return RulePrediction("HELLO", 0.95)

        # Show palm in front -> hello
        if open_palm:
            return RulePrediction("HELLO", 0.90)

        # Thumb-only gestures first (avoids confusion with STOP).
        thumb_only = self._thumb_only_label(hand, states)
        if thumb_only is not None:
            return thumb_only

        if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
            # Differentiate TWO vs PEACE by index-middle spread.
            spread = self._dist(hand[TIP["index"]], hand[TIP["middle"]])
            if spread > 0.09:
                return RulePrediction("PEACE", 0.84)
            return RulePrediction("TWO", 0.82)

        if self._is_fist(states):
            return RulePrediction("STOP", 0.80)

        if states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]:
            return RulePrediction("ONE", 0.83)

        if ok_dist < 0.05 and states["middle"] and states["ring"] and states["pinky"]:
            return RulePrediction("OKAY", 0.88)

        if states["thumb"] and states["pinky"] and not states["index"] and not states["middle"] and not states["ring"]:
            return RulePrediction("CALL ME", 0.82)

        if states["index"] and states["pinky"] and not states["middle"] and not states["ring"]:
            if states["thumb"]:
                return RulePrediction("I LOVE YOU", 0.86)
            return RulePrediction("ROCK", 0.80)

        return None

    def predict(self, detection: DetectionResult) -> Optional[RulePrediction]:
        if detection.hand_count == 0 or not detection.raw_hands:
            self.wrist_x_hist.clear()
            return None

        # Two-hand composite rules first.
        if detection.hand_count >= 2 and len(detection.raw_hands) >= 2:
            s0 = self._finger_states(detection.raw_hands[0])
            s1 = self._finger_states(detection.raw_hands[1])
            if self._is_open_palm(s0) and self._is_open_palm(s1):
                return RulePrediction("THANK YOU", 0.88)
            if self._is_fist(s0) and self._is_fist(s1):
                return RulePrediction("HELP", 0.82)

        # Fallback single-hand recognition on first hand.
        return self._single_hand_rule(detection.raw_hands[0])
