from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


@dataclass
class Prediction:
    label: str
    confidence: float


class Rules20:
    """
    Rule-based recognizer for ~20 hackathon demo signs.

    Works best for:
    - single clear hand in frame
    - moderate lighting
    - simple backgrounds
    """

    def __init__(self, wave_window: int = 14) -> None:
        self.wrist_x_hist: deque[float] = deque(maxlen=wave_window)

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _finger_states(self, hand: np.ndarray) -> dict[str, bool]:
        s: dict[str, bool] = {}

        # Index/middle/ring/pinky extension
        for n in ("index", "middle", "ring", "pinky"):
            s[n] = bool(hand[TIP[n], 1] < hand[PIP[n], 1] < hand[MCP[n], 1] + 0.02)

        # Thumb extension (robust-ish)
        thumb_tip = hand[TIP["thumb"]]
        thumb_ip = hand[PIP["thumb"]]
        wrist = hand[0]
        palm_center = (hand[MCP["index"]] + hand[MCP["middle"]] + hand[MCP["ring"]] + hand[MCP["pinky"]]) / 4.0
        open_metric = self._dist(thumb_tip, palm_center)
        fold_metric = self._dist(thumb_ip, palm_center)
        reach_tip = self._dist(thumb_tip, wrist)
        reach_ip = self._dist(thumb_ip, wrist)
        side_open = abs(float(thumb_tip[0] - hand[MCP["index"]][0])) > 0.07
        vertical_open = float(thumb_tip[1]) < float(thumb_ip[1] - 0.01)
        s["thumb"] = bool((open_metric > fold_metric * 1.04) and (reach_tip > reach_ip * 1.02) and (side_open or vertical_open))
        return s

    def _is_open_palm(self, s: dict[str, bool]) -> bool:
        return s["index"] and s["middle"] and s["ring"] and s["pinky"]

    def _is_fist(self, s: dict[str, bool]) -> bool:
        return not any(s.values())

    def _finger_count(self, s: dict[str, bool]) -> int:
        return int(s["thumb"]) + int(s["index"]) + int(s["middle"]) + int(s["ring"]) + int(s["pinky"])

    def _is_wave(self, hand: np.ndarray, open_palm: bool) -> bool:
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

    @staticmethod
    def _time_greeting() -> str:
        h = datetime.now().hour
        if 5 <= h < 12:
            return "GOOD MORNING"
        if 12 <= h < 17:
            return "GOOD AFTERNOON"
        if 17 <= h < 21:
            return "GOOD EVENING"
        return "GOOD NIGHT"

    def _single_hand(self, hand: np.ndarray) -> Optional[Prediction]:
        s = self._finger_states(hand)
        open_palm = self._is_open_palm(s)

        if self._is_wave(hand, open_palm):
            return Prediction("HELLO", 0.94)

        # Thumb-only YES/NO
        others_folded = all(hand[TIP[n], 1] > hand[PIP[n], 1] - 0.004 for n in ("index", "middle", "ring", "pinky"))
        if others_folded:
            thumb_tip = hand[TIP["thumb"]]
            wrist = hand[0]
            if thumb_tip[1] < wrist[1] - 0.06:
                return Prediction("YES", 0.90)
            if thumb_tip[1] > wrist[1] + 0.06:
                return Prediction("NO", 0.90)

        count = self._finger_count(s)
        if count == 1 and s["index"] and not s["thumb"]:
            return Prediction("ONE", 0.84)

        if s["index"] and s["middle"] and not s["ring"] and not s["pinky"]:
            spread = self._dist(hand[TIP["index"]], hand[TIP["middle"]])
            if spread > 0.09:
                return Prediction("PEACE", 0.85)
            return Prediction("TWO", 0.83)

        if s["index"] and s["middle"] and s["ring"] and not s["pinky"] and not s["thumb"]:
            return Prediction("THREE", 0.82)

        if s["index"] and s["middle"] and s["ring"] and s["pinky"] and not s["thumb"]:
            return Prediction("FOUR", 0.83)

        if open_palm and s["thumb"]:
            return Prediction("FIVE", 0.84)

        if self._is_fist(s):
            return Prediction("STOP", 0.82)

        if self._dist(hand[TIP["index"]], hand[TIP["thumb"]]) < 0.05 and s["middle"] and s["ring"] and s["pinky"]:
            return Prediction("OKAY", 0.86)

        if s["thumb"] and s["pinky"] and not s["index"] and not s["middle"] and not s["ring"]:
            return Prediction("CALL ME", 0.83)

        if s["index"] and s["pinky"] and not s["middle"] and not s["ring"]:
            if s["thumb"]:
                return Prediction("I LOVE YOU", 0.86)
            return Prediction("ROCK", 0.82)

        return None

    def predict(self, hand_count: int, raw_hands: list[np.ndarray]) -> Optional[Prediction]:
        if hand_count == 0 or not raw_hands:
            self.wrist_x_hist.clear()
            return None

        # Two-hand rules
        if hand_count >= 2 and len(raw_hands) >= 2:
            s0 = self._finger_states(raw_hands[0])
            s1 = self._finger_states(raw_hands[1])

            if self._is_fist(s0) and self._is_fist(s1):
                return Prediction("HELP", 0.85)

            if self._is_open_palm(s0) and self._is_open_palm(s1):
                c0 = raw_hands[0][0]
                c1 = raw_hands[1][0]
                dist = self._dist(c0, c1)
                if dist < 0.18:
                    return Prediction("THANK YOU", 0.88)
                return Prediction(self._time_greeting(), 0.86)

        # Single-hand fallback
        return self._single_hand(raw_hands[0])

