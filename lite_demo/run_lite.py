r"""SignifyAI Lite Demo
Simple, stable hackathon-focused demo app.

Run:
    python -u .\lite_demo\run_lite.py
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Optional

import cv2
import numpy as np

# Ensure `src` is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from signifyai.hand_tracking import HandTracker, check_camera, open_camera, warmup_camera  # noqa: E402
from signifyai.tts import SpeechEngine  # noqa: E402


TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


@dataclass
class LitePrediction:
    label: str
    confidence: float


class LiteRules20:
    """Simple hand-sign rules tuned for stable live demos."""

    def __init__(self, wave_window: int = 14) -> None:
        self.wrist_x_hist: deque[float] = deque(maxlen=wave_window)

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _finger_states(self, hand: np.ndarray) -> dict[str, bool]:
        states: dict[str, bool] = {}
        for n in ("index", "middle", "ring", "pinky"):
            states[n] = bool(hand[TIP[n], 1] < hand[PIP[n], 1] < hand[MCP[n], 1] + 0.02)

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
        states["thumb"] = bool(
            (open_metric > fold_metric * 1.04)
            and (reach_tip > reach_ip * 1.02)
            and (side_open or vertical_open)
        )
        return states

    def _is_open_palm(self, s: dict[str, bool]) -> bool:
        return s["index"] and s["middle"] and s["ring"] and s["pinky"]

    def _is_fist(self, s: dict[str, bool]) -> bool:
        return not any(s.values())

    def _finger_count(self, s: dict[str, bool]) -> int:
        return int(s["thumb"]) + int(s["index"]) + int(s["middle"]) + int(s["ring"]) + int(s["pinky"])

    def _is_wave(self, hand: np.ndarray, is_open_palm: bool) -> bool:
        if not is_open_palm:
            self.wrist_x_hist.clear()
            return False
        self.wrist_x_hist.append(float(hand[0, 0]))
        if len(self.wrist_x_hist) < 8:
            return False
        xs = np.asarray(self.wrist_x_hist, dtype=np.float32)
        amp = float(xs.max() - xs.min())
        dx = np.diff(xs)
        dirs = np.sign(dx)
        dirs[np.abs(dx) < 0.01] = 0
        dirs = dirs[dirs != 0]
        turns = int(np.sum(np.abs(np.diff(dirs)) > 0)) if len(dirs) >= 2 else 0
        return amp > 0.10 and turns >= 2

    def _time_greeting(self) -> str:
        h = datetime.now().hour
        if 5 <= h < 12:
            return "GOOD MORNING"
        if 12 <= h < 17:
            return "GOOD AFTERNOON"
        if 17 <= h < 21:
            return "GOOD EVENING"
        return "GOOD NIGHT"

    def _single_hand(self, hand: np.ndarray) -> Optional[LitePrediction]:
        s = self._finger_states(hand)
        open_palm = self._is_open_palm(s)
        ok_dist = self._dist(hand[TIP["index"]], hand[TIP["thumb"]])
        if self._is_wave(hand, open_palm):
            return LitePrediction("HELLO", 0.94)

        # thumbs up/down
        others_folded = all(hand[TIP[n], 1] > hand[PIP[n], 1] - 0.004 for n in ("index", "middle", "ring", "pinky"))
        if others_folded:
            thumb_tip = hand[TIP["thumb"]]
            thumb_ip = hand[PIP["thumb"]]
            wrist = hand[0]
            index_mcp = hand[MCP["index"]]
            pinky_mcp = hand[MCP["pinky"]]
            dy = float(thumb_tip[1] - thumb_ip[1])
            dx = float(thumb_tip[0] - thumb_ip[0])
            if abs(dx) <= abs(dy) * 1.10 and thumb_tip[1] < min((wrist[1] - 0.055), (min(float(index_mcp[1]), float(pinky_mcp[1])) - 0.02)):
                return LitePrediction("YES", 0.90)
            if abs(dx) <= abs(dy) * 1.10 and thumb_tip[1] > max((wrist[1] + 0.055), (max(float(index_mcp[1]), float(pinky_mcp[1])) + 0.02)):
                return LitePrediction("NO", 0.90)

        if ok_dist < 0.042 and s["middle"] and s["ring"] and s["pinky"]:
            return LitePrediction("OKAY", 0.90)

        # finger count classes
        count = self._finger_count(s)
        if count == 1 and s["index"] and not s["thumb"]:
            return LitePrediction("ONE", 0.84)
        if s["index"] and s["middle"] and not s["ring"] and not s["pinky"]:
            spread = self._dist(hand[TIP["index"]], hand[TIP["middle"]])
            if spread > 0.09:
                return LitePrediction("PEACE", 0.85)
            return LitePrediction("TWO", 0.83)
        if s["index"] and s["middle"] and s["ring"] and not s["pinky"] and not s["thumb"]:
            return LitePrediction("THREE", 0.82)
        if s["index"] and s["middle"] and s["ring"] and s["pinky"] and not s["thumb"]:
            return LitePrediction("FOUR", 0.83)
        if open_palm and s["thumb"]:
            return LitePrediction("FIVE", 0.84)

        # symbolic signs
        if self._is_fist(s):
            return LitePrediction("STOP", 0.82)

        if s["thumb"] and s["pinky"] and not s["index"] and not s["middle"] and not s["ring"]:
            return LitePrediction("CALL ME", 0.83)

        if s["index"] and s["pinky"] and not s["middle"] and not s["ring"]:
            palm_center = (hand[MCP["index"]] + hand[MCP["middle"]] + hand[MCP["ring"]] + hand[MCP["pinky"]]) / 4.0
            thumb_tip = hand[TIP["thumb"]]
            thumb_far = self._dist(thumb_tip, palm_center) > 0.115 or abs(float(thumb_tip[0] - hand[MCP["index"]][0])) > 0.09
            thumb_folded = self._dist(thumb_tip, palm_center) < 0.090
            if s["thumb"] and thumb_far:
                return LitePrediction("I LOVE YOU", 0.86)
            if thumb_folded:
                return LitePrediction("ROCK", 0.84)
            return LitePrediction("ROCK", 0.82)

        return None

    def predict(self, hand_count: int, raw_hands: list[np.ndarray]) -> Optional[LitePrediction]:
        if hand_count == 0 or not raw_hands:
            self.wrist_x_hist.clear()
            return None

        if hand_count >= 2 and len(raw_hands) >= 2:
            s0 = self._finger_states(raw_hands[0])
            s1 = self._finger_states(raw_hands[1])
            if self._is_fist(s0) and self._is_fist(s1):
                return LitePrediction("HELP", 0.85)
            if self._is_open_palm(s0) and self._is_open_palm(s1):
                c0 = raw_hands[0][0]
                c1 = raw_hands[1][0]
                dist = self._dist(c0, c1)
                y_gap = abs(float(c0[1] - c1[1]))
                if dist < 0.20 and y_gap < 0.085:
                    return LitePrediction("THANK YOU", 0.88)
                return LitePrediction(self._time_greeting(), 0.86)

        return self._single_hand(raw_hands[0])


def speech_for_label(label: str) -> str:
    pretty = label.replace("_", " ").title()
    if pretty.upper().startswith("GOOD "):
        return pretty
    if pretty == "I Love You":
        return "I love you"
    return pretty


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    tuned = cv2.convertScaleAbs(frame, alpha=1.05, beta=4)
    blur = cv2.GaussianBlur(tuned, (0, 0), 1.1)
    sharp = cv2.addWeighted(tuned, 1.20, blur, -0.20, 0)
    return sharp


def run_lite() -> None:
    cap = open_camera(index=0, width=960, height=720)
    err = check_camera(cap)
    if err:
        raise RuntimeError(err)
    warmup_camera(cap)

    tracker = HandTracker(max_num_hands=2, inference_scale=0.75, min_detection_confidence=0.72, min_tracking_confidence=0.68)
    rules = LiteRules20()
    tts = SpeechEngine(rate=170, volume=1.0)

    cv2.namedWindow("SignifyAI Lite Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SignifyAI Lite Demo", 960, 720)

    voice_on = True
    last_label = "NO_HAND"
    spoken_label = ""
    last_spoken = 0.0
    no_hand_streak = 0
    pred_window: deque[str] = deque(maxlen=7)
    prev_t = time.time()
    fps = 0.0

    signs_list = [
        "HELLO", "YES", "NO", "STOP", "ONE", "TWO", "THREE", "FOUR", "FIVE", "PEACE",
        "OKAY", "CALL ME", "ROCK", "I LOVE YOU", "THANK YOU", "HELP",
        "GOOD MORNING", "GOOD AFTERNOON", "GOOD EVENING", "GOOD NIGHT",
    ]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame = enhance_frame(frame)
            det = tracker.process(frame, draw=True)
            out = det.frame

            pred = rules.predict(det.hand_count, det.raw_hands)
            if det.hand_count == 0:
                pred_window.append("NO_HAND")
            elif pred is not None:
                pred_window.append(pred.label)
            else:
                pred_window.append("UNKNOWN")

            label = Counter(pred_window).most_common(1)[0][0] if pred_window else "NO_HAND"
            confidence = pred.confidence if pred is not None and pred.label == label else 0.0

            if label == "NO_HAND":
                no_hand_streak += 1
            else:
                no_hand_streak = 0
            if no_hand_streak >= 4:
                spoken_label = ""

            now = time.time()
            dt = max(now - prev_t, 1e-6)
            fps = 0.90 * fps + 0.10 * (1.0 / dt)
            prev_t = now

            if voice_on and label not in {"NO_HAND", "UNKNOWN"}:
                can_speak = (label != spoken_label) or ((now - last_spoken) >= 7.0)
                if can_speak and (now - last_spoken) >= 1.4:
                    tts.say_latest(speech_for_label(label))
                    spoken_label = label
                    last_spoken = now

            # UI
            h, w = out.shape[:2]
            cv2.rectangle(out, (0, 0), (w, 84), (0, 0, 0), -1)
            cv2.addWeighted(out, 0.80, out, 0.20, 0, out)
            cv2.putText(out, f"Label: {label}", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
            cv2.putText(
                out,
                f"Conf {confidence:.2f} | FPS {fps:.1f} | Voice {'ON' if voice_on else 'OFF'}",
                (18, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (220, 220, 220),
                2,
            )

            # right legend (short)
            cv2.rectangle(out, (w - 300, 100), (w - 12, h - 16), (16, 16, 16), -1)
            cv2.rectangle(out, (w - 300, 100), (w - 12, h - 16), (70, 70, 70), 1)
            cv2.putText(out, "Demo signs (20)", (w - 286, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            y = 150
            for i, name in enumerate(signs_list, start=1):
                cv2.putText(out, f"{i:02d}. {name}", (w - 286, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (220, 220, 220), 1)
                y += 20
                if y > h - 26:
                    break

            cv2.imshow("SignifyAI Lite Demo", out)

            if cv2.getWindowProperty("SignifyAI Lite Demo", cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("v"):
                voice_on = not voice_on
            if key == ord("r"):
                spoken_label = ""
                pred_window.clear()
            last_label = label
    finally:
        tracker.close()
        tts.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("SignifyAI Lite Demo")
    print("Controls: q quit | v voice on/off | r reset speech memory")
    run_lite()
