from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import os
import time
from typing import Optional

import cv2
import numpy as np

from .rules_20 import Rules20
from .speech_engine import SpeechEngine

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@dataclass
class RealtimeConfig:
    camera_index: int = 0
    width: int = 960
    height: int = 540
    camera_fps: int = 30
    voice_enabled: bool = True


def _patch_protobuf_compat() -> None:
    try:
        from google.protobuf import message_factory
        from google.protobuf import symbol_database
    except Exception:
        return

    if not hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
        def _symbol_get_prototype(self, descriptor):
            return message_factory.GetMessageClass(descriptor)
        symbol_database.SymbolDatabase.GetPrototype = _symbol_get_prototype

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):
        def _factory_get_prototype(self, descriptor):
            return message_factory.GetMessageClass(descriptor)
        message_factory.MessageFactory.GetPrototype = _factory_get_prototype


def _open_camera(cfg: RealtimeConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.height))
    cap.set(cv2.CAP_PROP_FPS, int(cfg.camera_fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")
    return cap


def _speech_for_label(label: str) -> str:
    pretty = label.replace("_", " ").title()
    if pretty == "I Love You":
        return "I love you"
    return pretty


def _draw_hud(frame: np.ndarray, label: str, conf: float, fps: float, voice_on: bool) -> None:
    _, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 78), (0, 0, 0), -1)
    cv2.putText(frame, f"Label: {label}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
    cv2.putText(
        frame,
        f"Conf {conf:.2f} | FPS {fps:.1f} | Voice {'ON' if voice_on else 'OFF'}",
        (18, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (210, 210, 210),
        2,
    )


def _draw_landmarks(frame: np.ndarray, raw_hands: list[np.ndarray]) -> None:
    if not raw_hands:
        return
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]
    h, w = frame.shape[:2]
    for hand in raw_hands:
        for a, b in connections:
            xa = int(hand[a, 0] * w)
            ya = int(hand[a, 1] * h)
            xb = int(hand[b, 0] * w)
            yb = int(hand[b, 1] * h)
            cv2.line(frame, (xa, ya), (xb, yb), (130, 230, 130), 2)
        for i in (4, 8, 12, 16, 20):
            x = int(hand[i, 0] * w)
            y = int(hand[i, 1] * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def run_realtime(cfg: RealtimeConfig) -> None:
    _patch_protobuf_compat()
    import mediapipe as mp

    cap = _open_camera(cfg)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.72,
        min_tracking_confidence=0.68,
    )

    rules = Rules20()
    speaker = SpeechEngine(rate=180, volume=1.0)

    pred_window: deque[str] = deque(maxlen=5)
    voice_on = bool(cfg.voice_enabled)
    spoken_label = ""
    last_spoken = 0.0
    prev_time = time.time()
    fps = 0.0
    frame_index = 0
    last_results: Optional[object] = None

    cv2.namedWindow("SignifyAI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SignifyAI", cfg.width, cfg.height)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame_index += 1

            if (frame_index % 2 == 0) or (last_results is None):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                last_results = results
            else:
                results = last_results

            raw_hands: list[np.ndarray] = []
            hand_count = 0
            if getattr(results, "multi_hand_landmarks", None):
                hand_lms = getattr(results, "multi_hand_landmarks")
                hand_count = len(hand_lms)
                for hand_landmarks in hand_lms:
                    arr = np.asarray([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
                    raw_hands.append(arr)

            pred = rules.predict(hand_count, raw_hands)
            if hand_count == 0:
                pred_window.append("NO_HAND")
            elif pred is not None:
                pred_window.append(pred.label)
            else:
                pred_window.append("UNKNOWN")

            label = Counter(pred_window).most_common(1)[0][0] if pred_window else "NO_HAND"
            confidence = pred.confidence if (pred is not None and pred.label == label) else 0.0

            now = time.time()
            dt = max(now - prev_time, 1e-6)
            fps = 0.90 * fps + 0.10 * (1.0 / dt)
            prev_time = now

            if voice_on and label not in {"NO_HAND", "UNKNOWN"}:
                if (label != spoken_label) or ((now - last_spoken) >= 2.0):
                    if (now - last_spoken) >= 0.8:
                        speaker.say_latest(_speech_for_label(label))
                        spoken_label = label
                        last_spoken = now

            _draw_landmarks(frame, raw_hands)
            _draw_hud(frame, label, confidence, fps, voice_on)

            cv2.imshow("SignifyAI", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("v"):
                voice_on = not voice_on
            if key == ord("r"):
                pred_window.clear()
                spoken_label = ""

            if cv2.getWindowProperty("SignifyAI", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        hands.close()
        speaker.close()
        cap.release()
        cv2.destroyAllWindows()
