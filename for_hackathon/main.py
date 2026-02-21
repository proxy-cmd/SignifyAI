from __future__ import annotations

from collections import Counter, deque
import time

import cv2
import mediapipe as mp
import numpy as np

from rules_20 import Rules20
from speech_engine import SpeechEngine


def speech_for_label(label: str) -> str:
    pretty = label.replace("_", " ").title()
    if pretty.upper().startswith("GOOD "):
        return pretty
    if pretty == "I Love You":
        return "I love you"
    return pretty


def compute_hint(frame: np.ndarray, hand_count: int, confidence: float, label: str) -> tuple[str, tuple[int, int, int]]:
    brightness = float(frame.mean())
    if brightness < 48:
        return "Low light: increase lighting", (0, 180, 255)
    if hand_count == 0:
        return "Show hand in frame", (200, 220, 255)
    if label == "UNKNOWN" or confidence < 0.55:
        return "Hold hand steady", (0, 220, 255)
    return "Tracking good", (90, 240, 120)


def draw_hint(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    x1 = max(10, w - tw - 28)
    y1 = 10
    x2 = w - 10
    y2 = 42
    cv2.rectangle(frame, (x1, y1), (x2, y2), (24, 24, 24), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
    cv2.putText(frame, text, (x1 + 8, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)


def main() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.72,
        min_tracking_confidence=0.68,
    )

    rules = Rules20()
    tts = SpeechEngine(rate=170, volume=1.0)

    for _ in range(12):
        cap.read()

    cv2.namedWindow("Hackathon Sign Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hackathon Sign Demo", 1100, 760)

    pred_window: deque[str] = deque(maxlen=7)
    voice_on = True
    spoken_label = ""
    no_hand_streak = 0
    last_spoken = 0.0
    prev_t = time.time()
    fps = 0.0

    signs = [
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            raw_hands: list[np.ndarray] = []
            hand_count = 0
            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                for lm in results.multi_hand_landmarks:
                    arr = np.asarray([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                    raw_hands.append(arr)
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            pred = rules.predict(hand_count, raw_hands)
            if hand_count == 0:
                pred_window.append("NO_HAND")
            elif pred is not None:
                pred_window.append(pred.label)
            else:
                pred_window.append("UNKNOWN")

            label = Counter(pred_window).most_common(1)[0][0] if pred_window else "NO_HAND"
            confidence = pred.confidence if pred is not None and pred.label == label else 0.0

            now = time.time()
            dt = max(now - prev_t, 1e-6)
            fps = 0.90 * fps + 0.10 * (1.0 / dt)
            prev_t = now

            if label == "NO_HAND":
                no_hand_streak += 1
            else:
                no_hand_streak = 0
            if no_hand_streak >= 4:
                spoken_label = ""

            if voice_on and label not in {"NO_HAND", "UNKNOWN"}:
                allow_repeat = (label != spoken_label) or ((now - last_spoken) >= 7.0)
                if allow_repeat and (now - last_spoken) >= 1.4:
                    tts.say_latest(speech_for_label(label))
                    spoken_label = label
                    last_spoken = now

            # UI
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 84), (0, 0, 0), -1)
            cv2.putText(frame, f"Label: {label}", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
            cv2.putText(
                frame,
                f"Conf {confidence:.2f} | FPS {fps:.1f} | Voice {'ON' if voice_on else 'OFF'}",
                (18, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (220, 220, 220),
                2,
            )

            draw_hint(frame, *compute_hint(frame, hand_count, confidence, label))

            cv2.rectangle(frame, (w - 300, 100), (w - 12, h - 16), (16, 16, 16), -1)
            cv2.rectangle(frame, (w - 300, 100), (w - 12, h - 16), (70, 70, 70), 1)
            cv2.putText(frame, "Demo signs (20)", (w - 286, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            y = 150
            for i, s in enumerate(signs, start=1):
                cv2.putText(frame, f"{i:02d}. {s}", (w - 286, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (220, 220, 220), 1)
                y += 20
                if y > h - 26:
                    break

            cv2.imshow("Hackathon Sign Demo", frame)

            if cv2.getWindowProperty("Hackathon Sign Demo", cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("v"):
                voice_on = not voice_on
            if key == ord("r"):
                spoken_label = ""
                pred_window.clear()
    finally:
        hands.close()
        tts.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hackathon Sign Demo")
    print("Controls: q quit | v voice on/off | r reset speech memory")
    main()

