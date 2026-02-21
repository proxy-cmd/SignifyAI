from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np

from .analytics import append_event
from .config import DEFAULT_LABELS_PATH, DEFAULT_MODEL_PATH, DEFAULT_SESSION_LOG_PATH
from .feature_extraction import normalize_features
from .hand_tracking import HandTracker, check_camera, open_camera, warmup_camera
from .language import sentence_to_text
from .modeling import load_model
from .tts import SpeechEngine


@dataclass
class RealtimeConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    labels_path: Path = DEFAULT_LABELS_PATH
    session_log_path: Path = DEFAULT_SESSION_LOG_PATH
    camera_index: int = 0
    width: int = 960
    height: int = 720
    confidence_threshold: float = 0.60
    smoothing_window: int = 7
    min_stable_frames_for_speech: int = 3


def _draw_confidence_bar(frame, confidence: float) -> None:
    confidence = max(0.0, min(1.0, confidence))
    x, y, w, h = 20, 140, 240, 20
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.rectangle(frame, (x, y), (x + int(w * confidence), y + h), (80, 220, 80), -1)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)


def _draw_help(frame: np.ndarray) -> None:
    help_lines = [
        "q: quit",
        "v: voice on/off",
        "space: add word to sentence",
        "enter: speak sentence",
        "c: clear sentence",
        "p: save screenshot",
        "h: toggle help",
    ]
    x, y = 20, 180
    for line in help_lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        y += 28


def run_realtime(cfg: RealtimeConfig) -> None:
    model, labels = load_model(cfg.model_path, cfg.labels_path)

    cap = open_camera(index=cfg.camera_index, width=cfg.width, height=cfg.height)
    err = check_camera(cap)
    if err:
        raise RuntimeError(err)

    warmup_camera(cap)
    tracker = HandTracker(max_num_hands=2)
    speaker = SpeechEngine(rate=170, volume=1.0)

    window_name = "SignifyAI Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, cfg.width, cfg.height)

    pred_window = deque(maxlen=cfg.smoothing_window)
    spoken_label = ""
    stable_hits = 0
    sentence: list[str] = []
    voice_enabled = True
    show_help = True

    prev_time = time.time()
    fps = 0.0

    print("Controls: q quit | v voice | h help | p screenshot | space add | enter speak sentence | c clear")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            detection = tracker.process(frame, draw=True)
            features = normalize_features(detection.features)

            label = "NO_HAND"
            confidence = 0.0

            if detection.hand_count > 0:
                probs = model.predict_proba([features])[0]
                best_idx = int(np.argmax(probs))
                candidate = str(model.classes_[best_idx])
                confidence = float(probs[best_idx])
                if confidence >= cfg.confidence_threshold:
                    pred_window.append(candidate)
                else:
                    pred_window.append("UNKNOWN")
            else:
                pred_window.append("NO_HAND")

            if pred_window:
                label = Counter(pred_window).most_common(1)[0][0]

            # Speak when stable label changes to a meaningful class.
            if label == spoken_label:
                stable_hits += 1
            else:
                stable_hits = 1

            if (
                voice_enabled
                and label not in {"NO_HAND", "UNKNOWN"}
                and label != spoken_label
                and stable_hits >= cfg.min_stable_frames_for_speech
            ):
                speaker.say(label)
                append_event(cfg.session_log_path, label=label, confidence=confidence, hand_count=detection.hand_count)
                spoken_label = label

            # FPS estimate.
            now = time.time()
            dt = max(now - prev_time, 1e-6)
            fps = 0.92 * fps + 0.08 * (1.0 / dt)
            prev_time = now

            out = detection.frame
            cv2.putText(out, f"Label: {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(out, f"Hands: {detection.hand_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(out, f"FPS: {fps:.1f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 255), 2)
            cv2.putText(out, f"Voice: {'ON' if voice_enabled else 'OFF'}", (280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            cv2.putText(out, f"Known labels: {len(labels)}", (280, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

            sentence_text = sentence_to_text(sentence[-8:])
            cv2.putText(out, f"Sentence: {sentence_text}", (20, out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)

            _draw_confidence_bar(out, confidence)
            if show_help:
                _draw_help(out)
            cv2.imshow(window_name, out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("v"):
                voice_enabled = not voice_enabled
            if key == ord("h"):
                show_help = not show_help
            if key == ord("c"):
                sentence.clear()
            if key == 32 and label not in {"NO_HAND", "UNKNOWN"}:  # space
                sentence.append(label)
            if key == 13 and sentence:
                speaker.say(sentence_to_text(sentence))
            if key == ord("p"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                shots_dir = cfg.session_log_path.parent / "screenshots"
                shots_dir.mkdir(parents=True, exist_ok=True)
                shot_path = shots_dir / f"frame_{ts}.png"
                cv2.imwrite(str(shot_path), out)
                print(f"Saved screenshot: {shot_path}")

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        tracker.close()
        speaker.close()
        cap.release()
        cv2.destroyAllWindows()
