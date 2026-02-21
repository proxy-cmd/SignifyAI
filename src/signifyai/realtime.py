from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Optional

import cv2
import numpy as np

from .analytics import append_event
from .config import DEFAULT_LABELS_PATH, DEFAULT_MODEL_PATH, DEFAULT_SESSION_LOG_PATH
from .feature_extraction import normalize_features
from .hand_tracking import HandTracker, check_camera, open_camera, warmup_camera
from .language import sentence_to_text
from .modeling import load_model
from .rules import RuleBasedInterpreter
from .tts import SpeechEngine


@dataclass
class RealtimeConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    labels_path: Path = DEFAULT_LABELS_PATH
    session_log_path: Path = DEFAULT_SESSION_LOG_PATH
    camera_index: int = 0
    width: int = 960
    height: int = 720
    confidence_threshold: float = 0.62
    smoothing_window: int = 7
    min_stable_frames_for_speech: int = 3
    mode: str = "hybrid"  # rules | ml | hybrid
    rule_confidence_threshold: float = 0.78
    inference_interval: int = 1
    inference_scale: float = 0.75
    adaptive_performance: bool = True
    target_fps: float = 20.0
    repeat_same_label_sec: float = 8.0
    speak_cooldown_sec: float = 1.6
    per_label_cooldown_sec: float = 2.6
    show_sentence: bool = False
    stage_mode: bool = True
    label_hold_sec: float = 0.28
    demo_script: bool = False


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
        "m: switch mode (rules/hybrid/ml)",
        "s: show/hide sentence bar",
        "tab: stage/dev UI",
        "f: fullscreen",
        "space: add word to sentence",
        "enter: speak sentence",
        "c: clear sentence",
        "p: save screenshot",
        "n/r: demo next/reset",
        "h: toggle help",
    ]
    x, y = 20, 180
    for line in help_lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        y += 28


def _draw_compact_hud(
    frame: np.ndarray,
    label: str,
    hands: int,
    fps: float,
    confidence: float,
    mode_text: str,
    voice_enabled: bool,
    sentence_text: str,
    perf_text: str,
) -> None:
    h, w = frame.shape[:2]

    # Top-left compact card
    card_w = min(430, w - 20)
    cv2.rectangle(frame, (10, 10), (10 + card_w, 124), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (10 + card_w, 124), (70, 70, 70), 1)
    cv2.putText(frame, f"Label: {label}", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2)
    cv2.putText(frame, f"Hands: {hands}    FPS: {fps:.1f}", (22, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 220, 255), 2)
    cv2.putText(
        frame,
        f"Mode: {mode_text} | {perf_text} | Voice: {'ON' if voice_enabled else 'OFF'}",
        (22, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (190, 255, 190),
        2,
    )

    # Bottom sentence strip (optional)
    if sentence_text:
        cv2.rectangle(frame, (10, h - 52), (w - 10, h - 10), (18, 18, 18), -1)
        cv2.rectangle(frame, (10, h - 52), (w - 10, h - 10), (70, 70, 70), 1)
        cv2.putText(
            frame,
            f"Sentence: {sentence_text}",
            (22, h - 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (235, 235, 235),
            2,
        )

    _draw_confidence_bar(frame, confidence)


def _draw_stage_hud(
    frame: np.ndarray,
    label: str,
    confidence: float,
    fps: float,
    voice_enabled: bool,
    perf_text: str,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    center_text = label if label not in {"NO_HAND", "UNKNOWN"} else ("NO HAND" if label == "NO_HAND" else "...")
    color = (255, 255, 0) if label not in {"NO_HAND", "UNKNOWN"} else (220, 220, 220)
    scale = 2.0 if len(center_text) <= 9 else 1.45
    thick = 4 if scale > 1.8 else 3
    (tw, th), _ = cv2.getTextSize(center_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    tx = max(20, (w - tw) // 2)
    ty = max(150, (h + th) // 2)
    cv2.putText(frame, center_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    cv2.putText(
        frame,
        f"Conf {confidence:.2f} | FPS {fps:.1f} | {perf_text} | Voice {'ON' if voice_enabled else 'OFF'}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (235, 235, 235),
        2,
    )


def _draw_demo_prompt(frame: np.ndarray, prompt: str, progress: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 92), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.9, frame, 0.1, 0, frame)
    cv2.putText(frame, f"Demo Prompt: {prompt}", (18, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, progress, (18, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)


def _draw_cached_points(frame: np.ndarray, raw_hands: list[np.ndarray]) -> None:
    if not raw_hands:
        return
    h, w = frame.shape[:2]
    for hand in raw_hands:
        for i in range(hand.shape[0]):
            x = int(hand[i, 0] * w)
            y = int(hand[i, 1] * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)


def run_realtime(cfg: RealtimeConfig) -> None:
    model = None
    labels: list[str] = []
    mode = cfg.mode.lower().strip()
    if mode not in {"rules", "ml", "hybrid"}:
        mode = "hybrid"

    if mode in {"ml", "hybrid"}:
        try:
            model, labels = load_model(cfg.model_path, cfg.labels_path)
        except Exception as ex:
            # Keep console clean; fallback silently unless explicitly in ml mode.
            if mode == "ml":
                print(f"[INFO] ML model unavailable: {ex}")
                print("[INFO] Falling back to rules mode.")
            mode = "rules"

    cap = open_camera(index=cfg.camera_index, width=cfg.width, height=cfg.height)
    err = check_camera(cap)
    if err:
        raise RuntimeError(err)

    warmup_camera(cap)
    tracker = HandTracker(max_num_hands=2, inference_scale=cfg.inference_scale)
    rules = RuleBasedInterpreter()
    speaker = SpeechEngine(rate=170, volume=1.0)

    window_name = "SignifyAI Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, cfg.width, cfg.height)

    pred_window = deque(maxlen=cfg.smoothing_window)
    spoken_label = ""
    last_frame_label = "NO_HAND"
    stable_hits = 0
    no_hand_streak = 0
    pending_label = "NO_HAND"
    pending_since = time.time()
    accepted_label = "NO_HAND"
    sentence: list[str] = []
    voice_enabled = True
    show_help = False
    last_spoken_time = 0.0
    show_sentence = cfg.show_sentence
    stage_mode = cfg.stage_mode
    is_fullscreen = False

    prev_time = time.time()
    fps = 0.0

    print("Controls: q quit | v voice | m mode | h help | s sentence | p screenshot | space add | enter speak sentence | c clear")
    print("UI: TAB stage/dev | f fullscreen")
    print(f"Prediction mode: {mode.upper()}")
    print(f"Performance: interval={cfg.inference_interval}, scale={cfg.inference_scale}")
    if cfg.demo_script:
        print("Demo Script: ON (n: next prompt, r: reset)")

    # Startup countdown (camera + TTS warmup time).
    countdown_start = time.time()
    abort_start = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        left = 3 - int(time.time() - countdown_start)
        if left <= 0:
            break
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        cv2.putText(frame, "Starting...", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
        cv2.putText(frame, str(left), (frame.shape[1] // 2 - 20, frame.shape[0] // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (255, 255, 0), 5)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            abort_start = True
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            abort_start = True
            break

    if abort_start:
        tracker.close()
        speaker.close()
        cap.release()
        cv2.destroyAllWindows()
        return

    infer_every = max(1, int(cfg.inference_interval))
    perf_target = max(8.0, float(cfg.target_fps))
    adaptive_perf = bool(cfg.adaptive_performance)
    last_tune_ts = time.time()
    frame_idx = 0
    last_detection = None
    last_label = "NO_HAND"
    last_confidence = 0.0
    last_source = "NONE"
    spoken_counter: Counter[str] = Counter()
    last_spoken_by_label: dict[str, float] = {}
    demo_steps = [
        "HELLO",
        "YES",
        "NO",
        "ONE",
        "TWO",
        "PEACE",
        "OKAY",
        "CALL ME",
        "I LOVE YOU",
        "THANK YOU",
    ]
    demo_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_idx += 1

            run_inference = (frame_idx % infer_every == 0) or (last_detection is None)
            if run_inference:
                detection = tracker.process(frame, draw=True)
                last_detection = detection
            else:
                # Reuse last inference result but keep current frame for smooth display.
                detection = last_detection
                detection = type(detection)(
                    features=detection.features,
                    hand_count=detection.hand_count,
                    frame=frame.copy(),
                    raw_hands=detection.raw_hands,
                    handedness=detection.handedness,
                )
                _draw_cached_points(detection.frame, detection.raw_hands)

            features = normalize_features(detection.features)

            label = "NO_HAND"
            confidence = 0.0
            source = "NONE"

            rule_label: Optional[str] = None
            rule_conf = 0.0
            if mode in {"rules", "hybrid"}:
                rule_pred = rules.predict(detection)
                if rule_pred is not None:
                    rule_label = rule_pred.label
                    rule_conf = rule_pred.confidence

            ml_label: Optional[str] = None
            ml_conf = 0.0
            if mode in {"ml", "hybrid"} and model is not None and detection.hand_count > 0:
                probs = model.predict_proba([features])[0]
                best_idx = int(np.argmax(probs))
                ml_label = str(model.classes_[best_idx])
                ml_conf = float(probs[best_idx])

            if mode == "rules":
                if detection.hand_count == 0:
                    pred_window.append("NO_HAND")
                elif rule_label is not None and rule_conf >= cfg.rule_confidence_threshold:
                    pred_window.append(rule_label)
                    confidence = rule_conf
                    source = "RULE"
                else:
                    pred_window.append("UNKNOWN")
                    confidence = rule_conf
                    source = "RULE"

            elif mode == "ml":
                if detection.hand_count > 0 and ml_label is not None:
                    if ml_conf >= cfg.confidence_threshold:
                        pred_window.append(ml_label)
                        confidence = ml_conf
                        source = "ML"
                    else:
                        pred_window.append("UNKNOWN")
                        confidence = ml_conf
                        source = "ML"
                else:
                    pred_window.append("NO_HAND")

            else:  # hybrid
                if detection.hand_count == 0:
                    pred_window.append("NO_HAND")
                    source = "NONE"
                elif rule_label is not None and rule_conf >= cfg.rule_confidence_threshold:
                    pred_window.append(rule_label)
                    confidence = rule_conf
                    source = "RULE"
                elif ml_label is not None:
                    if ml_conf >= cfg.confidence_threshold:
                        pred_window.append(ml_label)
                    else:
                        pred_window.append("UNKNOWN")
                    confidence = ml_conf
                    source = "ML"
                else:
                    pred_window.append("UNKNOWN")
                    source = "NONE"

            if pred_window:
                label = Counter(pred_window).most_common(1)[0][0]

            # Temporal debouncing: a new label must persist for a short hold time.
            now_event = time.time()
            if label != pending_label:
                pending_label = label
                pending_since = now_event
            hold_ok = (now_event - pending_since) >= cfg.label_hold_sec
            if hold_ok:
                accepted_label = pending_label
            label = accepted_label

            # Speak when stable label changes to a meaningful class.
            if label == last_frame_label:
                stable_hits += 1
            else:
                stable_hits = 1
                last_frame_label = label

            if label == "NO_HAND":
                no_hand_streak += 1
            else:
                no_hand_streak = 0

            # Retrigger same phrase after hand goes away for a while.
            if no_hand_streak >= 4:
                spoken_label = ""

            now_speak = time.time()
            can_repeat_same = (label == spoken_label) and ((now_speak - last_spoken_time) >= cfg.repeat_same_label_sec)
            if (
                voice_enabled
                and label not in {"NO_HAND", "UNKNOWN"}
                and stable_hits >= cfg.min_stable_frames_for_speech
                and (now_speak - last_spoken_time) >= cfg.speak_cooldown_sec
                and ((now_speak - last_spoken_by_label.get(label, 0.0)) >= cfg.per_label_cooldown_sec)
                and (label != spoken_label or can_repeat_same)
            ):
                # Avoid queued old labels causing delayed speaking.
                speaker.say_latest(label)
                append_event(cfg.session_log_path, label=label, confidence=confidence, hand_count=detection.hand_count)
                spoken_label = label
                last_spoken_time = now_speak
                last_spoken_by_label[label] = now_speak
                spoken_counter[label] += 1
                if cfg.demo_script and demo_index < len(demo_steps) and label == demo_steps[demo_index]:
                    demo_index += 1

            last_label = label
            last_confidence = confidence
            last_source = source

            # FPS estimate.
            now = time.time()
            dt = max(now - prev_time, 1e-6)
            fps = 0.92 * fps + 0.08 * (1.0 / dt)
            prev_time = now

            # Adaptive performance controller for older PCs.
            if adaptive_perf and (now - last_tune_ts) > 1.0:
                if fps < (perf_target - 3.0) and infer_every < 4:
                    infer_every += 1
                elif fps > (perf_target + 4.0) and infer_every > 1:
                    infer_every -= 1
                last_tune_ts = now

            sentence_text = sentence_to_text(sentence[-8:]) if show_sentence else ""
            perf_text = f"intv {infer_every}"
            out = detection.frame
            if stage_mode:
                _draw_stage_hud(
                    out,
                    label=last_label,
                    confidence=last_confidence,
                    fps=fps,
                    voice_enabled=voice_enabled,
                    perf_text=perf_text,
                )
            else:
                _draw_compact_hud(
                    out,
                    label=last_label,
                    hands=detection.hand_count,
                    fps=fps,
                    confidence=last_confidence,
                    mode_text=f"{mode.upper()} {last_source}",
                    voice_enabled=voice_enabled,
                    sentence_text=sentence_text,
                    perf_text=perf_text,
                )
            if show_help and not stage_mode:
                _draw_help(out)
            if cfg.demo_script:
                if demo_index < len(demo_steps):
                    prompt = demo_steps[demo_index]
                    progress = f"Step {demo_index + 1}/{len(demo_steps)}  (show this sign)"
                else:
                    prompt = "DONE"
                    progress = f"Completed {len(demo_steps)}/{len(demo_steps)}"
                _draw_demo_prompt(out, prompt, progress)
            cv2.imshow(window_name, out)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKeyEx(1)
            if key == -1:
                # Ensure clicking window close (X) exits immediately.
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue

            ch = ""
            low = key & 0xFF
            if 0 <= low <= 255:
                ch = chr(low).lower()

            if key == 27 or ch == "q":
                break
            if ch == "v":
                voice_enabled = not voice_enabled
            if ch == "m":
                order = ["rules", "hybrid", "ml"]
                idx = order.index(mode) if mode in order else 0
                tried = 0
                while tried < len(order):
                    idx = (idx + 1) % len(order)
                    next_mode = order[idx]
                    tried += 1
                    if next_mode in {"ml", "hybrid"} and model is None:
                        continue
                    mode = next_mode
                    pred_window.clear()
                    pending_label = "NO_HAND"
                    accepted_label = "NO_HAND"
                    last_frame_label = "NO_HAND"
                    stable_hits = 0
                    print(f"[INFO] Switched mode: {mode.upper()}")
                    break
            if ch == "h":
                show_help = not show_help
            if key == 9:  # TAB
                stage_mode = not stage_mode
            if ch == "s":
                show_sentence = not show_sentence
            if ch == "f":
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            if ch == "c":
                sentence.clear()
            if key == 32 and label not in {"NO_HAND", "UNKNOWN"}:  # space
                sentence.append(label)
            if key == 13 and sentence:
                speaker.say(sentence_to_text(sentence))
            if ch == "p":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                shots_dir = cfg.session_log_path.parent / "screenshots"
                shots_dir.mkdir(parents=True, exist_ok=True)
                shot_path = shots_dir / f"frame_{ts}.png"
                cv2.imwrite(str(shot_path), out)
                print(f"Saved screenshot: {shot_path}")
            if ch == "n" and cfg.demo_script:
                demo_index = min(demo_index + 1, len(demo_steps))
            if ch == "r" and cfg.demo_script:
                demo_index = 0

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Save quick session summary for post-demo evidence.
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "spoken_counts": dict(spoken_counter),
            "demo_script": cfg.demo_script,
            "demo_progress": f"{demo_index}/{len(demo_steps)}",
        }
        summary_path = cfg.session_log_path.parent / "session_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Session summary saved: {summary_path}")

        tracker.close()
        speaker.close()
        cap.release()
        cv2.destroyAllWindows()
