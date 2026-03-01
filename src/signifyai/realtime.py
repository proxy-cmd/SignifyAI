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
from .config import (
    DEFAULT_LABELS_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROTOTYPE_DB_PATH,
    DEFAULT_SESSION_LOG_PATH,
    DEFAULT_TEMPORAL_LABELS_PATH,
    DEFAULT_TEMPORAL_METADATA_PATH,
    DEFAULT_TEMPORAL_MODEL_PATH,
)
from .feature_extraction import normalize_features
from .hand_tracking import HandTracker, check_camera, open_camera, warmup_camera
from .language import sentence_to_text, speech_text_for_label
from .modeling import load_model
from .rules import RuleBasedInterpreter
from .temporal_model import load_temporal_model
from .prototype_adapt import load_prototype_db, predict_prototype
from .sentence_runtime import can_auto_append_token, should_auto_speak_sentence
from .tts import SpeechEngine


@dataclass
class RealtimeConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    labels_path: Path = DEFAULT_LABELS_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    session_log_path: Path = DEFAULT_SESSION_LOG_PATH
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    camera_fps: int = 60
    confidence_threshold: float = 0.62
    smoothing_window: int = 7
    min_stable_frames_for_speech: int = 3
    mode: str = "hybrid"  # rules | ml | temporal | hybrid
    rule_confidence_threshold: float = 0.78
    inference_interval: int = 1
    inference_scale: float = 0.75
    model_complexity: int = 0
    landmark_smoothing: float = 0.78
    adaptive_performance: bool = True
    target_fps: float = 24.0
    repeat_same_label_sec: float = 8.0
    speak_cooldown_sec: float = 1.6
    per_label_cooldown_sec: float = 2.6
    auto_speak: bool = True
    show_sentence: bool = False
    stage_mode: bool = True
    label_hold_sec: float = 0.28
    demo_script: bool = False
    temporal_model_path: Path = DEFAULT_TEMPORAL_MODEL_PATH
    temporal_labels_path: Path = DEFAULT_TEMPORAL_LABELS_PATH
    temporal_metadata_path: Path = DEFAULT_TEMPORAL_METADATA_PATH
    temporal_confidence_threshold: float = 0.60
    enhance_frame: bool = True
    quality_gate: bool = True
    min_brightness: float = 45.0
    min_blur_var: float = 55.0
    min_hand_area: float = 0.012
    strict_consensus: bool = False
    strict_override_conf: float = 0.92
    ml_min_margin: float = 0.08
    use_prototypes: bool = True
    prototype_db_path: Path = DEFAULT_PROTOTYPE_DB_PATH
    prototype_threshold: float = 0.84
    prototype_margin: float = 0.03
    continuous_sentence: bool = False
    sentence_pause_speak_sec: float = 2.5
    sentence_append_cooldown_sec: float = 1.1
    sentence_max_tokens: int = 10


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
        "a: auto-speak on/off",
        "t: continuous sentence on/off",
        "m: switch mode (rules/hybrid/ml/temporal)",
        "k: start/stop recording",
        "s: show/hide sentence bar",
        "tab: stage/dev UI",
        "f: fullscreen",
        "space: add word to sentence",
        "enter: speak sentence",
        "c: clear sentence",
        "p: save screenshot",
        "n/r: demo next/reset",
        "h: toggle help",
        "OKAY: touch thumb tip + index tip",
        "YES: thumbs up (others folded)",
        "NO: thumbs down (others folded)",
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
    auto_speak: bool,
    continuous_sentence: bool,
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
        f"Mode: {mode_text} | {perf_text} | Voice: {'ON' if voice_enabled else 'OFF'} | Auto: {'ON' if auto_speak else 'OFF'} | Cont: {'ON' if continuous_sentence else 'OFF'}",
        (22, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
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
    auto_speak: bool,
    continuous_sentence: bool,
    perf_text: str,
    recording: bool,
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
        f"Conf {confidence:.2f} | FPS {fps:.1f} | {perf_text} | Voice {'ON' if voice_enabled else 'OFF'} | Auto {'ON' if auto_speak else 'OFF'} | Cont {'ON' if continuous_sentence else 'OFF'}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        2,
    )
    if recording:
        cv2.putText(frame, "REC", (w - 90, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)


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


def _compute_quality_hint(frame: np.ndarray, hand_count: int, confidence: float, label: str) -> tuple[str, tuple[int, int, int]]:
    brightness, blur_metric = _frame_metrics(frame)
    if blur_metric < 70:
        return "Image blurry: hold steady / clean lens", (0, 180, 255)
    if brightness < 48:
        return "Low light: increase lighting", (0, 180, 255)
    if hand_count == 0:
        return "Show hand in frame", (180, 220, 255)
    if label == "UNKNOWN" or confidence < 0.55:
        return "Hold steady for better recognition", (0, 220, 255)
    return "Tracking good", (90, 240, 120)


def _draw_quality_hint(frame: np.ndarray, hint: str, color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    x1 = max(10, w - tw - 28)
    y1 = 10
    x2 = w - 10
    y2 = 42
    cv2.rectangle(frame, (x1, y1), (x2, y2), (24, 24, 24), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
    cv2.putText(frame, hint, (x1 + 8, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)


def _enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Lightweight clarity boost for webcam feed."""
    tuned = cv2.convertScaleAbs(frame, alpha=1.05, beta=4)
    blur = cv2.GaussianBlur(tuned, (0, 0), 1.1)
    sharp = cv2.addWeighted(tuned, 1.20, blur, -0.20, 0)
    return sharp


def _frame_metrics(frame: np.ndarray) -> tuple[float, float]:
    brightness = float(frame.mean())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_metric = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, blur_metric


def _max_hand_area(raw_hands: list[np.ndarray]) -> float:
    if not raw_hands:
        return 0.0
    areas = []
    for hand in raw_hands:
        xs = hand[:, 0]
        ys = hand[:, 1]
        w = float(xs.max() - xs.min())
        h = float(ys.max() - ys.min())
        areas.append(w * h)
    return float(max(areas)) if areas else 0.0


def _strict_consensus_decision(
    rule_label: Optional[str],
    rule_conf: float,
    proto_label: Optional[str],
    proto_conf: float,
    ml_label: Optional[str],
    ml_conf: float,
    temporal_label: Optional[str],
    temporal_conf: float,
    override_conf: float,
) -> Optional[tuple[str, float, str]]:
    """
    Require agreement across at least two sources.
    If no agreement, allow a single source only when confidence is very high.
    """
    candidates: list[tuple[str, str, float]] = []
    if rule_label:
        candidates.append(("RULE", rule_label, rule_conf))
    if proto_label:
        candidates.append(("PROTO", proto_label, proto_conf))
    if ml_label:
        candidates.append(("ML", ml_label, ml_conf))
    if temporal_label:
        candidates.append(("TEMP", temporal_label, temporal_conf))

    if len(candidates) < 2:
        return None

    votes: dict[str, list[tuple[str, float]]] = {}
    for src, lbl, conf in candidates:
        votes.setdefault(lbl, []).append((src, conf))

    # Label with most agreeing sources wins. Confidence breaks ties.
    best_label = None
    best_votes = -1
    best_conf = -1.0
    for lbl, entries in votes.items():
        vote_count = len(entries)
        lbl_conf = max(c for _, c in entries)
        if vote_count > best_votes or (vote_count == best_votes and lbl_conf > best_conf):
            best_label = lbl
            best_votes = vote_count
            best_conf = lbl_conf

    if best_label is not None and best_votes >= 2:
        return best_label, best_conf, "CONSENSUS"

    src, lbl, conf = max(candidates, key=lambda x: x[2])
    if conf >= override_conf:
        return lbl, conf, f"{src}_OVERRIDE"
    return "UNKNOWN", conf, "CONSENSUS"


def run_realtime(cfg: RealtimeConfig) -> None:
    model = None
    labels: list[str] = []
    ml_label_thresholds: dict[str, float] = {}
    temporal_model = None
    temporal_labels: list[str] = []
    temporal_seq_len = 24
    prototype_db = None
    mode = cfg.mode.lower().strip()
    if mode not in {"rules", "ml", "temporal", "hybrid"}:
        mode = "hybrid"

    if mode in {"ml", "hybrid"}:
        try:
            model, labels = load_model(cfg.model_path, cfg.labels_path)
            if cfg.metadata_path.exists():
                try:
                    meta = json.loads(cfg.metadata_path.read_text(encoding="utf-8"))
                    raw = meta.get("label_thresholds", {})
                    if isinstance(raw, dict):
                        for k, v in raw.items():
                            try:
                                ml_label_thresholds[str(k)] = float(v)
                            except Exception:
                                continue
                except Exception:
                    ml_label_thresholds = {}
        except Exception as ex:
            # Keep console clean; fallback silently unless explicitly in ml mode.
            if mode == "ml":
                print(f"[INFO] ML model unavailable: {ex}")
                print("[INFO] Falling back to rules mode.")
                mode = "rules"

    if mode in {"temporal", "hybrid"}:
        try:
            temporal_model, temporal_labels, temporal_seq_len = load_temporal_model(
                cfg.temporal_model_path,
                cfg.temporal_labels_path,
                cfg.temporal_metadata_path,
            )
        except Exception as ex:
            if mode == "temporal":
                print(f"[INFO] Temporal model unavailable: {ex}")
                print("[INFO] Falling back to rules mode.")
                mode = "rules"

    if cfg.use_prototypes:
        try:
            prototype_db = load_prototype_db(cfg.prototype_db_path)
            if prototype_db.vectors.shape[0] > 0:
                print(f"[INFO] Loaded prototypes: {prototype_db.vectors.shape[0]} samples from {cfg.prototype_db_path}")
        except Exception as ex:
            print(f"[WARN] Prototype DB unavailable: {ex}")
            prototype_db = None

    cap = open_camera(index=cfg.camera_index, width=cfg.width, height=cfg.height, fps=cfg.camera_fps)
    err = check_camera(cap)
    if err:
        raise RuntimeError(err)

    warmup_camera(cap)
    tracker = HandTracker(
        max_num_hands=2,
        inference_scale=cfg.inference_scale,
        model_complexity=cfg.model_complexity,
        landmark_smoothing=cfg.landmark_smoothing,
    )
    rules = RuleBasedInterpreter()
    speaker = SpeechEngine(rate=170, volume=1.0)

    window_name = "SignifyAI Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, cfg.width, cfg.height)

    pred_window = deque(maxlen=cfg.smoothing_window)
    seq_buffer = deque(maxlen=max(4, int(temporal_seq_len)))
    spoken_label = ""
    last_frame_label = "NO_HAND"
    stable_hits = 0
    no_hand_streak = 0
    pending_label = "NO_HAND"
    pending_since = time.time()
    accepted_label = "NO_HAND"
    sentence: list[str] = []
    voice_enabled = True
    auto_speak = cfg.auto_speak
    continuous_sentence = cfg.continuous_sentence
    show_help = False
    last_spoken_time = 0.0
    show_sentence = cfg.show_sentence
    stage_mode = cfg.stage_mode
    is_fullscreen = False

    prev_time = time.time()
    fps = 0.0

    print("Controls: q quit | v voice | a auto-speak | m mode | h help | s sentence | p screenshot | space add | enter speak sentence | c clear")
    print("Continuous sentence: t toggle")
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
        if cfg.enhance_frame:
            frame = _enhance_frame(frame)
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
    last_sentence_append_time = 0.0
    last_sentence_token_time = 0.0
    last_sentence_spoken_time = 0.0
    last_spoken_sentence = ""
    demo_steps = [
        "HELLO",
        "YES",
        "NO",
        "ONE",
        "TWO",
        "PEACE",
        "STOP",
        "CALL ME",
        "I LOVE YOU",
        "THANK YOU",
    ]
    demo_index = 0
    recording = False
    video_writer = None
    video_path: Optional[Path] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            if cfg.enhance_frame:
                frame = _enhance_frame(frame)
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
                if not stage_mode:
                    _draw_cached_points(detection.frame, detection.raw_hands)

            features = normalize_features(detection.features)
            if detection.hand_count > 0:
                seq_buffer.append(features.astype(np.float32))
            else:
                seq_buffer.clear()

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
            ml_margin = 0.0
            if mode in {"ml", "hybrid"} and model is not None and detection.hand_count > 0:
                probs = model.predict_proba([features])[0]
                top_idx = np.argsort(probs)[::-1]
                best_idx = int(top_idx[0])
                second_idx = int(top_idx[1]) if len(top_idx) > 1 else best_idx
                ml_label = str(model.classes_[best_idx])
                ml_conf = float(probs[best_idx])
                ml_margin = float(probs[best_idx] - probs[second_idx]) if len(top_idx) > 1 else 1.0

            proto_label: Optional[str] = None
            proto_conf = 0.0
            if detection.hand_count > 0 and prototype_db is not None and prototype_db.vectors.shape[0] > 0:
                pm = predict_prototype(
                    features=features,
                    db=prototype_db,
                    min_similarity=cfg.prototype_threshold,
                    min_margin=cfg.prototype_margin,
                )
                if pm is not None:
                    proto_label = pm.label
                    proto_conf = pm.similarity

            temporal_label: Optional[str] = None
            temporal_conf = 0.0
            if (
                mode in {"temporal", "hybrid"}
                and temporal_model is not None
                and detection.hand_count > 0
                and len(seq_buffer) >= temporal_seq_len
            ):
                seq = np.asarray(list(seq_buffer)[-temporal_seq_len:], dtype=np.float32).reshape(1, -1)
                probs_t = temporal_model.predict_proba(seq)[0]
                best_t = int(np.argmax(probs_t))
                classes_t = list(getattr(temporal_model, "classes_", temporal_labels))
                temporal_label = str(classes_t[best_t]) if classes_t else None
                temporal_conf = float(probs_t[best_t])

            quality_ok = True
            if cfg.quality_gate and detection.hand_count > 0:
                brightness, blur_metric = _frame_metrics(frame)
                hand_area = _max_hand_area(detection.raw_hands)
                quality_ok = (
                    brightness >= cfg.min_brightness
                    and blur_metric >= cfg.min_blur_var
                    and hand_area >= cfg.min_hand_area
                )

            if mode == "rules":
                if detection.hand_count == 0:
                    pred_window.append("NO_HAND")
                elif not quality_ok:
                    pred_window.append("UNKNOWN")
                    source = "QGATE"
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
                    ml_threshold = max(
                        cfg.confidence_threshold,
                        float(ml_label_thresholds.get(ml_label, cfg.confidence_threshold)),
                    )
                    if not quality_ok:
                        pred_window.append("UNKNOWN")
                        confidence = ml_conf
                        source = "QGATE"
                    elif proto_label is not None and proto_conf >= max(cfg.prototype_threshold, ml_conf + 0.02):
                        pred_window.append(proto_label)
                        confidence = proto_conf
                        source = "PROTO"
                    elif ml_conf >= ml_threshold and ml_margin >= cfg.ml_min_margin:
                        pred_window.append(ml_label)
                        confidence = ml_conf
                        source = "ML"
                    else:
                        pred_window.append("UNKNOWN")
                        confidence = ml_conf
                        source = "ML"
                elif detection.hand_count > 0 and proto_label is not None and proto_conf >= cfg.prototype_threshold:
                    pred_window.append(proto_label)
                    confidence = proto_conf
                    source = "PROTO"
                elif detection.hand_count > 0:
                    pred_window.append("UNKNOWN")
                    source = "ML"
                else:
                    pred_window.append("NO_HAND")

            elif mode == "temporal":
                if detection.hand_count == 0:
                    pred_window.append("NO_HAND")
                elif not quality_ok:
                    pred_window.append("UNKNOWN")
                    source = "QGATE"
                elif temporal_label is not None:
                    if temporal_conf >= cfg.temporal_confidence_threshold:
                        pred_window.append(temporal_label)
                    else:
                        pred_window.append("UNKNOWN")
                    confidence = temporal_conf
                    source = "TEMP"
                else:
                    pred_window.append("UNKNOWN")
                    source = "TEMP"

            else:  # hybrid
                if detection.hand_count == 0:
                    pred_window.append("NO_HAND")
                    source = "NONE"
                elif not quality_ok:
                    pred_window.append("UNKNOWN")
                    source = "QGATE"
                elif cfg.strict_consensus:
                    cons = _strict_consensus_decision(
                        rule_label=rule_label,
                        rule_conf=rule_conf,
                        proto_label=proto_label,
                        proto_conf=proto_conf,
                        ml_label=ml_label,
                        ml_conf=ml_conf,
                        temporal_label=temporal_label,
                        temporal_conf=temporal_conf,
                        override_conf=cfg.strict_override_conf,
                    )
                    if cons is not None:
                        cons_label, cons_conf, cons_src = cons
                        pred_window.append(cons_label)
                        confidence = cons_conf
                        source = cons_src
                    elif rule_label is not None and rule_conf >= cfg.rule_confidence_threshold:
                        pred_window.append(rule_label)
                        confidence = rule_conf
                        source = "RULE"
                    elif temporal_label is not None and temporal_conf >= cfg.temporal_confidence_threshold:
                        pred_window.append(temporal_label)
                        confidence = temporal_conf
                        source = "TEMP"
                    elif proto_label is not None and proto_conf >= cfg.prototype_threshold:
                        pred_window.append(proto_label)
                        confidence = proto_conf
                        source = "PROTO"
                    elif ml_label is not None:
                        ml_threshold = max(
                            cfg.confidence_threshold,
                            float(ml_label_thresholds.get(ml_label, cfg.confidence_threshold)),
                        )
                        if ml_conf >= ml_threshold and ml_margin >= cfg.ml_min_margin:
                            pred_window.append(ml_label)
                        else:
                            pred_window.append("UNKNOWN")
                        confidence = ml_conf
                        source = "ML"
                    else:
                        pred_window.append("UNKNOWN")
                        source = "NONE"
                elif rule_label is not None and rule_conf >= cfg.rule_confidence_threshold:
                    pred_window.append(rule_label)
                    confidence = rule_conf
                    source = "RULE"
                elif temporal_label is not None and temporal_conf >= cfg.temporal_confidence_threshold:
                    pred_window.append(temporal_label)
                    confidence = temporal_conf
                    source = "TEMP"
                elif proto_label is not None and proto_conf >= cfg.prototype_threshold:
                    pred_window.append(proto_label)
                    confidence = proto_conf
                    source = "PROTO"
                elif ml_label is not None:
                    ml_threshold = max(
                        cfg.confidence_threshold,
                        float(ml_label_thresholds.get(ml_label, cfg.confidence_threshold)),
                    )
                    if ml_conf >= ml_threshold and ml_margin >= cfg.ml_min_margin:
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
            if continuous_sentence and can_auto_append_token(
                label=label,
                stable_hits=stable_hits,
                min_stable_frames=cfg.min_stable_frames_for_speech,
                now_ts=now_speak,
                last_append_ts=last_sentence_append_time,
                append_cooldown_sec=cfg.sentence_append_cooldown_sec,
                last_token=(sentence[-1] if sentence else None),
            ):
                sentence.append(label)
                if len(sentence) > max(1, int(cfg.sentence_max_tokens)):
                    sentence.pop(0)
                last_sentence_append_time = now_speak
                last_sentence_token_time = now_speak

            if (
                continuous_sentence
                and voice_enabled
                and auto_speak
                and should_auto_speak_sentence(
                    tokens_count=len(sentence),
                    now_ts=now_speak,
                    last_token_ts=last_sentence_token_time,
                    last_sentence_speak_ts=last_sentence_spoken_time,
                    pause_sec=cfg.sentence_pause_speak_sec,
                )
            ):
                spoken_text = sentence_to_text(sentence)
                if spoken_text:
                    speaker.say_latest(spoken_text)
                    last_spoken_sentence = spoken_text
                    last_sentence_spoken_time = now_speak
                sentence.clear()

            can_repeat_same = (label == spoken_label) and ((now_speak - last_spoken_time) >= cfg.repeat_same_label_sec)
            if (
                voice_enabled
                and auto_speak
                and label not in {"NO_HAND", "UNKNOWN"}
                and stable_hits >= cfg.min_stable_frames_for_speech
                and (now_speak - last_spoken_time) >= cfg.speak_cooldown_sec
                and ((now_speak - last_spoken_by_label.get(label, 0.0)) >= cfg.per_label_cooldown_sec)
                and (label != spoken_label or can_repeat_same)
            ):
                # Avoid queued old labels causing delayed speaking.
                speech = speech_text_for_label(label)
                if speech:
                    speaker.say_latest(speech)
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

            sentence_text = ""
            if show_sentence:
                sentence_text = sentence_to_text(sentence[-8:])
                if not sentence_text and last_spoken_sentence:
                    sentence_text = f"(last) {last_spoken_sentence}"
            perf_text = f"intv {infer_every}"
            out = detection.frame
            if stage_mode:
                _draw_stage_hud(
                    out,
                    label=last_label,
                    confidence=last_confidence,
                    fps=fps,
                    voice_enabled=voice_enabled,
                    auto_speak=auto_speak,
                    continuous_sentence=continuous_sentence,
                    perf_text=perf_text,
                    recording=recording,
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
                    auto_speak=auto_speak,
                    continuous_sentence=continuous_sentence,
                    sentence_text=sentence_text,
                    perf_text=perf_text,
                )
            hint, hint_color = _compute_quality_hint(out, detection.hand_count, last_confidence, last_label)
            _draw_quality_hint(out, hint, hint_color)
            if show_help and not stage_mode:
                _draw_help(out)
            if cfg.demo_script:
                if demo_index < len(demo_steps):
                    prompt = demo_steps[demo_index]
                    progress = f"Step {demo_index + 1}/{len(demo_steps)}  (show this sign | n skip)"
                else:
                    prompt = "DONE"
                    progress = f"Completed {len(demo_steps)}/{len(demo_steps)}"
                _draw_demo_prompt(out, prompt, progress)
            cv2.imshow(window_name, out)
            if recording and video_writer is not None:
                video_writer.write(out)

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
            if ch == "a":
                auto_speak = not auto_speak
            if ch == "t":
                continuous_sentence = not continuous_sentence
                if continuous_sentence:
                    show_sentence = True
                print(f"[INFO] Continuous sentence: {'ON' if continuous_sentence else 'OFF'}")
            if ch == "m":
                order = ["rules", "hybrid", "ml", "temporal"]
                idx = order.index(mode) if mode in order else 0
                tried = 0
                while tried < len(order):
                    idx = (idx + 1) % len(order)
                    next_mode = order[idx]
                    tried += 1
                    if next_mode == "ml" and model is None:
                        continue
                    if next_mode == "temporal" and temporal_model is None:
                        continue
                    mode = next_mode
                    pred_window.clear()
                    seq_buffer.clear()
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
                last_spoken_sentence = sentence_to_text(sentence)
                last_sentence_spoken_time = time.time()
                sentence.clear()
            if ch == "p":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                shots_dir = cfg.session_log_path.parent / "screenshots"
                shots_dir.mkdir(parents=True, exist_ok=True)
                shot_path = shots_dir / f"frame_{ts}.png"
                cv2.imwrite(str(shot_path), out)
                print(f"Saved screenshot: {shot_path}")
            if ch == "k":
                if not recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    videos_dir = cfg.session_log_path.parent / "videos"
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    video_path = videos_dir / f"demo_{ts}.mp4"
                    h, w = out.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (w, h))
                    if video_writer.isOpened():
                        recording = True
                        print(f"[INFO] Recording started: {video_path}")
                    else:
                        video_writer.release()
                        video_writer = None
                        video_path = None
                        print("[WARN] Failed to start recording.")
                else:
                    recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    if video_path is not None:
                        print(f"[INFO] Recording saved: {video_path}")
                        video_path = None
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
        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
