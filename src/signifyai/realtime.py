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
from .async_inference import LatestFrameWorker
from .config import (
    FEATURE_SIZE,
    DEFAULT_DATASET_PATH,
    DEFAULT_DEEP_LABELS_PATH,
    DEFAULT_DEEP_METADATA_PATH,
    DEFAULT_DEEP_MODEL_PATH,
    DEFAULT_DEEP_PREPROCESS_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROTOTYPE_DB_PATH,
    DEFAULT_SEQUENCE_DATASET_PATH,
    DEFAULT_SESSION_LOG_PATH,
    DEFAULT_TEMPORAL_LABELS_PATH,
    DEFAULT_TEMPORAL_METADATA_PATH,
    DEFAULT_TEMPORAL_MODEL_PATH,
)
from .dataset import save_records
from .feature_extraction import normalize_features
from .hand_tracking import DetectionResult, HandTracker, check_camera, open_camera, warmup_camera
from .language import sentence_to_text, speech_text_for_label
from .modeling import load_model
from .rules import RuleBasedInterpreter
from .sequence_dataset import append_sequence_records
from .temporal_model import load_temporal_model
from .deep_infer import load_deep_runtime, predict_deep
from .prototype_adapt import load_prototype_db, predict_prototype
from .sentence_decoder import SentenceDecoder, SentenceDecoderConfig
from .train import TrainConfig, run_training
from .tts import SpeechEngine

BUILTIN_RULE_LABELS = {
    "HELLO",
    "GOOD MORNING",
    "GOOD AFTERNOON",
    "GOOD EVENING",
    "GOOD NIGHT",
    "OKAY",
    "YES",
    "NO",
    "TWO",
    "PEACE",
    "STOP",
    "ONE",
    "CALL ME",
    "I LOVE YOU",
    "ROCK",
    "HELP",
    "THANK YOU",
}
DIGIT_LABELS = {str(i) for i in range(10)}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


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
    min_stable_frames_for_speech: int = 2
    mode: str = "hybrid"  # rules | ml | temporal | hybrid
    rule_confidence_threshold: float = 0.78
    inference_interval: int = 1
    inference_scale: float = 0.75
    model_complexity: int = 0
    max_num_hands: int = 2
    landmark_smoothing: float = 0.78
    adaptive_performance: bool = True
    target_fps: float = 30.0
    repeat_same_label_sec: float = 8.0
    speak_cooldown_sec: float = 0.45
    per_label_cooldown_sec: float = 0.9
    auto_speak: bool = True
    show_sentence: bool = True
    stage_mode: bool = False
    label_hold_sec: float = 0.12
    demo_script: bool = False
    temporal_model_path: Path = DEFAULT_TEMPORAL_MODEL_PATH
    temporal_labels_path: Path = DEFAULT_TEMPORAL_LABELS_PATH
    temporal_metadata_path: Path = DEFAULT_TEMPORAL_METADATA_PATH
    temporal_confidence_threshold: float = 0.60
    enhance_frame: bool = False
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
    use_deep_model: bool = False
    deep_model_path: Path = DEFAULT_DEEP_MODEL_PATH
    deep_labels_path: Path = DEFAULT_DEEP_LABELS_PATH
    deep_metadata_path: Path = DEFAULT_DEEP_METADATA_PATH
    deep_preprocess_path: Path = DEFAULT_DEEP_PREPROCESS_PATH
    deep_confidence_threshold: float = 0.62
    deep_min_margin: float = 0.06
    continuous_sentence: bool = False
    sentence_pause_speak_sec: float = 1.0
    sentence_append_cooldown_sec: float = 0.35
    sentence_max_tokens: int = 14
    async_inference: bool = False
    tts_rate: int = 180
    tts_volume: float = 1.0
    tts_dedup_sec: float = 0.30
    tts_min_gap_sec: float = 0.14
    static_frame_skip: bool = True
    static_frame_diff_threshold: float = 1.8
    static_skip_max_frames: int = 10
    deep_auto_throttle: bool = True
    deep_disable_fps_drop: float = 10.0
    deep_disable_streak: int = 12
    deep_reenable_margin: float = 3.0
    deep_reenable_streak: int = 24
    prediction_interval: int = 1
    pose_reset_delta: float = 0.040
    live_teach_label: str = ""
    live_dataset_path: Path = DEFAULT_DATASET_PATH
    live_sequence_dataset_path: Path = DEFAULT_SEQUENCE_DATASET_PATH
    live_capture_enabled: bool = False
    live_capture_interval_sec: float = 0.30
    live_min_feature_delta: float = 0.010
    live_flush_every: int = 20
    live_sequence_enabled: bool = True
    live_sequence_len: int = 24
    live_sequence_min_visible_frames: int = 14
    live_auto_retrain: bool = False
    live_retrain_every_samples: int = 60
    live_min_samples_per_label: int = 5
    quality_hint_interval_sec: float = 0.35
    quality_gate_eval_interval_sec: float = 0.20
    motion_gate_delta: float = 0.026
    mini_runtime: bool = False


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
        "u: retrain live model now",
        "p/k: screenshot/recording",
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


def _draw_cached_points(
    frame: np.ndarray,
    raw_hands: list[np.ndarray],
    handedness: Optional[list[str]] = None,
) -> None:
    if not raw_hands:
        return
    h, w = frame.shape[:2]
    tips = {4, 8, 12, 16, 20}
    handedness = handedness or []
    for idx, hand in enumerate(raw_hands):
        side = handedness[idx] if idx < len(handedness) else "unknown"
        if side == "left":
            conn_color = (120, 255, 120)  # green-ish
        elif side == "right":
            conn_color = (120, 200, 255)  # cyan-ish
        else:
            conn_color = (240, 240, 240)
        for a, b in HAND_CONNECTIONS:
            xa = int(hand[a, 0] * w)
            ya = int(hand[a, 1] * h)
            xb = int(hand[b, 0] * w)
            yb = int(hand[b, 1] * h)
            cv2.line(frame, (xa, ya), (xb, yb), conn_color, 2)
        for i in range(hand.shape[0]):
            x = int(hand[i, 0] * w)
            y = int(hand[i, 1] * h)
            if i == 0:
                pt_color = (0, 255, 255)  # yellow wrist
            elif i in tips:
                pt_color = (0, 0, 255)  # red tips
            else:
                pt_color = (0, 255, 0)  # green joints
            cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)
            cv2.circle(frame, (x, y), 3, pt_color, -1)


def _tune_infer_interval(
    *,
    infer_every: int,
    fps: float,
    perf_target: float,
    hand_count: int,
    max_interval: int = 4,
) -> int:
    """
    Responsiveness-first tuning:
    - with visible hands: prefer 1, but allow 2 on low FPS to avoid stalls
    - without hands: allow higher interval for FPS recovery
    """
    infer_every = max(1, int(infer_every))
    if hand_count > 0:
        hand_max_interval = min(max_interval, 2)
        if fps < (perf_target - 6.0) and infer_every < hand_max_interval:
            return infer_every + 1
        if fps > (perf_target - 2.0) and infer_every > 1:
            return infer_every - 1
        return infer_every
    if fps < (perf_target - 3.0) and infer_every < max_interval:
        return infer_every + 1
    if fps > (perf_target + 4.0) and infer_every > 1:
        return infer_every - 1
    return infer_every


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
    """Display-only clarity boost for webcam feed."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    y2 = clahe.apply(y)
    tuned = cv2.merge((y2, cr, cb))
    tuned_bgr = cv2.cvtColor(tuned, cv2.COLOR_YCrCb2BGR)
    blur = cv2.GaussianBlur(tuned_bgr, (0, 0), 1.0)
    sharp = cv2.addWeighted(tuned_bgr, 1.16, blur, -0.16, 0)
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


def _fuse_frame_models(
    ml_label: Optional[str],
    ml_conf: float,
    ml_margin: float,
    ml_threshold: float,
    ml_min_margin: float,
    deep_label: Optional[str],
    deep_conf: float,
    deep_margin: float,
    deep_threshold: float,
    deep_min_margin: float,
) -> tuple[Optional[str], float, str]:
    """
    Fuse classic ML and deep predictions.
    If both are strong and disagree with similar confidence, block prediction to reduce false positives.
    """
    ml_ok = ml_label is not None and ml_conf >= ml_threshold and ml_margin >= ml_min_margin
    deep_ok = deep_label is not None and deep_conf >= deep_threshold and deep_margin >= deep_min_margin

    if ml_ok and deep_ok:
        if ml_label == deep_label:
            return ml_label, max(ml_conf, deep_conf), "ML+DEEP"
        if abs(ml_conf - deep_conf) <= 0.07:
            return None, max(ml_conf, deep_conf), "ML_DEEP_DISAGREE"
        if ml_conf > deep_conf:
            return ml_label, ml_conf, "ML"
        return deep_label, deep_conf, "DEEP"

    if deep_ok:
        return deep_label, deep_conf, "DEEP"
    if ml_ok:
        return ml_label, ml_conf, "ML"
    return None, max(ml_conf, deep_conf), "ML_DEEP_LOW_CONF"


def _weighted_label_vote(labels: deque[str], confidences: deque[float]) -> tuple[str, float]:
    """
    Weighted voting for stability:
    - each frame vote contributes by confidence plus small base weight
    - common uncertain labels get lower base weight
    """
    if not labels:
        return "NO_HAND", 0.0

    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for lbl, conf in zip(labels, confidences):
        base = 0.20 if lbl not in {"NO_HAND", "UNKNOWN"} else 0.10
        weight = base + max(0.0, min(1.0, float(conf)))
        scores[lbl] = scores.get(lbl, 0.0) + weight
        counts[lbl] = counts.get(lbl, 0) + 1

    best_label = max(scores.keys(), key=lambda k: (scores[k], counts.get(k, 0)))
    avg_weight = scores[best_label] / max(1, counts.get(best_label, 1))
    # Convert weight-like scale roughly back to 0..1 confidence.
    voted_conf = max(0.0, min(1.0, avg_weight - (0.20 if best_label not in {"NO_HAND", "UNKNOWN"} else 0.10)))
    return best_label, voted_conf


def _frame_motion_score(prev_gray: Optional[np.ndarray], frame_bgr: np.ndarray) -> tuple[float, np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return 999.0, gray
    diff = cv2.absdiff(prev_gray, gray)
    score = float(diff.mean())
    return score, gray


def _feature_motion_delta(prev_features: Optional[np.ndarray], features: np.ndarray) -> float:
    if prev_features is None or prev_features.shape != features.shape:
        return 0.0
    return float(np.mean(np.abs(features - prev_features)))


def _prefer_custom_label(
    *,
    rule_label: Optional[str],
    rule_conf: float,
    fused_label: Optional[str],
    fused_conf: float,
    proto_label: Optional[str],
    proto_conf: float,
    temporal_label: Optional[str],
    temporal_conf: float,
) -> Optional[tuple[str, float, str]]:
    candidates: list[tuple[str, float, str]] = []
    for lbl, conf, src in [
        (fused_label, fused_conf, "ML"),
        (proto_label, proto_conf, "PROTO"),
        (temporal_label, temporal_conf, "TEMP"),
    ]:
        if lbl and lbl not in BUILTIN_RULE_LABELS:
            candidates.append((lbl, conf, src))
    if not candidates:
        return None

    best_label, best_conf, best_src = max(candidates, key=lambda item: item[1])
    if rule_label is None:
        return best_label, best_conf, best_src
    if best_label == rule_label:
        return best_label, best_conf, best_src
    if best_conf >= max(0.72, rule_conf + 0.06):
        return best_label, best_conf, f"{best_src}_CUSTOM"
    return None


def _suppress_digit_prediction(
    *,
    hand_count: int,
    label: Optional[str],
    confidence: float,
    margin: float,
    rule_label: Optional[str],
    temporal_label: Optional[str],
    strict_conf: float = 0.98,
    strict_margin: float = 0.20,
) -> bool:
    if label is None or label not in DIGIT_LABELS:
        return False
    if temporal_label is not None:
        return False
    if hand_count >= 2:
        return True
    if rule_label is not None and rule_label not in DIGIT_LABELS:
        return True
    if confidence < strict_conf:
        return True
    if margin < strict_margin:
        return True
    return False


def _should_force_sync_detection(
    *,
    run_inference: bool,
    last_detection: Optional[DetectionResult],
    motion_score: float,
    static_diff_threshold: float,
    frames_since_fresh_result: int,
) -> bool:
    """
    Rescue path for async mode: if a new hand likely entered the frame and the
    worker has not produced a fresh result yet, do one synchronous pass to keep
    perceived latency low.
    """
    if not run_inference:
        return False
    if frames_since_fresh_result < 2:
        return False
    if motion_score <= max(2.0, float(static_diff_threshold) * 1.35):
        return False
    if last_detection is None:
        return True
    return last_detection.hand_count == 0


def _normalize_live_label(raw: str) -> str:
    return str(raw).strip().lower().replace(" ", "_")


def _run_realtime_lite(cfg: RealtimeConfig) -> None:
    """
    Lightweight runtime loop modeled after the mini prototype:
    - rules-only prediction
    - tracker runs every alternate frame
    - small smoothing window
    - aggressive speech queue replacement
    """
    cap = open_camera(cfg.camera_index, cfg.width, cfg.height, cfg.camera_fps)
    err = check_camera(cap)
    if err is not None:
        raise RuntimeError(err)
    warmup_camera(cap, frames=8)

    tracker = HandTracker(
        max_num_hands=min(2, int(cfg.max_num_hands)),
        inference_scale=min(float(cfg.inference_scale), 0.62),
        model_complexity=0,
        landmark_smoothing=min(float(cfg.landmark_smoothing), 0.72),
    )
    rules = RuleBasedInterpreter()
    speaker = SpeechEngine(
        rate=cfg.tts_rate,
        volume=cfg.tts_volume,
        dedup_sec=min(0.28, float(cfg.tts_dedup_sec)),
        min_gap_sec=min(0.12, float(cfg.tts_min_gap_sec)),
    )

    window_name = "SignifyAI Lite"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, cfg.width, cfg.height)

    pred_window: deque[str] = deque(maxlen=max(3, min(5, int(cfg.smoothing_window))))
    last_detection: Optional[DetectionResult] = None
    frame_idx = 0
    prev_time = time.time()
    fps = 0.0
    voice_enabled = bool(cfg.auto_speak)
    spoken_label = ""
    no_hand_streak = 0
    last_spoken_time = 0.0
    last_spoken_by_label: dict[str, float] = {}
    last_label = "NO_HAND"
    last_conf = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame_idx += 1

            run_tracking = (frame_idx % 2 == 0) or (last_detection is None)
            if run_tracking:
                detection = tracker.process(frame, draw=True)
                last_detection = detection
            else:
                if last_detection is None:
                    detection = tracker.process(frame, draw=True)
                    last_detection = detection
                else:
                    detection = DetectionResult(
                        features=last_detection.features,
                        hand_count=last_detection.hand_count,
                        frame=frame.copy(),
                        raw_hands=last_detection.raw_hands,
                        handedness=last_detection.handedness,
                    )
                    _draw_cached_points(detection.frame, detection.raw_hands, detection.handedness)

            pred = rules.predict(detection) if detection.hand_count > 0 else None
            if detection.hand_count == 0:
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

            if label == "NO_HAND":
                no_hand_streak += 1
                if no_hand_streak == 1:
                    speaker.stop_current()
            else:
                no_hand_streak = 0
            if no_hand_streak >= 3:
                spoken_label = ""

            if (
                voice_enabled
                and label not in {"NO_HAND", "UNKNOWN"}
                and (now - last_spoken_time) >= max(0.20, float(cfg.speak_cooldown_sec))
                and (label != spoken_label or (now - last_spoken_time) >= max(2.0, float(cfg.repeat_same_label_sec)))
                and (now - last_spoken_by_label.get(label, 0.0)) >= max(0.30, float(cfg.per_label_cooldown_sec))
            ):
                speech = speech_text_for_label(label)
                if speech:
                    speaker.say_latest(speech)
                    spoken_label = label
                    last_spoken_time = now
                    last_spoken_by_label[label] = now
                    append_event(cfg.session_log_path, label=label, confidence=confidence, hand_count=detection.hand_count)

            last_label = label
            last_conf = confidence

            out = detection.frame
            _draw_compact_hud(
                out,
                label=last_label,
                hands=detection.hand_count,
                fps=fps,
                confidence=last_conf,
                mode_text="LITE RULES",
                voice_enabled=voice_enabled,
                auto_speak=voice_enabled,
                continuous_sentence=False,
                sentence_text="",
                perf_text="mini",
            )
            hint, hint_color = _compute_quality_hint(out, detection.hand_count, last_conf, last_label)
            _draw_quality_hint(out, hint, hint_color)
            cv2.imshow(window_name, out)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            low = key & 0xFF
            ch = chr(low).lower() if 0 <= low <= 255 else ""
            if key == 27 or ch == "q":
                break
            if ch == "v":
                voice_enabled = not voice_enabled
            if ch == "r":
                pred_window.clear()
                spoken_label = ""
    finally:
        tracker.close()
        speaker.close()
        cap.release()
        cv2.destroyAllWindows()


def run_realtime(cfg: RealtimeConfig) -> None:
    if cfg.mini_runtime:
        _run_realtime_lite(cfg)
        return

    model = None
    labels: list[str] = []
    ml_label_thresholds: dict[str, float] = {}
    deep_bundle = None
    temporal_model = None
    temporal_labels: list[str] = []
    temporal_seq_len = 24
    prototype_db = None
    live_teach_label = _normalize_live_label(cfg.live_teach_label)
    live_capture_enabled = bool(cfg.live_capture_enabled and live_teach_label)
    mode = cfg.mode.lower().strip()
    if mode not in {"rules", "ml", "temporal", "hybrid"}:
        mode = "hybrid"

    def _reload_frame_model_runtime() -> None:
        nonlocal model, labels, ml_label_thresholds
        model, labels = load_model(cfg.model_path, cfg.labels_path)
        ml_label_thresholds = {}
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

    if mode in {"ml", "hybrid"}:
        try:
            _reload_frame_model_runtime()
        except Exception as ex:
            # Keep console clean; fallback silently unless explicitly in ml mode.
            if mode == "ml":
                print(f"[INFO] ML model unavailable: {ex}")
                print("[INFO] Falling back to rules mode.")
                mode = "rules"

        if cfg.use_deep_model:
            try:
                deep_bundle = load_deep_runtime(
                    model_path=cfg.deep_model_path,
                    labels_path=cfg.deep_labels_path,
                    preprocess_path=cfg.deep_preprocess_path,
                    metadata_path=cfg.deep_metadata_path,
                )
                print(f"[INFO] Deep runtime enabled: {cfg.deep_model_path}")
            except Exception as ex:
                print(f"[WARN] Deep runtime unavailable: {ex}")
                deep_bundle = None

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
        max_num_hands=cfg.max_num_hands,
        inference_scale=cfg.inference_scale,
        model_complexity=cfg.model_complexity,
        landmark_smoothing=cfg.landmark_smoothing,
    )
    async_tracker: Optional[HandTracker] = None
    tracker_worker: Optional[LatestFrameWorker[np.ndarray, DetectionResult]] = None
    if cfg.async_inference:
        async_tracker = HandTracker(
            max_num_hands=cfg.max_num_hands,
            inference_scale=cfg.inference_scale,
            model_complexity=cfg.model_complexity,
            landmark_smoothing=cfg.landmark_smoothing,
        )
        tracker_worker = LatestFrameWorker(lambda img: async_tracker.process(img, draw=False))
        tracker_worker.start()
    rules = RuleBasedInterpreter()
    speaker = SpeechEngine(
        rate=cfg.tts_rate,
        volume=cfg.tts_volume,
        dedup_sec=cfg.tts_dedup_sec,
        min_gap_sec=cfg.tts_min_gap_sec,
        max_queue=2,
    )

    window_name = "SignifyAI Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, cfg.width, cfg.height)

    pred_window = deque(maxlen=cfg.smoothing_window)
    pred_conf_window = deque(maxlen=cfg.smoothing_window)
    seq_buffer = deque(maxlen=max(4, int(temporal_seq_len)))
    spoken_label = ""
    last_frame_label = "NO_HAND"
    stable_hits = 0
    no_hand_streak = 0
    pending_label = "NO_HAND"
    pending_since = time.time()
    accepted_label = "NO_HAND"
    sentence_decoder = SentenceDecoder(
        SentenceDecoderConfig(
            min_stable_frames=1 if cfg.continuous_sentence else cfg.min_stable_frames_for_speech,
            append_cooldown_sec=cfg.sentence_append_cooldown_sec,
            transition_cooldown_sec=min(0.10, cfg.sentence_append_cooldown_sec),
            pause_speak_sec=cfg.sentence_pause_speak_sec,
            max_tokens=cfg.sentence_max_tokens,
            no_hand_flush_frames=2,
        )
    )
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
    prev_gray_for_skip: Optional[np.ndarray] = None
    static_skip_streak = 0
    deep_runtime_active = deep_bundle is not None
    low_fps_streak = 0
    high_fps_streak = 0
    last_async_result_id = 0
    frames_since_fresh_async_result = 999
    prev_pose_features: Optional[np.ndarray] = None
    last_pred_hand_count = -1
    cached_ml_label: Optional[str] = None
    cached_ml_conf = 0.0
    cached_ml_margin = 0.0
    cached_deep_label: Optional[str] = None
    cached_deep_conf = 0.0
    cached_deep_margin = 0.0
    cached_proto_label: Optional[str] = None
    cached_proto_conf = 0.0
    cached_temporal_label: Optional[str] = None
    cached_temporal_conf = 0.0
    cached_resolved_label: Optional[str] = None
    cached_resolved_conf = 0.0
    cached_resolved_source = "NONE"
    last_quality_hint_ts = 0.0
    last_quality_hint = "Tracking ready"
    last_quality_hint_color = (180, 220, 255)
    last_quality_hand_count = -1
    last_quality_eval_ts = 0.0
    last_quality_eval_hand_count = -1
    last_quality_brightness = 999.0
    last_quality_blur_metric = 999.0
    live_pending_records: list[tuple[np.ndarray, str]] = []
    live_saved_total = 0
    live_saved_sequences = 0
    live_capture_status = "OFF"
    live_last_capture_ts = 0.0
    live_last_saved_features: Optional[np.ndarray] = None
    live_hand_streak = 0
    live_sequence_buffer: list[np.ndarray] = []
    live_sequence_visible_frames = 0
    live_samples_since_retrain = 0

    def _flush_live_records() -> int:
        nonlocal live_pending_records, live_saved_total
        if not live_pending_records:
            return 0
        saved_now = save_records(live_pending_records, cfg.live_dataset_path)
        live_saved_total += saved_now
        live_pending_records = []
        return saved_now

    def _capture_live_sample(features: np.ndarray, now_ts: float) -> bool:
        nonlocal live_last_capture_ts, live_last_saved_features, live_capture_status
        if (now_ts - live_last_capture_ts) < max(0.05, float(cfg.live_capture_interval_sec)):
            return False
        if live_last_saved_features is not None:
            delta = float(np.mean(np.abs(features - live_last_saved_features)))
            if delta < float(cfg.live_min_feature_delta):
                live_capture_status = f"Teach {live_teach_label}: skip duplicate ({delta:.4f})"
                return False
        live_pending_records.append((features.copy(), live_teach_label))
        live_last_saved_features = features.copy()
        live_last_capture_ts = now_ts
        live_capture_status = f"Teach {live_teach_label}: captured {live_saved_total + len(live_pending_records)}"
        if len(live_pending_records) >= max(1, int(cfg.live_flush_every)):
            saved_now = _flush_live_records()
            if saved_now:
                print(f"[INFO] Live teach flushed {saved_now} frame samples to {cfg.live_dataset_path}")
        return True

    def _flush_live_sequence(now_reason: str) -> None:
        nonlocal live_sequence_buffer, live_sequence_visible_frames, live_saved_sequences, live_capture_status
        if not cfg.live_sequence_enabled:
            live_sequence_buffer = []
            live_sequence_visible_frames = 0
            return
        seq_len = max(4, int(cfg.live_sequence_len))
        if len(live_sequence_buffer) >= seq_len and live_sequence_visible_frames >= max(2, int(cfg.live_sequence_min_visible_frames)):
            clip = np.asarray(live_sequence_buffer[-seq_len:], dtype=np.float32)
            saved = append_sequence_records([(clip, live_teach_label)], cfg.live_sequence_dataset_path, seq_len=seq_len)
            live_saved_sequences += saved
            live_capture_status = f"Teach {live_teach_label}: saved {live_saved_sequences} live clips"
            print(f"[INFO] Live teach saved sequence clip ({now_reason}) to {cfg.live_sequence_dataset_path}")
        live_sequence_buffer = []
        live_sequence_visible_frames = 0

    def _run_live_retrain(force: bool = False) -> None:
        nonlocal live_samples_since_retrain, live_capture_status, model, mode
        if not live_capture_enabled:
            return
        if (not force) and (not cfg.live_auto_retrain):
            return
        if (not force) and live_samples_since_retrain < max(1, int(cfg.live_retrain_every_samples)):
            return
        flushed_now = _flush_live_records()
        live_samples_since_retrain = 0
        if flushed_now <= 0 and not force:
            return
        try:
            acc = run_training(
                TrainConfig(
                    dataset_csv=cfg.live_dataset_path,
                    model_path=cfg.model_path,
                    labels_path=cfg.labels_path,
                    metadata_path=cfg.metadata_path,
                    calibrate_probs=True,
                    automl=False,
                    min_samples_per_label=max(1, int(cfg.live_min_samples_per_label)),
                )
            )
            _reload_frame_model_runtime()
            if mode == "rules":
                mode = "ml"
            live_capture_status = f"Teach {live_teach_label}: retrained model acc={acc:.2f}"
            print(f"[INFO] Live teach retrain complete. Accuracy={acc:.4f}")
        except Exception as ex:
            live_capture_status = f"Teach {live_teach_label}: retrain skipped ({ex})"
            print(f"[WARN] Live teach retrain skipped: {ex}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_idx += 1
            motion_score, curr_gray = _frame_motion_score(prev_gray_for_skip, frame)
            prev_gray_for_skip = curr_gray

            run_inference = (frame_idx % infer_every == 0) or (last_detection is None)
            if tracker_worker is not None:
                detection: Optional[DetectionResult] = None
                should_skip_static = False
                if run_inference:
                    should_skip_static = (
                        cfg.static_frame_skip
                        and last_detection is not None
                        and last_detection.hand_count == 0
                        and motion_score <= cfg.static_frame_diff_threshold
                        and static_skip_streak < max(1, int(cfg.static_skip_max_frames))
                    )
                    if should_skip_static:
                        static_skip_streak += 1
                    else:
                        static_skip_streak = 0
                        tracker_worker.submit(frame.copy())
                async_detection, async_result_id = tracker_worker.poll_latest_result_with_id()
                if async_detection is not None and async_result_id != last_async_result_id:
                    last_detection = async_detection
                    last_async_result_id = async_result_id
                    frames_since_fresh_async_result = 0
                else:
                    frames_since_fresh_async_result += 1

                if _should_force_sync_detection(
                    run_inference=run_inference and (not should_skip_static),
                    last_detection=last_detection,
                    motion_score=motion_score,
                    static_diff_threshold=cfg.static_frame_diff_threshold,
                    frames_since_fresh_result=frames_since_fresh_async_result,
                ):
                    detection = tracker.process(frame, draw=False)
                    last_detection = detection
                    frames_since_fresh_async_result = 0
                if last_detection is not None:
                    if detection is None:
                        detection = DetectionResult(
                            features=last_detection.features.copy(),
                            hand_count=last_detection.hand_count,
                            frame=frame.copy(),
                            raw_hands=[h.copy() for h in last_detection.raw_hands],
                            handedness=list(last_detection.handedness),
                        )
                else:
                    detection = DetectionResult(
                        features=np.zeros((FEATURE_SIZE,), dtype=np.float32),
                        hand_count=0,
                        frame=frame.copy(),
                        raw_hands=[],
                        handedness=[],
                    )
            else:
                if run_inference:
                    should_skip_static = (
                        cfg.static_frame_skip
                        and last_detection is not None
                        and last_detection.hand_count == 0
                        and motion_score <= cfg.static_frame_diff_threshold
                        and static_skip_streak < max(1, int(cfg.static_skip_max_frames))
                    )
                    if should_skip_static:
                        static_skip_streak += 1
                        detection = type(last_detection)(
                            features=last_detection.features,
                            hand_count=last_detection.hand_count,
                            frame=frame.copy(),
                            raw_hands=last_detection.raw_hands,
                            handedness=last_detection.handedness,
                        )
                    else:
                        static_skip_streak = 0
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
                    _draw_cached_points(detection.frame, detection.raw_hands, detection.handedness)
            if tracker_worker is not None:
                _draw_cached_points(detection.frame, detection.raw_hands, detection.handedness)

            features = normalize_features(detection.features)
            pose_delta = _feature_motion_delta(prev_pose_features, features)
            motion_block = detection.hand_count > 0 and pose_delta >= float(cfg.motion_gate_delta)
            if detection.hand_count > 0 and pose_delta >= float(cfg.pose_reset_delta):
                pred_window.clear()
                pred_conf_window.clear()
                stable_hits = 0
                pending_label = "UNKNOWN"
                pending_since = time.time()
                cached_resolved_label = None
                cached_resolved_conf = 0.0
                cached_resolved_source = "RESET"
                speaker.stop_current()
            prev_pose_features = features.copy() if detection.hand_count > 0 else None
            if detection.hand_count > 0:
                seq_buffer.append(features.astype(np.float32))
            else:
                seq_buffer.clear()
                cached_ml_label = None
                cached_ml_conf = 0.0
                cached_ml_margin = 0.0
                cached_deep_label = None
                cached_deep_conf = 0.0
                cached_deep_margin = 0.0
                cached_proto_label = None
                cached_proto_conf = 0.0
                cached_temporal_label = None
                cached_temporal_conf = 0.0
                cached_resolved_label = None
                cached_resolved_conf = 0.0
                cached_resolved_source = "NONE"

            run_classifier = (
                detection.hand_count > 0
                and (not motion_block)
                and (
                    frame_idx % max(1, int(cfg.prediction_interval)) == 0
                    or detection.hand_count != last_pred_hand_count
                )
            )
            last_pred_hand_count = detection.hand_count

            label = "NO_HAND"
            confidence = 0.0
            source = "NONE"

            quality_ok = True
            if cfg.quality_gate and detection.hand_count > 0:
                now_quality = time.time()
                refresh_quality = (
                    (now_quality - last_quality_eval_ts) >= max(0.05, float(cfg.quality_gate_eval_interval_sec))
                    or detection.hand_count != last_quality_eval_hand_count
                )
                if refresh_quality:
                    brightness, blur_metric = _frame_metrics(frame)
                    last_quality_brightness = brightness
                    last_quality_blur_metric = blur_metric
                    last_quality_eval_ts = now_quality
                    last_quality_eval_hand_count = detection.hand_count
                else:
                    brightness = last_quality_brightness
                    blur_metric = last_quality_blur_metric
                hand_area = _max_hand_area(detection.raw_hands)
                quality_ok = (
                    brightness >= cfg.min_brightness
                    and blur_metric >= cfg.min_blur_var
                    and hand_area >= cfg.min_hand_area
                )
            else:
                hand_area = _max_hand_area(detection.raw_hands)

            if live_capture_enabled:
                if detection.hand_count > 0 and quality_ok:
                    live_hand_streak += 1
                    live_features = normalize_features(detection.features).astype(np.float32)
                    if (not motion_block) and live_hand_streak >= 2 and _capture_live_sample(live_features, time.time()):
                        live_samples_since_retrain += 1
                    if cfg.live_sequence_enabled:
                        live_sequence_buffer.append(live_features)
                        if len(live_sequence_buffer) > max(4, int(cfg.live_sequence_len) * 2):
                            live_sequence_buffer = live_sequence_buffer[-max(4, int(cfg.live_sequence_len) * 2) :]
                        live_sequence_visible_frames += 1
                else:
                    if live_sequence_buffer:
                        _flush_live_sequence("boundary")
                    live_hand_streak = 0
                if cfg.live_auto_retrain and live_samples_since_retrain >= max(1, int(cfg.live_retrain_every_samples)):
                    _run_live_retrain(force=False)

            reuse_cached_prediction = (
                detection.hand_count > 0
                and (not motion_block)
                and (not run_classifier)
                and cached_resolved_label is not None
            )
            if reuse_cached_prediction:
                label = cached_resolved_label
                confidence = cached_resolved_conf
                source = cached_resolved_source
            else:
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
                if mode in {"ml", "hybrid"} and model is not None and detection.hand_count > 0 and run_classifier:
                    probs = model.predict_proba([features])[0]
                    top_idx = np.argsort(probs)[::-1]
                    best_idx = int(top_idx[0])
                    second_idx = int(top_idx[1]) if len(top_idx) > 1 else best_idx
                    ml_label = str(model.classes_[best_idx])
                    ml_conf = float(probs[best_idx])
                    ml_margin = float(probs[best_idx] - probs[second_idx]) if len(top_idx) > 1 else 1.0
                    cached_ml_label = ml_label
                    cached_ml_conf = ml_conf
                    cached_ml_margin = ml_margin
                elif detection.hand_count > 0:
                    ml_label = cached_ml_label
                    ml_conf = cached_ml_conf
                    ml_margin = cached_ml_margin

                deep_label: Optional[str] = None
                deep_conf = 0.0
                deep_margin = 0.0
                if mode in {"ml", "hybrid"} and deep_runtime_active and deep_bundle is not None and detection.hand_count > 0 and run_classifier:
                    deep_label, deep_conf, deep_margin = predict_deep(deep_bundle, features)
                    cached_deep_label = deep_label
                    cached_deep_conf = deep_conf
                    cached_deep_margin = deep_margin
                elif detection.hand_count > 0:
                    deep_label = cached_deep_label
                    deep_conf = cached_deep_conf
                    deep_margin = cached_deep_margin

                ml_threshold = cfg.confidence_threshold
                if ml_label is not None:
                    ml_threshold = max(
                        cfg.confidence_threshold,
                        float(ml_label_thresholds.get(ml_label, cfg.confidence_threshold)),
                    )

                temporal_label: Optional[str] = None
                temporal_conf = 0.0
                if (
                    mode in {"temporal", "hybrid"}
                    and temporal_model is not None
                    and detection.hand_count > 0
                    and len(seq_buffer) >= temporal_seq_len
                    and run_classifier
                ):
                    seq = np.asarray(list(seq_buffer)[-temporal_seq_len:], dtype=np.float32).reshape(1, -1)
                    probs_t = temporal_model.predict_proba(seq)[0]
                    best_t = int(np.argmax(probs_t))
                    classes_t = list(getattr(temporal_model, "classes_", temporal_labels))
                    temporal_label = str(classes_t[best_t]) if classes_t else None
                    temporal_conf = float(probs_t[best_t])
                    cached_temporal_label = temporal_label
                    cached_temporal_conf = temporal_conf
                elif detection.hand_count > 0:
                    temporal_label = cached_temporal_label
                    temporal_conf = cached_temporal_conf

                fused_label, fused_conf, fused_source = _fuse_frame_models(
                    ml_label=ml_label,
                    ml_conf=ml_conf,
                    ml_margin=ml_margin,
                    ml_threshold=ml_threshold,
                    ml_min_margin=cfg.ml_min_margin,
                    deep_label=deep_label,
                    deep_conf=deep_conf,
                    deep_margin=deep_margin,
                    deep_threshold=cfg.deep_confidence_threshold,
                    deep_min_margin=cfg.deep_min_margin,
                )
                if _suppress_digit_prediction(
                    hand_count=detection.hand_count,
                    label=fused_label,
                    confidence=fused_conf,
                    margin=max(ml_margin, deep_margin),
                    rule_label=rule_label,
                    temporal_label=temporal_label,
                ):
                    fused_label = None
                    fused_conf = 0.0
                    fused_source = "DIGIT_BLOCK"

                proto_label: Optional[str] = None
                proto_conf = 0.0
                if detection.hand_count > 0 and prototype_db is not None and prototype_db.vectors.shape[0] > 0 and run_classifier:
                    pm = predict_prototype(
                        features=features,
                        db=prototype_db,
                        min_similarity=cfg.prototype_threshold,
                        min_margin=cfg.prototype_margin,
                    )
                    if pm is not None:
                        proto_label = pm.label
                        proto_conf = pm.similarity
                    if _suppress_digit_prediction(
                        hand_count=detection.hand_count,
                        label=proto_label,
                        confidence=proto_conf,
                        margin=cfg.prototype_margin,
                        rule_label=rule_label,
                        temporal_label=temporal_label,
                        strict_conf=0.995,
                        strict_margin=max(0.08, cfg.prototype_margin),
                    ):
                        proto_label = None
                        proto_conf = 0.0
                    cached_proto_label = proto_label
                    cached_proto_conf = proto_conf
                elif detection.hand_count > 0:
                    proto_label = cached_proto_label
                    proto_conf = cached_proto_conf

                if motion_block:
                    pred_window.append("UNKNOWN")
                    pred_conf_window.append(0.0)
                    label = "UNKNOWN"
                    confidence = 0.0
                    source = "MOTION"
                elif mode == "rules":
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
                    if detection.hand_count > 0 and (fused_label is not None or ml_label is not None or deep_label is not None):
                        if not quality_ok:
                            pred_window.append("UNKNOWN")
                            confidence = max(ml_conf, deep_conf, fused_conf)
                            source = "QGATE"
                        elif proto_label is not None and proto_conf >= max(cfg.prototype_threshold, max(ml_conf, deep_conf) + 0.02):
                            pred_window.append(proto_label)
                            confidence = proto_conf
                            source = "PROTO"
                        elif fused_label is not None:
                            pred_window.append(fused_label)
                            confidence = fused_conf
                            source = fused_source
                        else:
                            pred_window.append("UNKNOWN")
                            confidence = max(ml_conf, deep_conf, fused_conf)
                            source = fused_source
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
                    custom_choice = _prefer_custom_label(
                        rule_label=rule_label,
                        rule_conf=rule_conf,
                        fused_label=fused_label,
                        fused_conf=fused_conf,
                        proto_label=proto_label,
                        proto_conf=proto_conf,
                        temporal_label=temporal_label,
                        temporal_conf=temporal_conf,
                    )
                    if detection.hand_count == 0:
                        pred_window.append("NO_HAND")
                        source = "NONE"
                    elif not quality_ok:
                        pred_window.append("UNKNOWN")
                        source = "QGATE"
                    elif custom_choice is not None:
                        custom_label, custom_conf, custom_src = custom_choice
                        pred_window.append(custom_label)
                        confidence = custom_conf
                        source = custom_src
                    elif cfg.strict_consensus:
                        cons = _strict_consensus_decision(
                            rule_label=rule_label,
                            rule_conf=rule_conf,
                            proto_label=proto_label,
                            proto_conf=proto_conf,
                            ml_label=fused_label,
                            ml_conf=fused_conf,
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
                        elif fused_label is not None:
                            pred_window.append(fused_label)
                            confidence = fused_conf
                            source = fused_source
                        elif ml_label is not None or deep_label is not None:
                            pred_window.append("UNKNOWN")
                            confidence = max(ml_conf, deep_conf, fused_conf)
                            source = fused_source
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
                    elif fused_label is not None:
                        pred_window.append(fused_label)
                        confidence = fused_conf
                        source = fused_source
                    elif ml_label is not None or deep_label is not None:
                        pred_window.append("UNKNOWN")
                        confidence = max(ml_conf, deep_conf, fused_conf)
                        source = fused_source
                    else:
                        pred_window.append("UNKNOWN")
                        source = "NONE"

                if (not motion_block) and detection.hand_count > 0 and pred_window:
                    cached_resolved_label = pred_window[-1]
                    cached_resolved_conf = confidence
                    cached_resolved_source = source

            if not motion_block and (not reuse_cached_prediction):
                pred_conf_window.append(max(0.0, min(1.0, confidence)))
            if pred_window and not motion_block:
                label, voted_conf = _weighted_label_vote(pred_window, pred_conf_window)
                confidence = max(confidence, voted_conf)

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
                if no_hand_streak == 1:
                    speaker.stop_current()
            else:
                no_hand_streak = 0

            # Retrigger same phrase after hand goes away for a while.
            if no_hand_streak >= 4:
                spoken_label = ""

            now_speak = time.time()
            if continuous_sentence:
                sentence_decoder.cfg.min_stable_frames = 1
                update = sentence_decoder.update(
                    label=label,
                    stable_hits=stable_hits,
                    hand_count=detection.hand_count,
                    now_ts=now_speak,
                    auto_speak_enabled=(voice_enabled and auto_speak),
                )
                if update.auto_spoken_text:
                    speaker.say_latest(update.auto_spoken_text)
                    last_spoken_sentence = update.auto_spoken_text

            can_repeat_same = (label == spoken_label) and ((now_speak - last_spoken_time) >= cfg.repeat_same_label_sec)
            if (
                voice_enabled
                and auto_speak
                and (not continuous_sentence)
                and detection.hand_count > 0
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
                infer_every = _tune_infer_interval(
                    infer_every=infer_every,
                    fps=fps,
                    perf_target=perf_target,
                    hand_count=detection.hand_count,
                    max_interval=4,
                )
                last_tune_ts = now

            if cfg.deep_auto_throttle and deep_bundle is not None:
                if fps < (perf_target - float(cfg.deep_disable_fps_drop)):
                    low_fps_streak += 1
                else:
                    low_fps_streak = 0
                if fps > (perf_target - float(cfg.deep_reenable_margin)):
                    high_fps_streak += 1
                else:
                    high_fps_streak = 0

                if deep_runtime_active and low_fps_streak >= max(1, int(cfg.deep_disable_streak)):
                    deep_runtime_active = False
                    low_fps_streak = 0
                    print("[INFO] Deep runtime auto-disabled to protect FPS.")
                elif (not deep_runtime_active) and high_fps_streak >= max(1, int(cfg.deep_reenable_streak)):
                    deep_runtime_active = True
                    high_fps_streak = 0
                    print("[INFO] Deep runtime auto-reenabled.")

            sentence_text = ""
            if show_sentence:
                sentence_text = sentence_decoder.text()
                if not sentence_text and last_spoken_sentence:
                    sentence_text = f"(last) {last_spoken_sentence}"
            if live_capture_enabled and not sentence_text:
                sentence_text = (
                    f"Teach {live_teach_label}: frames {live_saved_total + len(live_pending_records)}"
                    f" | clips {live_saved_sequences} | {live_capture_status}"
                )
            deep_flag = "D1" if deep_runtime_active and deep_bundle is not None else "D0"
            perf_text = f"trk {infer_every} | cls {max(1, int(cfg.prediction_interval))} | {deep_flag}"
            out = detection.frame
            if cfg.enhance_frame:
                out = _enhance_frame(out)
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
            now_hint = time.time()
            if (
                (now_hint - last_quality_hint_ts) >= max(0.10, float(cfg.quality_hint_interval_sec))
                or detection.hand_count != last_quality_hand_count
            ):
                last_quality_hint, last_quality_hint_color = _compute_quality_hint(
                    out,
                    detection.hand_count,
                    last_confidence,
                    last_label,
                )
                last_quality_hint_ts = now_hint
                last_quality_hand_count = detection.hand_count
            _draw_quality_hint(out, last_quality_hint, last_quality_hint_color)
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
                    sentence_decoder.cfg.min_stable_frames = 1
                    sentence_decoder.cfg.transition_cooldown_sec = min(0.10, cfg.sentence_append_cooldown_sec)
                else:
                    sentence_decoder.cfg.min_stable_frames = max(1, int(cfg.min_stable_frames_for_speech))
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
                    pred_conf_window.clear()
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
                sentence_decoder.clear()
            if key == 32 and label not in {"NO_HAND", "UNKNOWN"}:  # space
                sentence_decoder.append_manual(label, now_ts=time.time())
            spoken_now = ""
            if key == 13:
                spoken_now = sentence_decoder.speak_now(now_ts=time.time())
            if spoken_now:
                speaker.say_latest(spoken_now)
                last_spoken_sentence = spoken_now
            if ch == "u":
                _run_live_retrain(force=True)
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
            "live_teach_label": live_teach_label,
            "live_teach_frames_saved": int(live_saved_total + len(live_pending_records)),
            "live_teach_sequences_saved": int(live_saved_sequences),
        }
        summary_path = cfg.session_log_path.parent / "session_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Session summary saved: {summary_path}")

        if live_capture_enabled:
            _flush_live_sequence("shutdown")
            flushed = _flush_live_records()
            if flushed:
                print(f"[INFO] Final live teach flush: {flushed} samples")

        if tracker_worker is not None:
            tracker_worker.close()
        if async_tracker is not None:
            async_tracker.close()
        tracker.close()
        speaker.close()
        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
