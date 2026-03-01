from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import (
    DEFAULT_LABELS_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROTOTYPE_DB_PATH,
    DEFAULT_TEMPORAL_LABELS_PATH,
    DEFAULT_TEMPORAL_METADATA_PATH,
    DEFAULT_TEMPORAL_MODEL_PATH,
)
from .feature_extraction import normalize_features
from .hand_tracking import HandTracker
from .language import sentence_to_text
from .modeling import load_model
from .rules import RuleBasedInterpreter
from .temporal_model import load_temporal_model
from .prototype_adapt import load_prototype_db, predict_prototype


@dataclass
class VideoInferConfig:
    input_video: Path
    out_json: Path
    mode: str = "hybrid"  # rules | ml | temporal | hybrid
    confidence_threshold: float = 0.60
    rule_confidence_threshold: float = 0.78
    temporal_confidence_threshold: float = 0.60
    smoothing_window: int = 7
    infer_interval: int = 1
    infer_scale: float = 0.75
    model_path: Path = DEFAULT_MODEL_PATH
    labels_path: Path = DEFAULT_LABELS_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    prototype_db_path: Path = DEFAULT_PROTOTYPE_DB_PATH
    temporal_model_path: Path = DEFAULT_TEMPORAL_MODEL_PATH
    temporal_labels_path: Path = DEFAULT_TEMPORAL_LABELS_PATH
    temporal_metadata_path: Path = DEFAULT_TEMPORAL_METADATA_PATH
    ml_min_margin: float = 0.08
    use_prototypes: bool = True
    prototype_threshold: float = 0.84
    prototype_margin: float = 0.03


def compress_labels(labels: list[str]) -> list[str]:
    out: list[str] = []
    for label in labels:
        if label in {"NO_HAND", "UNKNOWN"}:
            continue
        if out and out[-1] == label:
            continue
        out.append(label)
    return out


def run_video_inference(cfg: VideoInferConfig) -> Path:
    if not cfg.input_video.exists():
        raise FileNotFoundError(f"Input video not found: {cfg.input_video}")

    mode = cfg.mode.lower().strip()
    if mode not in {"rules", "ml", "temporal", "hybrid"}:
        mode = "hybrid"

    model = None
    ml_label_thresholds: dict[str, float] = {}
    prototype_db = None
    if mode in {"ml", "hybrid"}:
        try:
            model, _ = load_model(cfg.model_path, cfg.labels_path)
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
        except Exception:
            if mode == "ml":
                mode = "rules"

    temporal_model = None
    temporal_seq_len = 24
    if mode in {"temporal", "hybrid"}:
        try:
            temporal_model, _, temporal_seq_len = load_temporal_model(
                cfg.temporal_model_path,
                cfg.temporal_labels_path,
                cfg.temporal_metadata_path,
            )
        except Exception:
            if mode == "temporal":
                mode = "rules"

    if cfg.use_prototypes:
        try:
            prototype_db = load_prototype_db(cfg.prototype_db_path)
        except Exception:
            prototype_db = None

    cap = cv2.VideoCapture(str(cfg.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {cfg.input_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    tracker = HandTracker(max_num_hands=2, inference_scale=cfg.infer_scale)
    rules = RuleBasedInterpreter()

    pred_window = deque(maxlen=max(1, cfg.smoothing_window))
    seq_buffer = deque(maxlen=max(4, temporal_seq_len))

    frame_idx = 0
    labels_stream: list[str] = []
    events: list[dict] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if cfg.infer_interval > 1 and (frame_idx % cfg.infer_interval) != 0:
                continue

            detection = tracker.process(frame, draw=False)
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
                rp = rules.predict(detection)
                if rp is not None:
                    rule_label = rp.label
                    rule_conf = rp.confidence

            ml_label: Optional[str] = None
            ml_conf = 0.0
            ml_margin = 0.0
            if mode in {"ml", "hybrid"} and model is not None and detection.hand_count > 0:
                probs = model.predict_proba([features])[0]
                top_idx = np.argsort(probs)[::-1]
                i = int(top_idx[0])
                j2 = int(top_idx[1]) if len(top_idx) > 1 else i
                ml_label = str(model.classes_[i])
                ml_conf = float(probs[i])
                ml_margin = float(probs[i] - probs[j2]) if len(top_idx) > 1 else 1.0

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

            temp_label: Optional[str] = None
            temp_conf = 0.0
            if mode in {"temporal", "hybrid"} and temporal_model is not None and len(seq_buffer) >= temporal_seq_len:
                seq = np.asarray(list(seq_buffer)[-temporal_seq_len:], dtype=np.float32).reshape(1, -1)
                probs_t = temporal_model.predict_proba(seq)[0]
                j = int(np.argmax(probs_t))
                classes_t = list(getattr(temporal_model, "classes_", []))
                temp_label = str(classes_t[j]) if classes_t else None
                temp_conf = float(probs_t[j])

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
                ml_threshold = max(
                    cfg.confidence_threshold,
                    float(ml_label_thresholds.get(str(ml_label), cfg.confidence_threshold)) if ml_label else cfg.confidence_threshold,
                )
                if detection.hand_count == 0:
                    pred_window.append("NO_HAND")
                elif proto_label is not None and proto_conf >= max(cfg.prototype_threshold, ml_conf + 0.02):
                    pred_window.append(proto_label)
                    confidence = proto_conf
                    source = "PROTO"
                elif ml_label is not None and ml_conf >= ml_threshold and ml_margin >= cfg.ml_min_margin:
                    pred_window.append(ml_label)
                    confidence = ml_conf
                    source = "ML"
                else:
                    pred_window.append("UNKNOWN")
                    confidence = ml_conf
                    source = "ML"
            elif mode == "temporal":
                if detection.hand_count == 0:
                    pred_window.append("NO_HAND")
                elif temp_label is not None and temp_conf >= cfg.temporal_confidence_threshold:
                    pred_window.append(temp_label)
                    confidence = temp_conf
                    source = "TEMP"
                else:
                    pred_window.append("UNKNOWN")
                    confidence = temp_conf
                    source = "TEMP"
            else:
                if detection.hand_count == 0:
                    pred_window.append("NO_HAND")
                elif rule_label is not None and rule_conf >= cfg.rule_confidence_threshold:
                    pred_window.append(rule_label)
                    confidence = rule_conf
                    source = "RULE"
                elif temp_label is not None and temp_conf >= cfg.temporal_confidence_threshold:
                    pred_window.append(temp_label)
                    confidence = temp_conf
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
                        confidence = ml_conf
                        source = "ML"
                    else:
                        pred_window.append("UNKNOWN")
                else:
                    pred_window.append("UNKNOWN")

            if pred_window:
                label = Counter(pred_window).most_common(1)[0][0]
            labels_stream.append(label)
            if label not in {"NO_HAND", "UNKNOWN"}:
                events.append(
                    {
                        "frame": frame_idx,
                        "time_sec": round(frame_idx / max(fps, 1e-6), 3),
                        "label": label,
                        "confidence": round(float(confidence), 4),
                        "source": source,
                    }
                )
    finally:
        tracker.close()
        cap.release()

    tokens = compress_labels(labels_stream)
    transcript = sentence_to_text(tokens)

    payload = {
        "input_video": str(cfg.input_video),
        "mode": mode,
        "fps": fps,
        "num_frames_processed": frame_idx,
        "num_events": len(events),
        "tokens": tokens,
        "transcript": transcript,
        "events": events,
    }
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cfg.out_json
