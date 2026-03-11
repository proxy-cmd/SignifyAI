from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LandmarkFrame:
    timestamp_ms: int
    hand_count: int
    left_hand: np.ndarray | None
    right_hand: np.ndarray | None
    face: np.ndarray | None
    pose: np.ndarray | None
    quality: dict[str, float] = field(default_factory=dict)


@dataclass
class SequenceWindow:
    frames: list[LandmarkFrame]

    def to_feature_matrix(self) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for frame in self.frames:
            vectors.append(flatten_landmark_frame(frame))
        return np.stack(vectors, axis=0).astype(np.float32)


@dataclass
class PredictionOutput:
    intent_id: str
    intent_text: str
    confidence: float
    stability_state: str
    timestamp_ms: int
    latency_ms: float
    source_model_version: str
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionClip:
    session_id: str
    clip_id: str
    intent_id: str
    signer_id: str
    sequence: np.ndarray
    quality: dict[str, float]
    consent_raw_video: bool = False
    raw_video_path: str | None = None


def flatten_landmark_frame(frame: LandmarkFrame) -> np.ndarray:
    def _flat(arr: np.ndarray | None, size: int) -> np.ndarray:
        if arr is None:
            return np.zeros((size,), dtype=np.float32)
        x = arr.astype(np.float32).reshape(-1)
        if x.size >= size:
            return x[:size]
        out = np.zeros((size,), dtype=np.float32)
        out[: x.size] = x
        return out

    left = _flat(frame.left_hand, 21 * 3)
    right = _flat(frame.right_hand, 21 * 3)
    face = _flat(frame.face, 20 * 3)
    pose = _flat(frame.pose, 12 * 3)
    quality = np.asarray([
        float(frame.quality.get("brightness", 0.0)),
        float(frame.quality.get("blur", 0.0)),
        float(frame.quality.get("hand_area", 0.0)),
    ], dtype=np.float32)
    return np.concatenate([left, right, face, pose, quality], axis=0)
