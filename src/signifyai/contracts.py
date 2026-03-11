from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

HAND_VEC_SIZE = 21 * 3
FACE_VEC_SIZE = 20 * 3
POSE_VEC_SIZE = 12 * 3
QUALITY_VEC_SIZE = 3


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
        rows = [flatten_frame(frame) for frame in self.frames]
        return np.stack(rows, axis=0).astype(np.float32)


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


def pad_vec(arr: np.ndarray | None, size: int) -> np.ndarray:
    if arr is None:
        return np.zeros((size,), dtype=np.float32)

    flat = arr.astype(np.float32).reshape(-1)
    if flat.size >= size:
        return flat[:size]

    out = np.zeros((size,), dtype=np.float32)
    out[: flat.size] = flat
    return out


def flatten_frame(frame: LandmarkFrame) -> np.ndarray:
    left = pad_vec(frame.left_hand, HAND_VEC_SIZE)
    right = pad_vec(frame.right_hand, HAND_VEC_SIZE)
    face = pad_vec(frame.face, FACE_VEC_SIZE)
    pose = pad_vec(frame.pose, POSE_VEC_SIZE)
    quality_vec = np.asarray(
        [
            float(frame.quality.get("brightness", 0.0)),
            float(frame.quality.get("blur", 0.0)),
            float(frame.quality.get("hand_area", 0.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([left, right, face, pose, quality_vec], axis=0)


# Backward-compatible helper name.
flatten_landmark_frame = flatten_frame
