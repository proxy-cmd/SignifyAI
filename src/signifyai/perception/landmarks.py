from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from importlib import import_module

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from ..contracts import LandmarkFrame


@dataclass
class PerceptionConfig:
    model_path: Path = Path("data/models/hand_landmarker.task")
    max_hands: int = 2
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.6
    inference_scale: float = 0.65


def ensure_hand_model(path: Path) -> Path:
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    request_mod = import_module("urllib.request")
    request_mod.urlretrieve(url, str(path))
    return path


def to_landmark_array(landmarks, limit: int | None = None) -> np.ndarray | None:
    if landmarks is None:
        return None

    points = [[point.x, point.y, point.z] for point in landmarks]
    if limit is not None:
        points = points[:limit]
    return np.asarray(points, dtype=np.float32)


def frame_quality(frame: np.ndarray, left: np.ndarray | None, right: np.ndarray | None) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hand_area = 0.0
    for hand in (left, right):
        if hand is None:
            continue
        xs = hand[:, 0]
        ys = hand[:, 1]
        area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
        hand_area = max(hand_area, area)

    return {"brightness": brightness, "blur": blur, "hand_area": hand_area}


class MultiModalPerceptor:
    def __init__(self, cfg: PerceptionConfig) -> None:
        self.cfg = cfg
        model_path = ensure_hand_model(cfg.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self.model = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self.model.close()

    def process(self, frame_bgr: np.ndarray) -> LandmarkFrame:
        model_frame = frame_bgr
        if self.cfg.inference_scale < 0.999:
            model_frame = cv2.resize(
                frame_bgr,
                None,
                fx=self.cfg.inference_scale,
                fy=self.cfg.inference_scale,
                interpolation=cv2.INTER_LINEAR,
            )

        rgb = cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.model.detect(mp_img)

        left = None
        right = None
        if len(result.hand_landmarks) > 0:
            left = to_landmark_array(result.hand_landmarks[0], limit=21)
        if len(result.hand_landmarks) > 1:
            right = to_landmark_array(result.hand_landmarks[1], limit=21)

        hand_count = int(left is not None) + int(right is not None)
        quality = frame_quality(frame_bgr, left, right)

        return LandmarkFrame(
            timestamp_ms=int(time.time() * 1000),
            hand_count=hand_count,
            left_hand=left,
            right_hand=right,
            face=None,
            pose=None,
            quality=quality,
        )
