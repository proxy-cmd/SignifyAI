from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from google.protobuf import message_factory, symbol_database

from ..contracts import LandmarkFrame


def _patch_protobuf() -> None:
    if not hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
        def _symbol_get_prototype(self, descriptor):
            return message_factory.GetMessageClass(descriptor)

        symbol_database.SymbolDatabase.GetPrototype = _symbol_get_prototype  # type: ignore[attr-defined]

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):
        def _factory_get_prototype(self, descriptor):
            return message_factory.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _factory_get_prototype  # type: ignore[attr-defined]


_patch_protobuf()
import mediapipe as mp


@dataclass
class PerceptionConfig:
    model_complexity: int = 0
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.6
    inference_scale: float = 0.65


class MultiModalPerceptor:
    def __init__(self, cfg: PerceptionConfig) -> None:
        self.cfg = cfg
        self.model = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=cfg.model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def close(self) -> None:
        self.model.close()

    @staticmethod
    def _landmark_array(landmarks, limit: int | None = None) -> np.ndarray | None:
        if landmarks is None:
            return None
        points = [[p.x, p.y, p.z] for p in landmarks.landmark]
        if limit is not None:
            points = points[:limit]
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def _quality(frame: np.ndarray, left: np.ndarray | None, right: np.ndarray | None) -> dict[str, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hand_area = 0.0
        for hand in [left, right]:
            if hand is None:
                continue
            xs = hand[:, 0]
            ys = hand[:, 1]
            area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
            hand_area = max(hand_area, area)
        return {"brightness": brightness, "blur": blur, "hand_area": hand_area}

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
        result = self.model.process(rgb)

        left = self._landmark_array(result.left_hand_landmarks)
        right = self._landmark_array(result.right_hand_landmarks)
        face = self._landmark_array(result.face_landmarks, limit=20)
        pose = self._landmark_array(result.pose_landmarks, limit=12)
        hand_count = int(left is not None) + int(right is not None)
        quality = self._quality(frame_bgr, left, right)

        return LandmarkFrame(
            timestamp_ms=int(time.time() * 1000),
            hand_count=hand_count,
            left_hand=left,
            right_hand=right,
            face=face,
            pose=pose,
            quality=quality,
        )
