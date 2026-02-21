from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .config import FEATURE_SIZE, LANDMARKS_PER_HAND, MAX_HANDS


@dataclass
class DetectionResult:
    features: np.ndarray
    hand_count: int
    frame: np.ndarray


class HandTracker:
    """MediaPipe wrapper that returns a fixed-size feature vector for up to two hands."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        model_complexity: int = 1,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self.hands.close()

    def _empty_features(self) -> np.ndarray:
        return np.zeros((FEATURE_SIZE,), dtype=np.float32)

    def _hand_features(self, hand_landmarks) -> np.ndarray:
        values = []
        for lm in hand_landmarks.landmark:
            values.extend([lm.x, lm.y, lm.z])
        return np.asarray(values, dtype=np.float32)

    def process(self, frame_bgr: np.ndarray, draw: bool = True) -> DetectionResult:
        frame = frame_bgr.copy()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        all_features = self._empty_features()
        if not results.multi_hand_landmarks:
            return DetectionResult(features=all_features, hand_count=0, frame=frame)

        hand_count = min(len(results.multi_hand_landmarks), MAX_HANDS)

        # Keep a stable left/right order when handedness is available.
        slot_to_features: dict[int, np.ndarray] = {}
        handedness = results.multi_handedness or []

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:MAX_HANDS]):
            slot = i
            if i < len(handedness):
                label = handedness[i].classification[0].label.lower()
                slot = 0 if label == "left" else 1

            slot_to_features[slot] = self._hand_features(hand_landmarks)

            if draw:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

        for slot in range(MAX_HANDS):
            start = slot * LANDMARKS_PER_HAND * 3
            end = start + LANDMARKS_PER_HAND * 3
            all_features[start:end] = slot_to_features.get(
                slot,
                np.zeros((LANDMARKS_PER_HAND * 3,), dtype=np.float32),
            )

        return DetectionResult(features=all_features, hand_count=hand_count, frame=frame)


def open_camera(index: int = 0, width: int = 960, height: int = 720) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def warmup_camera(cap: cv2.VideoCapture, frames: int = 10) -> None:
    for _ in range(frames):
        cap.read()


def check_camera(cap: cv2.VideoCapture) -> Optional[str]:
    if not cap.isOpened():
        return "Failed to open camera."
    return None
