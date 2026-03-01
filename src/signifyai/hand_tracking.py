from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from absl import logging as absl_logging
from google.protobuf import message_factory, symbol_database

from .config import FEATURE_SIZE, LANDMARKS_PER_HAND, MAX_HANDS

def _patch_protobuf_for_mediapipe() -> None:
    """
    MediaPipe versions in this project still call protobuf GetPrototype().
    New protobuf versions removed it, so we provide a compatible shim.
    """
    if not hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
        def _symbol_get_prototype(self, descriptor):
            return message_factory.GetMessageClass(descriptor)
        symbol_database.SymbolDatabase.GetPrototype = _symbol_get_prototype  # type: ignore[attr-defined]

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):
        def _factory_get_prototype(self, descriptor):
            return message_factory.GetMessageClass(descriptor)
        message_factory.MessageFactory.GetPrototype = _factory_get_prototype  # type: ignore[attr-defined]


_patch_protobuf_for_mediapipe()
import mediapipe as mp

absl_logging.set_verbosity(absl_logging.ERROR)


@dataclass
class DetectionResult:
    features: np.ndarray
    hand_count: int
    frame: np.ndarray
    raw_hands: list[np.ndarray]
    handedness: list[str]


class HandTracker:
    """MediaPipe wrapper that returns a fixed-size feature vector for up to two hands."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.72,
        min_tracking_confidence: float = 0.68,
        model_complexity: int = 0,
        inference_scale: float = 0.75,
        landmark_smoothing: float = 0.78,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.inference_scale = max(0.4, min(1.0, float(inference_scale)))
        self.landmark_smoothing = max(0.0, min(0.95, float(landmark_smoothing)))
        self._prev_slot_raw: dict[int, np.ndarray] = {}
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

    def _bbox_from_hand(self, hand_landmarks) -> tuple[float, float, float, float]:
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def process(self, frame_bgr: np.ndarray, draw: bool = True) -> DetectionResult:
        frame = frame_bgr.copy()
        infer_bgr = frame_bgr
        if self.inference_scale < 0.999:
            infer_bgr = cv2.resize(
                frame_bgr,
                None,
                fx=self.inference_scale,
                fy=self.inference_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        all_features = self._empty_features()
        if not results.multi_hand_landmarks:
            self._prev_slot_raw.clear()
            return DetectionResult(features=all_features, hand_count=0, frame=frame, raw_hands=[], handedness=[])

        # Build candidates and filter tiny ghost detections.
        candidates = []
        handedness_list = results.multi_handedness or []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            bbox = self._bbox_from_hand(hand_landmarks)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < 0.010:
                continue
            center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
            hand_label = "unknown"
            if i < len(handedness_list):
                hand_label = handedness_list[i].classification[0].label.lower()
            candidates.append(
                {
                    "landmarks": hand_landmarks,
                    "bbox": bbox,
                    "center": center,
                    "area": area,
                    "label": hand_label,
                }
            )

        # Deduplicate overlapping detections that likely belong to one hand.
        selected = []
        for cand in sorted(candidates, key=lambda x: x["area"], reverse=True):
            duplicate = False
            for keep in selected:
                overlap = self._iou(cand["bbox"], keep["bbox"])
                same_side = cand["label"] == keep["label"] and cand["label"] != "unknown"
                cx0, cy0 = cand["center"]
                cx1, cy1 = keep["center"]
                center_dist = float(np.hypot(cx0 - cx1, cy0 - cy1))
                area_ratio = float(cand["area"] / max(keep["area"], 1e-6))
                similar_size = 0.45 <= area_ratio <= 2.2
                likely_same_hand = (overlap > 0.30) or (center_dist < 0.11 and similar_size)
                if likely_same_hand or (same_side and overlap > 0.20):
                    duplicate = True
                    break
            if not duplicate:
                selected.append(cand)
            if len(selected) >= MAX_HANDS:
                break

        # Keep a stable left/right order when handedness is available.
        slot_to_features: dict[int, np.ndarray] = {}
        slot_to_raw: dict[int, np.ndarray] = {}
        slot_to_label: dict[int, str] = {}

        for i, cand in enumerate(selected):
            hand_landmarks = cand["landmarks"]
            slot = i
            label = cand["label"]
            if label in {"left", "right"}:
                slot = 0 if label == "left" else 1

            hand_feat = self._hand_features(hand_landmarks)
            raw = hand_feat.reshape(LANDMARKS_PER_HAND, 3)
            if slot in self._prev_slot_raw:
                prev = self._prev_slot_raw[slot]
                raw = (self.landmark_smoothing * prev) + ((1.0 - self.landmark_smoothing) * raw)
            slot_to_raw[slot] = raw
            slot_to_features[slot] = raw.flatten()
            slot_to_label[slot] = label

            if draw:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        for slot in range(MAX_HANDS):
            start = slot * LANDMARKS_PER_HAND * 3
            end = start + LANDMARKS_PER_HAND * 3
            all_features[start:end] = slot_to_features.get(
                slot,
                np.zeros((LANDMARKS_PER_HAND * 3,), dtype=np.float32),
            )

        ordered_slots = [s for s in range(MAX_HANDS) if s in slot_to_raw]
        raw_hands = [slot_to_raw[s] for s in ordered_slots]
        hand_labels = [slot_to_label.get(s, "unknown") for s in ordered_slots]
        hand_count = len(raw_hands)
        self._prev_slot_raw = {s: slot_to_raw[s].copy() for s in ordered_slots}

        return DetectionResult(
            features=all_features,
            hand_count=hand_count,
            frame=frame,
            raw_hands=raw_hands,
            handedness=hand_labels,
        )


def open_camera(index: int = 0, width: int = 1280, height: int = 720, fps: int = 60) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    # Lower latency and avoid frame queue buildup on slower CPUs.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def warmup_camera(cap: cv2.VideoCapture, frames: int = 10) -> None:
    for _ in range(frames):
        cap.read()


def check_camera(cap: cv2.VideoCapture) -> Optional[str]:
    if not cap.isOpened():
        return "Failed to open camera."
    return None
