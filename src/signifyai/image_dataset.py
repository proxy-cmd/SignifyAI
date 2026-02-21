from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

from .config import FEATURE_SIZE, LANDMARKS_PER_HAND, MAX_HANDS
from .dataset import save_records
from .feature_extraction import normalize_features


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class BuildImageDatasetConfig:
    root_dir: Path
    out_csv: Path
    max_images_per_class: int = 0
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.5


def _iter_class_dirs(root_dir: Path) -> Iterable[Path]:
    for p in sorted(root_dir.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            yield p


def _iter_images(class_dir: Path) -> Iterable[Path]:
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _empty_features() -> np.ndarray:
    return np.zeros((FEATURE_SIZE,), dtype=np.float32)


def _hand_features(hand_landmarks) -> np.ndarray:
    values = []
    for lm in hand_landmarks.landmark:
        values.extend([lm.x, lm.y, lm.z])
    return np.asarray(values, dtype=np.float32)


def _extract_features(image_bgr: np.ndarray, hands) -> np.ndarray:
    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    features = _empty_features()

    if not results.multi_hand_landmarks:
        return features

    slot_to_features: dict[int, np.ndarray] = {}
    handedness = results.multi_handedness or []

    for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:MAX_HANDS]):
        slot = i
        if i < len(handedness):
            label = handedness[i].classification[0].label.lower()
            slot = 0 if label == "left" else 1
        slot_to_features[slot] = _hand_features(hand_landmarks)

    hand_size = LANDMARKS_PER_HAND * 3
    for slot in range(MAX_HANDS):
        start = slot * hand_size
        end = start + hand_size
        features[start:end] = slot_to_features.get(slot, np.zeros((hand_size,), dtype=np.float32))

    return normalize_features(features)


def build_dataset_from_images(cfg: BuildImageDatasetConfig) -> tuple[int, int]:
    if not cfg.root_dir.exists():
        raise FileNotFoundError(f"Image dataset root not found: {cfg.root_dir}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        model_complexity=1,
    )

    records: list[tuple[np.ndarray, str]] = []
    total_images = 0

    try:
        for class_dir in _iter_class_dirs(cfg.root_dir):
            class_name = class_dir.name
            used = 0
            for img_path in _iter_images(class_dir):
                if cfg.max_images_per_class > 0 and used >= cfg.max_images_per_class:
                    break

                total_images += 1
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                feats = _extract_features(img, hands)
                # Keep only usable detections.
                if np.allclose(feats, 0.0):
                    continue

                records.append((feats, class_name))
                used += 1
    finally:
        hands.close()

    saved = save_records(records, cfg.out_csv)
    return total_images, saved
