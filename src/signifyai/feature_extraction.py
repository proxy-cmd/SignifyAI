from __future__ import annotations

import numpy as np

from .config import LANDMARKS_PER_HAND, MAX_HANDS


def _normalize_single_hand(hand_flat: np.ndarray) -> np.ndarray:
    """Normalize one hand (63 values) translation + scale invariant."""
    if np.allclose(hand_flat, 0.0):
        return hand_flat

    pts = hand_flat.reshape(LANDMARKS_PER_HAND, 3).copy()
    wrist = pts[0].copy()
    pts -= wrist

    dists = np.linalg.norm(pts[:, :2], axis=1)
    scale = float(np.max(dists))
    if scale > 1e-6:
        pts /= scale

    return pts.flatten()


def normalize_features(features: np.ndarray) -> np.ndarray:
    if features.ndim != 1:
        raise ValueError("Expected a 1D feature vector")

    hand_size = LANDMARKS_PER_HAND * 3
    expected = hand_size * MAX_HANDS
    if features.size != expected:
        raise ValueError(f"Expected {expected} features, got {features.size}")

    normalized = np.zeros_like(features, dtype=np.float32)
    for i in range(MAX_HANDS):
        start = i * hand_size
        end = start + hand_size
        normalized[start:end] = _normalize_single_hand(features[start:end])

    return normalized
