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


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass
class BuildVideoDatasetConfig:
    root_dir: Path
    out_csv: Path
    max_videos_per_class: int = 0
    max_frames_per_video: int = 0
    frame_stride: int = 3
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.5


def _iter_class_dirs(root_dir: Path) -> Iterable[Path]:
    for p in sorted(root_dir.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            yield p


def _video_count(dir_path: Path, limit: int = 2000) -> int:
    count = 0
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            count += 1
            if count >= limit:
                return count
    return count


def resolve_video_class_root(root_dir: Path, max_depth: int = 4) -> Path:
    root_dir = root_dir.resolve()
    best_root = root_dir
    best_score: tuple[int, int] = (-1, -1)  # (classes_with_videos, total_videos)

    candidates: list[Path] = [root_dir]
    for d in root_dir.rglob("*"):
        if not d.is_dir():
            continue
        depth = len(d.relative_to(root_dir).parts)
        if depth <= max_depth:
            candidates.append(d)

    for cand in candidates:
        class_dirs = [p for p in cand.iterdir() if p.is_dir() and not p.name.startswith(".")]
        if len(class_dirs) < 2:
            continue
        counts = [_video_count(cd, limit=500) for cd in class_dirs]
        classes_with_videos = sum(1 for c in counts if c > 0)
        if classes_with_videos < 2:
            continue
        score = (classes_with_videos, int(sum(counts)))
        if score > best_score:
            best_score = score
            best_root = cand

    return best_root


def _iter_videos(class_dir: Path) -> Iterable[Path]:
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def _empty_features() -> np.ndarray:
    return np.zeros((FEATURE_SIZE,), dtype=np.float32)


def _hand_features(hand_landmarks) -> np.ndarray:
    values = []
    for lm in hand_landmarks.landmark:
        values.extend([lm.x, lm.y, lm.z])
    return np.asarray(values, dtype=np.float32)


def _extract_features(frame_bgr: np.ndarray, hands) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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


def build_dataset_from_videos(cfg: BuildVideoDatasetConfig) -> tuple[int, int, int]:
    if not cfg.root_dir.exists():
        raise FileNotFoundError(f"Video dataset root not found: {cfg.root_dir}")
    class_root = resolve_video_class_root(cfg.root_dir)
    if class_root != cfg.root_dir.resolve():
        print(f"[INFO] Auto-detected video class root: {class_root}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        model_complexity=0,
    )

    records: list[tuple[np.ndarray, str]] = []
    videos_processed = 0
    frames_processed = 0

    try:
        for class_dir in _iter_class_dirs(class_root):
            class_name = class_dir.name
            used_videos = 0
            for video_path in _iter_videos(class_dir):
                if cfg.max_videos_per_class > 0 and used_videos >= cfg.max_videos_per_class:
                    break
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    continue

                used_videos += 1
                videos_processed += 1
                frame_idx = 0
                saved_in_video = 0
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame_idx += 1
                        if cfg.frame_stride > 1 and (frame_idx % cfg.frame_stride) != 0:
                            continue

                        feats = _extract_features(frame, hands)
                        frames_processed += 1
                        if np.allclose(feats, 0.0):
                            continue

                        records.append((feats, class_name))
                        saved_in_video += 1
                        if cfg.max_frames_per_video > 0 and saved_in_video >= cfg.max_frames_per_video:
                            break
                finally:
                    cap.release()
    finally:
        hands.close()

    saved = save_records(records, cfg.out_csv)
    return videos_processed, frames_processed, saved
