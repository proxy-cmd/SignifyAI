from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_DATASET_PATH
from .dataset import save_records
from .feature_extraction import normalize_features
from .hand_tracking import HandTracker, check_camera, open_camera, warmup_camera


@dataclass
class CollectConfig:
    label: str
    samples: int = 300
    camera_index: int = 0
    width: int = 960
    height: int = 720
    out_csv: Path = DEFAULT_DATASET_PATH


def run_collection(cfg: CollectConfig) -> int:
    cap = open_camera(index=cfg.camera_index, width=cfg.width, height=cfg.height)
    err = check_camera(cap)
    if err:
        raise RuntimeError(err)

    warmup_camera(cap)
    tracker = HandTracker(max_num_hands=2)
    records: list[tuple[np.ndarray, str]] = []

    window_name = f"Collect: {cfg.label}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Instructions:")
    print("- Press 'c' to capture one sample")
    print("- Press 'q' to quit")
    print(f"- Target samples: {cfg.samples}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            result = tracker.process(frame, draw=True)
            show = result.frame

            cv2.putText(show, f"Label: {cfg.label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(show, f"Hands: {result.hand_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(show, f"Captured: {len(records)}/{cfg.samples}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow(window_name, show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                # Only save when at least one hand is visible.
                if result.hand_count > 0:
                    feats = normalize_features(result.features)
                    records.append((feats, cfg.label))
                    print(f"Captured {len(records)}/{cfg.samples}")

            if len(records) >= cfg.samples:
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    saved = save_records(records, cfg.out_csv)
    print(f"Saved {saved} samples to {cfg.out_csv}")
    return saved
