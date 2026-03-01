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
    auto_mode: bool = True
    capture_interval_sec: float = 0.35
    min_hand_frames: int = 2
    min_feature_delta: float = 0.010
    flush_every: int = 20


def run_collection(cfg: CollectConfig) -> int:
    cap = open_camera(index=cfg.camera_index, width=cfg.width, height=cfg.height)
    err = check_camera(cap)
    if err:
        raise RuntimeError(err)

    warmup_camera(cap)
    tracker = HandTracker(max_num_hands=2)
    pending_records: list[tuple[np.ndarray, str]] = []
    saved_total = 0

    window_name = f"Collect: {cfg.label}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Instructions:")
    print("- Press 'c' to capture one sample now")
    print("- Press 'a' to toggle auto capture mode")
    print("- Press 'q' to quit")
    print(f"- Target samples: {cfg.samples}")

    auto_mode = bool(cfg.auto_mode)
    hand_streak = 0
    last_capture_ts = 0.0
    last_saved: np.ndarray | None = None
    status_text = "Waiting for hand..."

    def _try_capture(feats: np.ndarray, now_ts: float, reason: str) -> bool:
        nonlocal saved_total, pending_records, last_capture_ts, last_saved, status_text
        if (now_ts - last_capture_ts) < max(0.05, cfg.capture_interval_sec):
            return False
        if last_saved is not None:
            delta = float(np.mean(np.abs(feats - last_saved)))
            if delta < cfg.min_feature_delta:
                status_text = f"Skip duplicate ({delta:.4f})"
                return False
        pending_records.append((feats, cfg.label))
        last_saved = feats.copy()
        last_capture_ts = now_ts
        status_text = f"Captured ({reason})"
        print(f"Captured {saved_total + len(pending_records)}/{cfg.samples} [{reason}]")
        if len(pending_records) >= max(1, cfg.flush_every):
            saved_now = save_records(pending_records, cfg.out_csv)
            saved_total += saved_now
            pending_records = []
            print(f"Flushed to CSV. Total saved: {saved_total}")
        return True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            result = tracker.process(frame, draw=True)
            show = result.frame

            now_ts = cv2.getTickCount() / cv2.getTickFrequency()
            if result.hand_count > 0:
                hand_streak += 1
            else:
                hand_streak = 0

            if auto_mode and result.hand_count > 0 and hand_streak >= max(1, cfg.min_hand_frames):
                feats_auto = normalize_features(result.features)
                _try_capture(feats_auto, now_ts, reason="auto")

            cv2.putText(show, f"Label: {cfg.label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(show, f"Hands: {result.hand_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(
                show,
                f"Captured: {saved_total + len(pending_records)}/{cfg.samples} | Mode: {'AUTO' if auto_mode else 'MANUAL'}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2,
            )
            cv2.putText(show, status_text, (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 220, 220), 2)

            cv2.imshow(window_name, show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                # Only save when at least one hand is visible.
                if result.hand_count > 0:
                    feats = normalize_features(result.features)
                    _try_capture(feats, now_ts, reason="manual")
                else:
                    status_text = "No hand visible for manual capture"
            if key == ord("a"):
                auto_mode = not auto_mode
                status_text = f"Mode -> {'AUTO' if auto_mode else 'MANUAL'}"

            if (saved_total + len(pending_records)) >= cfg.samples:
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    if pending_records:
        saved_total += save_records(pending_records, cfg.out_csv)
    print(f"Saved {saved_total} samples to {cfg.out_csv}")
    return saved_total
