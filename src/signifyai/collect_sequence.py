from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_SEQUENCE_DATASET_PATH, FEATURE_SIZE
from .feature_extraction import normalize_features
from .hand_tracking import HandTracker, check_camera, open_camera, warmup_camera
from .sequence_dataset import append_sequence_records


@dataclass
class CollectSequenceConfig:
    label: str
    clips: int = 120
    seq_len: int = 24
    min_visible_frames: int = 14
    auto_mode: bool = True
    clip_gap_sec: float = 1.2
    camera_index: int = 0
    width: int = 960
    height: int = 720
    out_npz: Path = DEFAULT_SEQUENCE_DATASET_PATH
    flush_every: int = 8


def run_sequence_collection(cfg: CollectSequenceConfig) -> int:
    cap = open_camera(index=cfg.camera_index, width=cfg.width, height=cfg.height)
    err = check_camera(cap)
    if err:
        raise RuntimeError(err)

    warmup_camera(cap)
    tracker = HandTracker(max_num_hands=2)
    buffer_records: list[tuple[np.ndarray, str]] = []
    saved_total = 0

    window_name = f"Collect Seq: {cfg.label}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Sequence collection instructions:")
    print("- Press 'c' to start recording one clip")
    print("- Press 'a' to toggle auto mode")
    print("- Hold gesture steady until clip completes")
    print("- Press 'q' to quit")
    print(f"- Target clips: {cfg.clips}, sequence length: {cfg.seq_len}")

    recording = False
    auto_mode = cfg.auto_mode
    clip_feats: list[np.ndarray] = []
    visible_frames = 0
    next_auto_start = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            result = tracker.process(frame, draw=True)
            show = result.frame

            cv2.putText(show, f"Label: {cfg.label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(show, f"Saved clips: {saved_total + len(buffer_records)}/{cfg.clips}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            now = cv2.getTickCount() / cv2.getTickFrequency()

            if recording:
                feats = normalize_features(result.features)
                if feats.shape != (FEATURE_SIZE,):
                    feats = np.zeros((FEATURE_SIZE,), dtype=np.float32)
                clip_feats.append(feats.astype(np.float32))
                if result.hand_count > 0:
                    visible_frames += 1
                cv2.putText(
                    show,
                    f"REC {len(clip_feats)}/{cfg.seq_len}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                if len(clip_feats) >= cfg.seq_len:
                    if visible_frames >= cfg.min_visible_frames:
                        clip = np.stack(clip_feats, axis=0).astype(np.float32)
                        buffer_records.append((clip, cfg.label))
                        print(f"Captured clip {saved_total + len(buffer_records)}/{cfg.clips}")
                        if len(buffer_records) >= max(1, cfg.flush_every):
                            saved_now = append_sequence_records(buffer_records, cfg.out_npz, seq_len=cfg.seq_len)
                            saved_total += saved_now
                            buffer_records = []
                            print(f"Flushed clips to NPZ. Total saved: {saved_total}")
                    else:
                        print("Clip discarded (too few visible-hand frames).")
                    recording = False
                    clip_feats = []
                    visible_frames = 0
                    if auto_mode:
                        next_auto_start = now + max(0.2, cfg.clip_gap_sec)
            else:
                if auto_mode:
                    if next_auto_start <= 0.0:
                        next_auto_start = now + max(0.2, cfg.clip_gap_sec)
                    left = max(0.0, next_auto_start - now)
                    cv2.putText(
                        show,
                        f"AUTO mode: next clip in {left:.1f}s",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )
                    if (saved_total + len(buffer_records)) < cfg.clips and now >= next_auto_start:
                        recording = True
                        clip_feats = []
                        visible_frames = 0
                        next_auto_start = 0.0
                else:
                    cv2.putText(show, "Press C to record clip", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window_name, show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and not recording:
                recording = True
                clip_feats = []
                visible_frames = 0
            if key == ord("a"):
                auto_mode = not auto_mode
                next_auto_start = 0.0

            if (saved_total + len(buffer_records)) >= cfg.clips:
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    if buffer_records:
        saved_total += append_sequence_records(buffer_records, cfg.out_npz, seq_len=cfg.seq_len)
    print(f"Saved {saved_total} sequence clips to {cfg.out_npz}")
    return saved_total
