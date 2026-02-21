from __future__ import annotations

import time
from dataclasses import dataclass

from .hand_tracking import HandTracker, check_camera, open_camera, warmup_camera


@dataclass
class BenchmarkResult:
    raw_fps: float
    tracker_fps: float
    frames: int
    seconds: float


def run_benchmark(camera_index: int = 0, width: int = 960, height: int = 720, seconds: float = 6.0) -> BenchmarkResult:
    cap = open_camera(index=camera_index, width=width, height=height)
    err = check_camera(cap)
    if err:
        raise RuntimeError(err)
    warmup_camera(cap)

    tracker = HandTracker(max_num_hands=2, inference_scale=0.75)

    # Raw camera FPS
    t0 = time.time()
    raw_frames = 0
    while (time.time() - t0) < seconds:
        ok, _ = cap.read()
        if not ok:
            break
        raw_frames += 1
    raw_elapsed = max(time.time() - t0, 1e-6)
    raw_fps = raw_frames / raw_elapsed

    # Tracker FPS
    t1 = time.time()
    tr_frames = 0
    while (time.time() - t1) < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        tracker.process(frame, draw=False)
        tr_frames += 1
    tr_elapsed = max(time.time() - t1, 1e-6)
    tracker_fps = tr_frames / tr_elapsed

    tracker.close()
    cap.release()
    return BenchmarkResult(raw_fps=raw_fps, tracker_fps=tracker_fps, frames=tr_frames, seconds=seconds)

