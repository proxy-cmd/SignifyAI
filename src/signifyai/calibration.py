from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .config import DEFAULT_CALIBRATION_PROFILE_PATH
from .hand_tracking import HandTracker, check_camera, open_camera, warmup_camera


@dataclass
class CalibrationConfig:
    camera_index: int = 0
    width: int = 960
    height: int = 720
    seconds: float = 20.0
    out_json: Path = DEFAULT_CALIBRATION_PROFILE_PATH
    model_complexity: int = 0
    inference_scale: float = 0.75


def _max_hand_area(raw_hands: list[np.ndarray]) -> float:
    if not raw_hands:
        return 0.0
    areas = []
    for hand in raw_hands:
        xs = hand[:, 0]
        ys = hand[:, 1]
        areas.append(float((xs.max() - xs.min()) * (ys.max() - ys.min())))
    return float(max(areas)) if areas else 0.0


def recommend_runtime_settings(
    *,
    avg_fps: float,
    p20_brightness: float,
    p20_blur: float,
    p15_hand_area: float,
) -> dict[str, float | int]:
    if avg_fps < 14.0:
        infer_interval = 3
        infer_scale = 0.58
        smooth = 5
        landmark_smoothing = 0.72
    elif avg_fps < 20.0:
        infer_interval = 2
        infer_scale = 0.66
        smooth = 6
        landmark_smoothing = 0.76
    elif avg_fps < 28.0:
        infer_interval = 1
        infer_scale = 0.72
        smooth = 7
        landmark_smoothing = 0.80
    else:
        infer_interval = 1
        infer_scale = 0.80
        smooth = 8
        landmark_smoothing = 0.84

    min_brightness = float(max(30.0, min(95.0, p20_brightness * 0.90)))
    min_blur_var = float(max(35.0, min(220.0, p20_blur * 0.82)))
    min_hand_area = float(max(0.008, min(0.050, p15_hand_area * 0.85)))

    threshold = 0.60
    if p20_blur < 70:
        threshold = 0.58
    if p20_blur < 45 or p20_brightness < 45:
        threshold = 0.56

    return {
        "infer_interval": int(infer_interval),
        "infer_scale": float(infer_scale),
        "smooth": int(smooth),
        "landmark_smoothing": float(landmark_smoothing),
        "threshold": float(threshold),
        "min_brightness": min_brightness,
        "min_blur_var": min_blur_var,
        "min_hand_area": min_hand_area,
        "target_fps": float(max(14.0, min(35.0, avg_fps * 0.85))),
    }


def run_calibration(cfg: CalibrationConfig) -> Path:
    cap = open_camera(index=cfg.camera_index, width=cfg.width, height=cfg.height, fps=60)
    err = check_camera(cap)
    if err:
        raise RuntimeError(err)
    warmup_camera(cap)
    tracker = HandTracker(
        max_num_hands=2,
        model_complexity=cfg.model_complexity,
        inference_scale=cfg.inference_scale,
        landmark_smoothing=0.75,
    )

    window = "SignifyAI Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, cfg.width, cfg.height)

    fps_samples: list[float] = []
    brightness_samples: list[float] = []
    blur_samples: list[float] = []
    hand_area_samples: list[float] = []
    started = time.time()
    prev = started
    aborted = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            dt = max(1e-6, now - prev)
            prev = now
            fps_samples.append(1.0 / dt)

            frame = cv2.flip(frame, 1)
            detection = tracker.process(frame, draw=True)
            out = detection.frame

            brightness = float(out.mean())
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness_samples.append(brightness)
            blur_samples.append(blur_var)
            if detection.hand_count > 0:
                hand_area_samples.append(_max_hand_area(detection.raw_hands))

            elapsed = now - started
            remain = max(0.0, float(cfg.seconds) - elapsed)
            cv2.putText(out, "Calibration: show your common signs naturally", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(out, f"Time left: {remain:0.1f}s", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            cv2.putText(out, f"Hands: {detection.hand_count}", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
            cv2.putText(out, "Press q or Esc to cancel", (20, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            cv2.imshow(window, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                aborted = True
                break
            if remain <= 0.0:
                break

        if aborted:
            raise RuntimeError("Calibration cancelled by user.")

        if not fps_samples:
            raise RuntimeError("Calibration failed: no camera frames captured.")
        if not hand_area_samples:
            raise RuntimeError("Calibration failed: no hand detected. Keep your hand in frame and retry.")

        avg_fps = float(np.mean(fps_samples))
        p20_brightness = float(np.percentile(np.asarray(brightness_samples, dtype=np.float32), 20))
        p20_blur = float(np.percentile(np.asarray(blur_samples, dtype=np.float32), 20))
        p15_hand_area = float(np.percentile(np.asarray(hand_area_samples, dtype=np.float32), 15))
        recommended = recommend_runtime_settings(
            avg_fps=avg_fps,
            p20_brightness=p20_brightness,
            p20_blur=p20_blur,
            p15_hand_area=p15_hand_area,
        )

        payload = {
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "camera": {"index": cfg.camera_index, "width": cfg.width, "height": cfg.height},
            "metrics": {
                "avg_fps": avg_fps,
                "p20_brightness": p20_brightness,
                "p20_blur_var": p20_blur,
                "p15_hand_area": p15_hand_area,
                "samples": {
                    "fps": len(fps_samples),
                    "brightness": len(brightness_samples),
                    "blur": len(blur_samples),
                    "hand_area": len(hand_area_samples),
                },
            },
            "recommended": recommended,
        }

        cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return cfg.out_json
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

