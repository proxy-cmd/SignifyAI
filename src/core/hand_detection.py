from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


class CamCfg:
    def __init__(self, idx=0, w=960, h=540, fps=30):
        self.idx = idx
        self.w = w
        self.h = h
        self.fps = fps


class CamStream:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.idx, cv2.CAP_DSHOW)
        self.apply_cfg()
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def apply_cfg(self):
        # keep webcam fast and low-lag
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.h)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    def close(self):
        if self.cap.isOpened():
            self.cap.release()


class FrameData:
    def __init__(self, ts_ms, hand_count, left, right, quality):
        self.ts_ms = ts_ms
        self.hand_count = hand_count
        self.left = left
        self.right = right
        self.quality = quality


class HandCfg:
    def __init__(self, model_path=Path("data/models/hand_landmarker.task"), max_hands=2, min_det=0.65, min_track=0.6, scale=0.65):
        self.model_path = model_path
        self.max_hands = max_hands
        self.min_det = min_det
        self.min_track = min_track
        self.scale = scale


def _ensure_model(path):
    # download model only once if missing
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    import urllib.request

    urllib.request.urlretrieve(url, str(path))
    return path


def _to_arr(points, max_n=21):
    # convert mediapipe points -> numpy array
    if points is None:
        return None
    out = []
    count = 0
    for p in points:
        if count >= max_n:
            break
        out.append([p.x, p.y, p.z])
        count += 1
    return np.asarray(out, dtype=np.float32)


def _quality(frame, left, right):
    # basic quality signals used in recording gate
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    area = 0.0
    for hand in (left, right):
        if hand is None:
            continue
        xs = hand[:, 0]
        ys = hand[:, 1]
        area = max(area, float((xs.max() - xs.min()) * (ys.max() - ys.min())))
    return {"brightness": bright, "blur": blur, "hand_area": area}


class HandDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        path = _ensure_model(cfg.model_path)
        opt = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_det,
            min_tracking_confidence=cfg.min_track,
        )
        self.model = vision.HandLandmarker.create_from_options(opt)

    def close(self):
        self.model.close()

    def process(self, frame_bgr):
        # run detection on smaller frame for speed, but quality uses original frame
        run_frame = frame_bgr
        if self.cfg.scale < 0.999:
            run_frame = cv2.resize(frame_bgr, None, fx=self.cfg.scale, fy=self.cfg.scale, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(run_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.model.detect(mp_img)

        left = _to_arr(res.hand_landmarks[0]) if len(res.hand_landmarks) > 0 else None
        right = _to_arr(res.hand_landmarks[1]) if len(res.hand_landmarks) > 1 else None
        count = int(left is not None) + int(right is not None)
        q = _quality(frame_bgr, left, right)
        return FrameData(int(time.time() * 1000), count, left, right, q)


def draw_hands(frame, data):
    # draw simple hand skeleton
    links = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    def draw(hand, color):
        if hand is None:
            return
        h, w = frame.shape[:2]
        pts = []
        for p in hand:
            x = int(float(p[0]) * w)
            y = int(float(p[1]) * h)
            pts.append((x, y))
            cv2.circle(frame, (x, y), 3, color, -1)
        for a, b in links:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], color, 1)

    draw(data.left, (0, 255, 255))
    draw(data.right, (255, 200, 0))
