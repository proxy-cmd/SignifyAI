from dataclasses import dataclass, field
from pathlib import Path
import time
import urllib.request
from typing import List, Tuple

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
import mediapipe as mp


@dataclass
class EyeCfg:
    model_path: Path = Path("data/models/face_landmarker.task")
    min_det: float = 0.5
    min_track: float = 0.5


def _ensure_model(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, str(path))
    return path


@dataclass
class EyeState:
    ts_ms: int
    face_found: bool
    left_ear: float
    right_ear: float
    gaze_x: float
    gaze_y: float = 0.5
    keypoints: List[Tuple[float, float]] = field(default_factory=list)


class EyeDetector:
    def __init__(self, cfg: EyeCfg):
        self.cfg = cfg
        model_path = _ensure_model(Path(cfg.model_path))
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=float(cfg.min_det),
            min_tracking_confidence=float(cfg.min_track),
        )
        self.mesh = vision.FaceLandmarker.create_from_options(opts)
        self.last_state = None
        self.last_ok_ts = 0
        self.hold_ms = 220

    def close(self):
        self.mesh.close()

    @staticmethod
    def _dist(a, b):
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _to_xy(points, idx):
        p = points[idx]
        return np.asarray([float(p.x), float(p.y)], dtype=np.float32)

    @staticmethod
    def _eye_mid(state):
        if state is None or len(state.keypoints) < 8:
            return np.asarray([0.5, 0.5], dtype=np.float32)
        xs = [float(p[0]) for p in state.keypoints[:8]]
        ys = [float(p[1]) for p in state.keypoints[:8]]
        return np.asarray([sum(xs) / 8.0, sum(ys) / 8.0], dtype=np.float32)

    @staticmethod
    def _ok_geom(left_ear, right_ear, l_h, r_h, l_outer, l_inner, r_outer, r_inner):
        if l_h < 1e-4 or r_h < 1e-4:
            return False
        ear = (float(left_ear) + float(right_ear)) * 0.5
        if ear < 0.05 or ear > 0.75:
            return False
        ly = (float(l_outer[1]) + float(l_inner[1])) * 0.5
        ry = (float(r_outer[1]) + float(r_inner[1])) * 0.5
        if abs(ly - ry) > 0.14:
            return False
        return True

    def _use_last(self, now_ms):
        if self.last_state is None:
            return None
        if (int(now_ms) - int(self.last_ok_ts)) > int(self.hold_ms):
            return None
        s = self.last_state
        return EyeState(int(now_ms), True, float(s.left_ear), float(s.right_ear), float(s.gaze_x), float(s.gaze_y), list(s.keypoints))

    def process(self, frame_bgr):
        now_ms = int(time.time() * 1000)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        out = self.mesh.detect(mp_img)
        if not out.face_landmarks:
            hold = self._use_last(now_ms)
            if hold is not None:
                return hold
            return EyeState(now_ms, False, 0.0, 0.0, 0.5, 0.5)

        lm = out.face_landmarks[0]
        has_iris = len(lm) > 473

        # Eye corners/lids and iris centers (requires refine_landmarks=True)
        l_outer = self._to_xy(lm, 33)
        l_inner = self._to_xy(lm, 133)
        l_top = self._to_xy(lm, 159)
        l_bottom = self._to_xy(lm, 145)

        r_outer = self._to_xy(lm, 263)
        r_inner = self._to_xy(lm, 362)
        r_top = self._to_xy(lm, 386)
        r_bottom = self._to_xy(lm, 374)

        l_h = max(self._dist(l_outer, l_inner), 1e-6)
        r_h = max(self._dist(r_outer, r_inner), 1e-6)
        l_v = self._dist(l_top, l_bottom)
        r_v = self._dist(r_top, r_bottom)

        left_ear = l_v / l_h
        right_ear = r_v / r_h
        if not self._ok_geom(left_ear, right_ear, l_h, r_h, l_outer, l_inner, r_outer, r_inner):
            hold = self._use_last(now_ms)
            if hold is not None:
                return hold
            return EyeState(now_ms, False, 0.0, 0.0, 0.5, 0.5)

        l_min = min(float(l_outer[0]), float(l_inner[0]))
        l_max = max(float(l_outer[0]), float(l_inner[0]))
        r_min = min(float(r_outer[0]), float(r_inner[0]))
        r_max = max(float(r_outer[0]), float(r_inner[0]))

        if has_iris:
            l_iris = self._to_xy(lm, 468)
            r_iris = self._to_xy(lm, 473)
            l_ratio = (float(l_iris[0]) - l_min) / max(l_max - l_min, 1e-6)
            r_ratio = (float(r_iris[0]) - r_min) / max(r_max - r_min, 1e-6)
            gaze_x = float(np.clip((l_ratio + r_ratio) * 0.5, 0.0, 1.0))

            l_y_ratio = (float(l_iris[1]) - float(l_top[1])) / max(float(l_bottom[1]) - float(l_top[1]), 1e-6)
            r_y_ratio = (float(r_iris[1]) - float(r_top[1])) / max(float(r_bottom[1]) - float(r_top[1]), 1e-6)
            gaze_y = float(np.clip((l_y_ratio + r_y_ratio) * 0.5, 0.0, 1.0))
        else:
            gaze_x = 0.5
            gaze_y = 0.5

        keypoints = [
            (float(l_outer[0]), float(l_outer[1])),
            (float(l_inner[0]), float(l_inner[1])),
            (float(l_top[0]), float(l_top[1])),
            (float(l_bottom[0]), float(l_bottom[1])),
            (float(r_outer[0]), float(r_outer[1])),
            (float(r_inner[0]), float(r_inner[1])),
            (float(r_top[0]), float(r_top[1])),
            (float(r_bottom[0]), float(r_bottom[1])),
        ]
        if has_iris:
            keypoints.append((float(l_iris[0]), float(l_iris[1])))
            keypoints.append((float(r_iris[0]), float(r_iris[1])))

        cur = EyeState(now_ms, True, float(left_ear), float(right_ear), float(gaze_x), float(gaze_y), keypoints)
        if self.last_state is not None:
            prev_mid = self._eye_mid(self.last_state)
            cur_mid = self._eye_mid(cur)
            jump = float(np.linalg.norm(cur_mid - prev_mid))
            if jump > 0.26 and (now_ms - int(self.last_ok_ts)) < 160:
                hold = self._use_last(now_ms)
                if hold is not None:
                    return hold
        self.last_state = cur
        self.last_ok_ts = now_ms
        return cur


def draw_eye_debug(frame, state: EyeState, show_landmarks: bool = True):
    if state is None:
        return
    if not state.face_found:
        cv2.putText(frame, "Eye: no face", (18, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 120, 255), 2)
        return

    if show_landmarks:
        h, w = frame.shape[:2]
        if len(state.keypoints) >= 8:
            pts = [(int(p[0] * w), int(p[1] * h)) for p in state.keypoints[:8]]
            # Keep eye debug clean and simple: center lines + tiny points.
            l_outer, l_inner, l_top, l_bottom = pts[0], pts[1], pts[2], pts[3]
            r_outer, r_inner, r_top, r_bottom = pts[4], pts[5], pts[6], pts[7]
            guide_color = (0, 220, 255)
            cv2.line(frame, l_outer, l_inner, guide_color, 1)
            cv2.line(frame, l_top, l_bottom, guide_color, 1)
            cv2.line(frame, r_outer, r_inner, guide_color, 1)
            cv2.line(frame, r_top, r_bottom, guide_color, 1)
            for p in pts:
                cv2.circle(frame, p, 1, (255, 255, 255), -1)

        # iris centers: tiny green points
        for p in state.keypoints[8:]:
            x = int(p[0] * w)
            y = int(p[1] * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    ear = (float(state.left_ear) + float(state.right_ear)) * 0.5
    text = f"Eye EAR {ear:.3f} | GazeX {state.gaze_x:.2f} | GazeY {state.gaze_y:.2f}"
    cv2.putText(frame, text, (18, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 180), 2)
