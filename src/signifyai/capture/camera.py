from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 960
    height: int = 540
    fps: int = 30


def apply_camera_settings(cap: cv2.VideoCapture, cfg: CameraConfig) -> None:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_fn(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class CameraStream:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.index, cv2.CAP_DSHOW)
        apply_camera_settings(self.cap, cfg)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
