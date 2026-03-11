from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 960
    height: int = 540
    fps: int = 30


class CameraStream:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def read(self) -> tuple[bool, object]:
        return self.cap.read()

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
