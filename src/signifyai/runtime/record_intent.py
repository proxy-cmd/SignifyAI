from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2

from ..capture.camera import CameraConfig, CameraStream
from ..data.recording import RecordingConfig as RecStoreCfg
from ..data.recording import RecordingModule as RecStore
from ..perception.landmarks import MultiModalPerceptor, PerceptionConfig


@dataclass
class RecordConfig:
    intent_id: str
    clips: int = 8
    clip_seconds: float = 1.2
    camera_index: int = 0
    width: int = 960
    height: int = 540
    fps: int = 30
    signer_id: str = "anonymous"
    consent_raw_video: bool = False


class IntentRecorder:
    def __init__(self, cfg: RecordConfig) -> None:
        self.cfg = cfg
        self.cam = CameraStream(CameraConfig(index=cfg.camera_index, width=cfg.width, height=cfg.height, fps=cfg.fps))
        self.perceptor = MultiModalPerceptor(PerceptionConfig(inference_scale=0.65))
        self.rec = RecStore(RecStoreCfg(root=Path("data/landmarks")))

    def close(self) -> None:
        self.perceptor.close()
        self.cam.close()

    def run(self) -> dict:
        _ = self.rec.start_session(
            intent_id=self.cfg.intent_id,
            signer_id=self.cfg.signer_id,
            consent_raw_video=self.cfg.consent_raw_video,
        )
        window = "SignifyAI Record"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, self.cfg.width, self.cfg.height)

        clips_saved = 0
        try:
            while clips_saved < self.cfg.clips:
                ok, frame = self.cam.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 88), (0, 0, 0), -1)
                cv2.putText(frame, f"Intent: {self.cfg.intent_id}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
                cv2.putText(frame, f"Clips: {clips_saved}/{self.cfg.clips} | SPACE capture | q quit", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (210, 210, 210), 2)
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == 32:
                    start = time.time()
                    frames = []
                    while (time.time() - start) < self.cfg.clip_seconds:
                        ok2, frame2 = self.cam.read()
                        if not ok2:
                            break
                        frame2 = cv2.flip(frame2, 1)
                        lm = self.perceptor.process(frame2)
                        frames.append(lm)
                        cv2.putText(frame2, "CAPTURING...", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        cv2.imshow(window, frame2)
                        cv2.waitKey(1)
                    result = self.rec.append_frames(frames)
                    if result.get("accepted"):
                        clips_saved += 1
                        print(f"Saved clip {clips_saved}/{self.cfg.clips}: {result.get('clip_id')}")
                    else:
                        print(f"Rejected clip: {result}")

                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()
            self.close()

        final = self.rec.finalize_clip()
        final["clips_saved"] = clips_saved
        return final
