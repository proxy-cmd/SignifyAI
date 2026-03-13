from pathlib import Path
import json
import time
import uuid

import cv2
import numpy as np

from core.hand_detection import CamCfg, CamStream, HandCfg, HandDetector

HAND_VEC = 21 * 3


class RecCfg:
    def __init__(
        self,
        intent,
        clips=8,
        clip_sec=1.2,
        cam_idx=0,
        w=960,
        h=540,
        fps=30,
        signer="anonymous",
        consent_raw=False,
        root=Path("data/landmarks"),
        min_brightness=35.0,
        min_blur=45.0,
        min_hand_area=0.009,
    ):
        self.intent = intent
        self.clips = clips
        self.clip_sec = clip_sec
        self.cam_idx = cam_idx
        self.w = w
        self.h = h
        self.fps = fps
        self.signer = signer
        self.consent_raw = consent_raw
        self.root = root
        self.min_brightness = min_brightness
        self.min_blur = min_blur
        self.min_hand_area = min_hand_area


def _pad(arr, size):
    if arr is None:
        return np.zeros((size,), dtype=np.float32)
    flat = arr.astype(np.float32).reshape(-1)
    if flat.size >= size:
        return flat[:size]
    out = np.zeros((size,), dtype=np.float32)
    out[: flat.size] = flat
    return out


def frame_to_vec(data):
    left = _pad(data.left, HAND_VEC)
    right = _pad(data.right, HAND_VEC)
    q = np.asarray(
        [
            float(data.quality.get("brightness", 0.0)),
            float(data.quality.get("blur", 0.0)),
            float(data.quality.get("hand_area", 0.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([left, right, q], axis=0)


def quality_report(frames):
    bright = [float(f.quality.get("brightness", 0.0)) for f in frames]
    blur = [float(f.quality.get("blur", 0.0)) for f in frames]
    area = [float(f.quality.get("hand_area", 0.0)) for f in frames]
    return {
        "brightness_avg": float(np.mean(bright)) if bright else 0.0,
        "blur_avg": float(np.mean(blur)) if blur else 0.0,
        "hand_area_max": float(np.max(area)) if area else 0.0,
    }


class RecSession:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cam = CamStream(CamCfg(idx=cfg.cam_idx, w=cfg.w, h=cfg.h, fps=cfg.fps))
        self.det = HandDetector(HandCfg(scale=0.65))
        self.sid = f"sess_{uuid.uuid4().hex[:12]}"
        self.sdir = cfg.root / "raw" / self.sid
        self.sdir.mkdir(parents=True, exist_ok=True)
        self.saved = 0
        self.state = {
            "session_id": self.sid,
            "intent_id": cfg.intent,
            "signer_id": cfg.signer,
            "consent_raw_video": bool(cfg.consent_raw),
            "clips": 0,
            "created_at": int(time.time() * 1000),
            "session_dir": str(self.sdir),
        }
        self._save_state()

    def _save_state(self):
        (self.sdir / "session.json").write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _pass_quality(self, q):
        # simple quality gate to reject bad clips
        return (
            q["brightness_avg"] >= self.cfg.min_brightness
            and q["blur_avg"] >= self.cfg.min_blur
            and q["hand_area_max"] >= self.cfg.min_hand_area
        )

    def _save_clip(self, frames):
        if not frames:
            return {"accepted": False, "reason": "empty"}

        q = quality_report(frames)
        if not self._pass_quality(q):
            return {"accepted": False, "reason": "quality_gate", "quality": q}

        clip_id = f"clip_{self.state['clips'] + 1:04d}"
        seq = np.stack([frame_to_vec(f) for f in frames], axis=0).astype(np.float32)
        ts = np.asarray([f.ts_ms for f in frames], dtype=np.int64)
        npz = self.sdir / f"{clip_id}.npz"
        np.savez_compressed(npz, sequence=seq, timestamps=ts)

        row = {
            "session_id": self.sid,
            "clip_id": clip_id,
            "intent_id": self.cfg.intent,
            "signer_id": self.cfg.signer,
            "consent_raw_video": bool(self.cfg.consent_raw),
            "npz_path": str(npz),
            "frames": int(seq.shape[0]),
            "quality": q,
        }
        with (self.sdir / "clips.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        self.state["clips"] += 1
        self.saved += 1
        self._save_state()
        return {"accepted": True, "clip_id": clip_id, "quality": q}

    def close(self):
        self.det.close()
        self.cam.close()

    def run(self):
        # open camera window and capture N clips on SPACE key
        win = "SignifyAI Record"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.cfg.w, self.cfg.h)
        try:
            while self.saved < self.cfg.clips:
                ok, frame = self.cam.read()
                if not ok or frame is None:
                    break
                frame = cv2.flip(frame, 1)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 88), (0, 0, 0), -1)
                cv2.putText(frame, f"Intent: {self.cfg.intent}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
                cv2.putText(
                    frame,
                    f"Clips: {self.saved}/{self.cfg.clips} | SPACE capture | q quit",
                    (18, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (210, 210, 210),
                    2,
                )
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == 32:
                    start = time.time()
                    clip_frames = []
                    while (time.time() - start) < self.cfg.clip_sec:
                        ok2, frame2 = self.cam.read()
                        if not ok2 or frame2 is None:
                            break
                        frame2 = cv2.flip(frame2, 1)
                        clip_frames.append(self.det.process(frame2))
                        cv2.putText(frame2, "CAPTURING...", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        cv2.imshow(win, frame2)
                        cv2.waitKey(1)
                    res = self._save_clip(clip_frames)
                    if res.get("accepted"):
                        print(f"Saved clip {self.saved}/{self.cfg.clips}: {res.get('clip_id')}")
                    else:
                        print(f"Rejected clip: {res}")
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()
            self.close()

        out = dict(self.state)
        out["clips_saved"] = self.saved
        return out
