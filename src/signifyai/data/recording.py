from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import uuid

import numpy as np

from ..contracts import LandmarkFrame, SequenceWindow, flatten_landmark_frame


@dataclass
class RecordingConfig:
    root: Path = Path("data/landmarks")
    min_brightness: float = 35.0
    min_blur: float = 45.0
    min_hand_area: float = 0.009


class RecordingModule:
    def __init__(self, cfg: RecordingConfig) -> None:
        self.cfg = cfg
        self.cfg.root.mkdir(parents=True, exist_ok=True)
        self.active_session: dict | None = None

    def start_session(self, intent_id: str, signer_id: str = "anonymous", consent_raw_video: bool = False) -> dict:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        session_dir = self.cfg.root / "raw" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self.active_session = {
            "session_id": session_id,
            "intent_id": intent_id,
            "signer_id": signer_id,
            "consent_raw_video": bool(consent_raw_video),
            "clips": 0,
            "created_at": int(time.time() * 1000),
            "session_dir": str(session_dir),
        }
        (session_dir / "session.json").write_text(json.dumps(self.active_session, indent=2), encoding="utf-8")
        return dict(self.active_session)

    def append_frames(self, frames: list[LandmarkFrame]) -> dict:
        if self.active_session is None:
            raise RuntimeError("No active session. Call start_session first.")
        if not frames:
            return {"accepted": False, "reason": "empty"}

        quality = self.quality_report(frames)
        accepted = (
            quality["brightness_avg"] >= self.cfg.min_brightness
            and quality["blur_avg"] >= self.cfg.min_blur
            and quality["hand_area_max"] >= self.cfg.min_hand_area
        )
        if not accepted:
            return {"accepted": False, "reason": "quality_gate", "quality": quality}

        session_dir = Path(self.active_session["session_dir"])
        clip_id = f"clip_{self.active_session['clips'] + 1:04d}"
        mat = np.stack([flatten_landmark_frame(f) for f in frames], axis=0).astype(np.float32)
        ts = np.asarray([f.timestamp_ms for f in frames], dtype=np.int64)
        out_npz = session_dir / f"{clip_id}.npz"
        np.savez_compressed(out_npz, sequence=mat, timestamps=ts)

        rec = {
            "session_id": self.active_session["session_id"],
            "clip_id": clip_id,
            "intent_id": self.active_session["intent_id"],
            "signer_id": self.active_session["signer_id"],
            "consent_raw_video": self.active_session["consent_raw_video"],
            "npz_path": str(out_npz),
            "frames": int(mat.shape[0]),
            "quality": quality,
        }
        with (session_dir / "clips.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        self.active_session["clips"] += 1
        (session_dir / "session.json").write_text(json.dumps(self.active_session, indent=2), encoding="utf-8")
        return {"accepted": True, "clip_id": clip_id, "quality": quality}

    def finalize_clip(self) -> dict:
        if self.active_session is None:
            raise RuntimeError("No active session")
        payload = dict(self.active_session)
        self.active_session = None
        return payload

    def quality_report(self, frames: list[LandmarkFrame]) -> dict[str, float]:
        brightness = [float(f.quality.get("brightness", 0.0)) for f in frames]
        blur = [float(f.quality.get("blur", 0.0)) for f in frames]
        hand_area = [float(f.quality.get("hand_area", 0.0)) for f in frames]
        return {
            "brightness_avg": float(np.mean(brightness)) if brightness else 0.0,
            "blur_avg": float(np.mean(blur)) if blur else 0.0,
            "hand_area_max": float(np.max(hand_area)) if hand_area else 0.0,
        }
