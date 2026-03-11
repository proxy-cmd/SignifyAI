from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import uuid
from typing import Any

import numpy as np

from ..contracts import LandmarkFrame, flatten_landmark_frame


@dataclass
class RecordingConfig:
    root: Path = Path("data/landmarks")
    min_brightness: float = 35.0
    min_blur: float = 45.0
    min_hand_area: float = 0.009


class RecordingModule:
    """Stores landmark-first session clips with simple quality gating."""

    def __init__(self, cfg: RecordingConfig) -> None:
        self.cfg = cfg
        self.cfg.root.mkdir(parents=True, exist_ok=True)
        self.active_session: dict[str, Any] | None = None

    def start_session(self, intent_id: str, signer_id: str = "anonymous", consent_raw_video: bool = False) -> dict[str, Any]:
        sid = f"sess_{uuid.uuid4().hex[:12]}"
        sdir = self.cfg.root / "raw" / sid
        sdir.mkdir(parents=True, exist_ok=True)

        self.active_session = {
            "session_id": sid,
            "intent_id": intent_id,
            "signer_id": signer_id,
            "consent_raw_video": bool(consent_raw_video),
            "clips": 0,
            "created_at": int(time.time() * 1000),
            "session_dir": str(sdir),
        }
        self._write_session_state()
        return dict(self.active_session)

    def append_frames(self, frames: list[LandmarkFrame]) -> dict[str, Any]:
        session = self._require_session()
        if not frames:
            return {"accepted": False, "reason": "empty"}

        quality = self.quality_report(frames)
        if not self._passes_quality_gate(quality):
            return {"accepted": False, "reason": "quality_gate", "quality": quality}

        clip_id = f"clip_{session['clips'] + 1:04d}"
        sdir = Path(session["session_dir"])

        seq = np.stack([flatten_landmark_frame(frame) for frame in frames], axis=0).astype(np.float32)
        ts = np.asarray([frame.timestamp_ms for frame in frames], dtype=np.int64)
        npz_path = sdir / f"{clip_id}.npz"
        np.savez_compressed(npz_path, sequence=seq, timestamps=ts)

        record = {
            "session_id": session["session_id"],
            "clip_id": clip_id,
            "intent_id": session["intent_id"],
            "signer_id": session["signer_id"],
            "consent_raw_video": session["consent_raw_video"],
            "npz_path": str(npz_path),
            "frames": int(seq.shape[0]),
            "quality": quality,
        }
        with (sdir / "clips.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        session["clips"] += 1
        self._write_session_state()
        return {"accepted": True, "clip_id": clip_id, "quality": quality}

    def finalize_clip(self) -> dict[str, Any]:
        session = self._require_session()
        payload = dict(session)
        self.active_session = None
        return payload

    @staticmethod
    def quality_report(frames: list[LandmarkFrame]) -> dict[str, float]:
        brightness_vals = [float(frame.quality.get("brightness", 0.0)) for frame in frames]
        blur_vals = [float(frame.quality.get("blur", 0.0)) for frame in frames]
        hand_area_vals = [float(frame.quality.get("hand_area", 0.0)) for frame in frames]
        return {
            "brightness_avg": float(np.mean(brightness_vals)) if brightness_vals else 0.0,
            "blur_avg": float(np.mean(blur_vals)) if blur_vals else 0.0,
            "hand_area_max": float(np.max(hand_area_vals)) if hand_area_vals else 0.0,
        }

    def _passes_quality_gate(self, quality: dict[str, float]) -> bool:
        return (
            quality["brightness_avg"] >= self.cfg.min_brightness
            and quality["blur_avg"] >= self.cfg.min_blur
            and quality["hand_area_max"] >= self.cfg.min_hand_area
        )

    def _require_session(self) -> dict[str, Any]:
        if self.active_session is None:
            raise RuntimeError("No active session. Call start_session first.")
        return self.active_session

    def _write_session_state(self) -> None:
        session = self._require_session()
        session_dir = Path(session["session_dir"])
        (session_dir / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
