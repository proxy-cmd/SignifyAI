from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class SessionEvent:
    timestamp: str
    label: str
    confidence: float
    hand_count: int


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_event(log_path: Path, label: str, confidence: float, hand_count: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = SessionEvent(
        timestamp=_now_iso(),
        label=label,
        confidence=float(confidence),
        hand_count=int(hand_count),
    )
    row = pd.DataFrame([event.__dict__])

    if log_path.exists():
        prev = pd.read_csv(log_path)
        row = pd.concat([prev, row], ignore_index=True)

    row.to_csv(log_path, index=False)
