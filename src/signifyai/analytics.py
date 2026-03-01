from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .safe_logging import csv_safe_text


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
        label=csv_safe_text(label),
        confidence=float(confidence),
        hand_count=int(hand_count),
    )
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "label", "confidence", "hand_count"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(event.__dict__)
