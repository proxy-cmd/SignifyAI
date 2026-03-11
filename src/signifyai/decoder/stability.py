from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import time

from .rules_intents import IntentHit


@dataclass
class StabilityConfig:
    window: int = 7
    min_confidence: float = 0.60
    hold_sec: float = 0.12


class StabilityFilter:
    def __init__(self, cfg: StabilityConfig) -> None:
        self.cfg = cfg
        self.labels: deque[str] = deque(maxlen=cfg.window)
        self.scores: deque[float] = deque(maxlen=cfg.window)
        self.pending = "unknown"
        self.pending_since = time.time()
        self.accepted = "unknown"

    def update(self, hit: IntentHit | None) -> tuple[str, float, str]:
        if hit is None:
            self.labels.append("silence")
            self.scores.append(0.0)
        else:
            lbl = hit.intent_id if hit.confidence >= self.cfg.min_confidence else "unknown"
            self.labels.append(lbl)
            self.scores.append(hit.confidence)

        voted = Counter(self.labels).most_common(1)[0][0] if self.labels else "unknown"
        now = time.time()
        if voted != self.pending:
            self.pending = voted
            self.pending_since = now

        if (now - self.pending_since) >= self.cfg.hold_sec:
            self.accepted = self.pending

        avg_score = float(sum(self.scores) / len(self.scores)) if self.scores else 0.0

        state = "stable" if self.accepted == voted else "pending"
        return self.accepted, avg_score, state

    def reset(self) -> None:
        self.labels.clear()
        self.scores.clear()
        self.pending = "unknown"
        self.accepted = "unknown"
        self.pending_since = time.time()
