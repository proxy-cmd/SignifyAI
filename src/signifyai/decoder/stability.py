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
        self.conf: deque[float] = deque(maxlen=cfg.window)
        self.pending_label = "unknown"
        self.pending_since = time.time()
        self.accepted_label = "unknown"

    def update(self, hit: IntentHit | None) -> tuple[str, float, str]:
        if hit is None:
            self.labels.append("silence")
            self.conf.append(0.0)
        else:
            lbl = hit.intent_id if hit.confidence >= self.cfg.min_confidence else "unknown"
            self.labels.append(lbl)
            self.conf.append(hit.confidence)

        voted = Counter(self.labels).most_common(1)[0][0] if self.labels else "unknown"
        now = time.time()
        if voted != self.pending_label:
            self.pending_label = voted
            self.pending_since = now

        if (now - self.pending_since) >= self.cfg.hold_sec:
            self.accepted_label = self.pending_label

        avg_conf = 0.0
        if self.conf:
            avg_conf = float(sum(self.conf) / len(self.conf))

        state = "stable" if self.accepted_label == voted else "pending"
        return self.accepted_label, avg_conf, state

    def reset(self) -> None:
        self.labels.clear()
        self.conf.clear()
        self.pending_label = "unknown"
        self.accepted_label = "unknown"
        self.pending_since = time.time()
