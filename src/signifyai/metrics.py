from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics
import time


@dataclass
class RollingStageMetrics:
    maxlen: int = 240
    stage_samples: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=240)))
    e2e_samples: deque[float] = field(default_factory=lambda: deque(maxlen=240))

    def add_stage(self, stage: str, ms: float) -> None:
        self.stage_samples[stage].append(float(ms))

    def add_e2e(self, ms: float) -> None:
        self.e2e_samples.append(float(ms))

    def snapshot(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for stage, vals in self.stage_samples.items():
            if vals:
                out[f"{stage}_median_ms"] = float(statistics.median(vals))
        if self.e2e_samples:
            out["e2e_median_ms"] = float(statistics.median(self.e2e_samples))
        return out


class StageTimer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
