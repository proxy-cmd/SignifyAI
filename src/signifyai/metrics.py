from __future__ import annotations

from collections import deque
import statistics
import time


class RollingStageMetrics:
    def __init__(self, size: int = 240) -> None:
        self.size = int(size)
        self.stage_samples: dict[str, deque[float]] = {}
        self.e2e_samples: deque[float] = deque(maxlen=self.size)

    def add_stage(self, stage: str, ms: float) -> None:
        if stage not in self.stage_samples:
            self.stage_samples[stage] = deque(maxlen=self.size)
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
    """Small helper for measuring stage durations."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
