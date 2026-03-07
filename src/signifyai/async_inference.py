from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Generic, Optional, TypeVar


InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


@dataclass
class WorkerStats:
    submitted: int = 0
    processed: int = 0
    dropped: int = 0
    errors: int = 0


class LatestFrameWorker(Generic[InputT, ResultT]):
    """
    Keeps only the latest input item and processes it on a background thread.
    Older queued inputs are dropped automatically.
    """

    def __init__(self, process_fn: Callable[[InputT], ResultT]) -> None:
        self._process_fn = process_fn
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_input: Optional[InputT] = None
        self._latest_result: Optional[ResultT] = None
        self._latest_result_id = 0
        self._last_error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, name="latest-frame-worker", daemon=True)
        self.stats = WorkerStats()

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def submit(self, item: InputT) -> None:
        with self._lock:
            if self._latest_input is not None:
                self.stats.dropped += 1
            self._latest_input = item
            self.stats.submitted += 1

    def poll_latest_result(self) -> Optional[ResultT]:
        with self._lock:
            return self._latest_result

    def poll_latest_result_with_id(self) -> tuple[Optional[ResultT], int]:
        with self._lock:
            return self._latest_result, self._latest_result_id

    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def close(self, timeout_sec: float = 1.2) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_sec)

    def _run(self) -> None:
        while not self._stop.is_set():
            current: Optional[InputT] = None
            with self._lock:
                if self._latest_input is not None:
                    current = self._latest_input
                    self._latest_input = None
            if current is None:
                time.sleep(0.001)
                continue
            try:
                result = self._process_fn(current)
                with self._lock:
                    self._latest_result = result
                    self._latest_result_id += 1
                self.stats.processed += 1
            except Exception as ex:
                with self._lock:
                    self._last_error = str(ex)
                self.stats.errors += 1
