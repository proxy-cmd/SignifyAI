from __future__ import annotations

import queue
import threading
from typing import Optional


class SpeechEngine:
    """Threaded speech wrapper to avoid blocking the video loop."""

    def __init__(self, rate: int = 170, volume: float = 1.0) -> None:
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._rate = rate
        self._volume = volume
        self._thread.start()

    def _worker(self) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)

        while not self._stop_event.is_set():
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as ex:
                print(f"[WARN] TTS failed: {ex}")

        engine.stop()

    def say(self, text: str) -> None:
        if self._queue.qsize() < 5:
            self._queue.put(text)

    def close(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=2.0)
