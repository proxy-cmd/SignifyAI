from __future__ import annotations

import queue
import threading
import time
from typing import Optional


def tts_rate_to_sapi_rate(rate: int) -> int:
    """Map pyttsx-style rate to SAPI rate range (-10..10)."""
    return int(max(-10, min(10, round((int(rate) - 170) / 15))))


class SpeechEngine:
    """Threaded speech wrapper to avoid blocking the video loop."""

    def __init__(
        self,
        rate: int = 170,
        volume: float = 1.0,
        dedup_sec: float = 0.35,
        min_gap_sec: float = 0.18,
        max_queue: int = 3,
    ) -> None:
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._rate = rate
        self._volume = volume
        self._dedup_sec = max(0.0, float(dedup_sec))
        self._min_gap_sec = max(0.0, float(min_gap_sec))
        self._max_queue = max(1, int(max_queue))
        self._state_lock = threading.Lock()
        self._last_enqueued_text = ""
        self._last_enqueued_ts = 0.0
        self._engine_lock = threading.Lock()
        self._speaker = None
        self._pyttsx_engine = None
        self._use_sapi = False
        self._thread.start()

    def _worker(self) -> None:
        # Prefer Windows SAPI for more reliable repeated speech events.
        use_sapi = False
        speaker = None
        pyttsx_engine = None
        pythoncom = None
        try:
            import pythoncom  # type: ignore
            from win32com.client import Dispatch  # type: ignore

            pythoncom.CoInitialize()
            speaker = Dispatch("SAPI.SpVoice")
            speaker.Rate = tts_rate_to_sapi_rate(self._rate)
            speaker.Volume = int(max(0, min(100, self._volume * 100)))
            use_sapi = True
        except Exception:
            use_sapi = False

        if not use_sapi:
            import pyttsx3

            pyttsx_engine = pyttsx3.init()
            pyttsx_engine.setProperty("rate", self._rate)
            pyttsx_engine.setProperty("volume", self._volume)

        with self._engine_lock:
            self._speaker = speaker
            self._pyttsx_engine = pyttsx_engine
            self._use_sapi = use_sapi

        try:
            while not self._stop_event.is_set():
                text = self._queue.get()
                if text is None:
                    break
                try:
                    if use_sapi and speaker is not None:
                        # Async + purge keeps speech responsive for rapid label updates.
                        speaker.Speak(text, 3)
                    elif pyttsx_engine is not None:
                        pyttsx_engine.say(text)
                        pyttsx_engine.runAndWait()
                except Exception as ex:
                    print(f"[WARN] TTS failed: {ex}")
        finally:
            if pyttsx_engine is not None:
                pyttsx_engine.stop()
            with self._engine_lock:
                self._speaker = None
                self._pyttsx_engine = None
                self._use_sapi = False
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def _should_enqueue(self, text: str, *, force: bool) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        now = time.time()
        with self._state_lock:
            if (
                normalized.casefold() == self._last_enqueued_text.casefold()
                and (now - self._last_enqueued_ts) < self._dedup_sec
            ):
                return False
            if (
                not force
                and (now - self._last_enqueued_ts) < self._min_gap_sec
                and self._queue.qsize() > 0
            ):
                return False
            self._last_enqueued_text = normalized
            self._last_enqueued_ts = now
        return True

    def say(self, text: str, *, force: bool = False) -> None:
        if not self._should_enqueue(text, force=force):
            return
        if self._queue.qsize() < self._max_queue:
            self._queue.put(text)

    def clear_pending(self) -> None:
        """Drop queued speech items that haven't started yet."""
        try:
            while True:
                item = self._queue.get_nowait()
                if item is None:
                    # keep shutdown sentinel semantics intact
                    self._queue.put(None)
                    break
        except queue.Empty:
            return

    def _interrupt_current(self) -> None:
        with self._engine_lock:
            speaker = self._speaker
            pyttsx_engine = self._pyttsx_engine
            use_sapi = self._use_sapi
        try:
            if use_sapi and speaker is not None:
                # Async + purge cancels the current utterance immediately.
                speaker.Speak("", 3)
            elif pyttsx_engine is not None:
                pyttsx_engine.stop()
        except Exception:
            return

    def say_latest(self, text: str) -> None:
        """Replace queued items with latest text to avoid backlog lag."""
        self._interrupt_current()
        self.clear_pending()
        self.say(text, force=True)

    def stop_current(self) -> None:
        self._interrupt_current()
        self.clear_pending()

    def close(self) -> None:
        self._stop_event.set()
        self._interrupt_current()
        self._queue.put(None)
        self._thread.join(timeout=2.0)
