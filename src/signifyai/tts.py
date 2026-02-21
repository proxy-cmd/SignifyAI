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
            speaker.Rate = 0
            speaker.Volume = int(max(0, min(100, self._volume * 100)))
            use_sapi = True
        except Exception:
            use_sapi = False

        if not use_sapi:
            import pyttsx3

            pyttsx_engine = pyttsx3.init()
            pyttsx_engine.setProperty("rate", self._rate)
            pyttsx_engine.setProperty("volume", self._volume)

        try:
            while not self._stop_event.is_set():
                text = self._queue.get()
                if text is None:
                    break
                try:
                    if use_sapi and speaker is not None:
                        speaker.Speak(text)
                    elif pyttsx_engine is not None:
                        pyttsx_engine.say(text)
                        pyttsx_engine.runAndWait()
                except Exception as ex:
                    print(f"[WARN] TTS failed: {ex}")
        finally:
            if pyttsx_engine is not None:
                pyttsx_engine.stop()
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def say(self, text: str) -> None:
        if self._queue.qsize() < 5:
            self._queue.put(text)

    def close(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=2.0)
