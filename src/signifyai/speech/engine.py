from __future__ import annotations

import queue
import threading
from typing import Optional


def _rate_to_sapi(rate: int) -> int:
    return int(max(-10, min(10, round((int(rate) - 170) / 15))))


class SpeechEngine:
    def __init__(self, rate: int = 180, volume: float = 1.0) -> None:
        self._q: queue.Queue[Optional[str]] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self.rate = int(rate)
        self.volume = float(max(0.0, min(1.0, volume)))
        self._thread.start()

    def _clear_queue(self) -> None:
        try:
            while True:
                item = self._q.get_nowait()
                if item is None:
                    self._q.put(None)
                    break
        except queue.Empty:
            return

    def _worker(self) -> None:
        speaker = None
        pyttsx_engine = None
        pythoncom = None
        use_sapi = False
        try:
            import pythoncom  # type: ignore
            from win32com.client import Dispatch  # type: ignore

            pythoncom.CoInitialize()
            speaker = Dispatch("SAPI.SpVoice")
            speaker.Rate = _rate_to_sapi(self.rate)
            speaker.Volume = int(self.volume * 100)
            use_sapi = True
        except Exception:
            use_sapi = False

        if not use_sapi:
            import pyttsx3

            pyttsx_engine = pyttsx3.init()
            pyttsx_engine.setProperty("rate", self.rate)
            pyttsx_engine.setProperty("volume", self.volume)

        try:
            while not self._stop.is_set():
                text = self._q.get()
                if text is None:
                    break
                try:
                    if use_sapi and speaker is not None:
                        speaker.Speak(text, 3)
                    elif pyttsx_engine is not None:
                        pyttsx_engine.say(text)
                        pyttsx_engine.runAndWait()
                except Exception:
                    pass
        finally:
            if pyttsx_engine is not None:
                pyttsx_engine.stop()
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def say_latest(self, text: str) -> None:
        if not text.strip():
            return
        self._clear_queue()
        self._q.put(text)

    def stop_current(self) -> None:
        self._clear_queue()

    def close(self) -> None:
        self._stop.set()
        self._q.put(None)
        self._thread.join(timeout=2.0)
