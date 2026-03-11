from __future__ import annotations

import queue
import threading
from importlib import import_module
from typing import Optional


def rate_to_sapi(rate: int) -> int:
    return int(max(-10, min(10, round((int(rate) - 170) / 15))))


class SpeechEngine:
    def __init__(self, rate: int = 180, volume: float = 1.0) -> None:
        self.queue: queue.Queue[Optional[str]] = queue.Queue()
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.rate = int(rate)
        self.volume = float(max(0.0, min(1.0, volume)))
        self.thread.start()

    def clear_queue(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                if item is None:
                    self.queue.put(None)
                    break
        except queue.Empty:
            return

    def worker(self) -> None:
        speaker = None
        tts = None
        py_com = None
        use_sapi = False
        try:
            py_com = import_module("pythoncom")
            win32_client = import_module("win32com.client")
            dispatch = getattr(win32_client, "Dispatch")

            py_com.CoInitialize()
            speaker = dispatch("SAPI.SpVoice")
            speaker.Rate = rate_to_sapi(self.rate)
            speaker.Volume = int(self.volume * 100)
            use_sapi = True
        except Exception:
            use_sapi = False

        if not use_sapi:
            import pyttsx3

            tts = pyttsx3.init()
            tts.setProperty("rate", self.rate)
            tts.setProperty("volume", self.volume)

        try:
            while not self.stop_flag.is_set():
                text = self.queue.get()
                if text is None:
                    break
                try:
                    if use_sapi and speaker is not None:
                        speaker.Speak(text, 3)
                    elif tts is not None:
                        tts.say(text)
                        tts.runAndWait()
                except Exception:
                    pass
        finally:
            if tts is not None:
                tts.stop()
            if py_com is not None:
                try:
                    py_com.CoUninitialize()
                except Exception:
                    pass

    def say_latest(self, text: str) -> None:
        if not text.strip():
            return
        self.clear_queue()
        self.queue.put(text)

    def stop_current(self) -> None:
        self.clear_queue()

    def close(self) -> None:
        self.stop_flag.set()
        self.queue.put(None)
        self.thread.join(timeout=2.0)
