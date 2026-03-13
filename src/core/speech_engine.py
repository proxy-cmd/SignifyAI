import queue
import threading


def _to_sapi_rate(rate):
    return int(max(-10, min(10, round((int(rate) - 170) / 15))))


class Speaker:
    def __init__(self, rate=180, volume=1.0):
        self.rate = int(rate)
        self.volume = float(max(0.0, min(1.0, volume)))
        self.queue = queue.Queue()
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _clear(self):
        try:
            while True:
                item = self.queue.get_nowait()
                if item is None:
                    self.queue.put(None)
                    break
        except queue.Empty:
            return

    def _worker(self):
        tts = None
        speaker = None
        py_com = None
        use_sapi = False
        try:
            import pythoncom
            import win32com.client

            py_com = pythoncom
            py_com.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = _to_sapi_rate(self.rate)
            speaker.Volume = int(self.volume * 100)
            use_sapi = True
        except Exception:
            import pyttsx3

            tts = pyttsx3.init()
            tts.setProperty("rate", self.rate)
            tts.setProperty("volume", self.volume)

        try:
            while not self.stop_evt.is_set():
                msg = self.queue.get()
                if msg is None:
                    break
                try:
                    if use_sapi and speaker is not None:
                        speaker.Speak(msg, 3)
                    elif tts is not None:
                        tts.say(msg)
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

    def say_latest(self, text):
        if not text.strip():
            return
        self._clear()
        self.queue.put(text)

    def close(self):
        self.stop_evt.set()
        self.queue.put(None)
        self.thread.join(timeout=2.0)
