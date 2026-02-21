import time

from signifyai.tts import SpeechEngine


if __name__ == "__main__":
    sp = SpeechEngine(rate=170, volume=1.0)
    sp.say("Signify AI voice test successful.")
    time.sleep(2.0)
    sp.close()
