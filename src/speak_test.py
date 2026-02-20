import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 170)   # speaking speed
engine.setProperty("volume", 1.0) # 0.0 to 1.0

engine.say("Signify AI voice test successful.")
engine.runAndWait()
engine.stop()
