import cv2
import mediapipe as mp
import pythoncom
from win32com.client import Dispatch
from datetime import datetime
import time
import threading
import queue

def get_time_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    if 17 <= hour < 21:
        return "Good evening"
    return "Good night"


def speech_worker(speech_queue, stop_event):
    """Background worker so TTS does not block the camera loop."""
    pythoncom.CoInitialize()
    speaker = Dispatch("SAPI.SpVoice")
    while not stop_event.is_set():
        try:
            text = speech_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if text is None:
            break

        print(f"[SPEAK] {text}")
        try:
            speaker.Speak(text)  # blocking in worker thread, non-blocking for camera loop
        except Exception as ex:
            print(f"TTS error: {ex}")
    pythoncom.CoUninitialize()


def speak(text, min_gap=0.8, force=False):
    """Queue speech events with cooldown to avoid spam."""
    global last_speak_time
    now = time.time()
    if not force and (now - last_speak_time < min_gap):
        return
    if speech_queue.qsize() < 3:
        speech_queue.put(text)
        last_speak_time = now


last_speak_time = 0.0
prev_hand_count = 0
hand_present = False
no_hand_frames = 0
no_hand_reset_threshold = 5
speech_queue = queue.Queue()
speech_stop_event = threading.Event()
speech_thread = threading.Thread(
    target=speech_worker,
    args=(speech_queue, speech_stop_event),
    daemon=True
)
speech_thread.start()

# Camera window
win_name = "Proxy Hand Tracker AI"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 920, 680)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Failed to open camera.")
    raise SystemExit(1)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Warm up camera
for _ in range(10):
    camera.read()

print("Press 'v' for manual voice test, 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_count = 0
    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Display label
    if hand_count == 0:
        predicted_label = "NO_HAND"
    elif hand_count == 1:
        predicted_label = "HELLO"
    else:
        predicted_label = get_time_greeting().upper()

    cv2.putText(
        frame, f"Hands: {hand_count}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.putText(
        frame, f"Label: {predicted_label}", (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
    )

    # Debounced presence tracking:
    # mark "no hand" only after a few consecutive empty frames.
    if hand_count == 0:
        no_hand_frames += 1
        if no_hand_frames >= no_hand_reset_threshold:
            hand_present = False
    else:
        no_hand_frames = 0

        # Hand appeared again after being absent.
        if not hand_present:
            if hand_count == 1:
                speak("Hello", force=True)
            else:
                speak(get_time_greeting(), force=True)
            hand_present = True

        # Upgrade from 1 hand to 2+ hands while still present.
        elif prev_hand_count < 2 and hand_count >= 2:
            speak(get_time_greeting(), force=True)

    prev_hand_count = hand_count

    cv2.imshow(win_name, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("v"):
        speak("Voice test working", force=True)
    elif key == ord("q"):
        break

    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()
hands.close()
speech_stop_event.set()
speech_queue.put(None)
speech_thread.join(timeout=2.0)
