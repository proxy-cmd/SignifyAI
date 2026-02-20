import cv2
import mediapipe as mp
import pyttsx3
import time

cv2.namedWindow("Proxy Hand Tracker AI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Proxy Hand Tracker AI", 920, 680)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Failed to open camera.")
    raise SystemExit(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

engine = pyttsx3.init()
last_spoken_label = None
last_spoken_time = 0.0
cooldown_sec = 1.5

# Camera warm-up
for _ in range(10):
    camera.read()

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
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

    predicted_label = "HELLO" if hand_count > 0 else "NO_HAND"

    cv2.putText(
        frame,
        f"Hands: {hand_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Label: {predicted_label}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )

    now = time.time()
    if predicted_label != last_spoken_label and (now - last_spoken_time) > cooldown_sec:
        engine.say(predicted_label)
        engine.runAndWait()
        last_spoken_label = predicted_label
        last_spoken_time = now

    cv2.imshow("Proxy Hand Tracker AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if cv2.getWindowProperty("Proxy Hand Tracker AI", cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()
hands.close()
engine.stop()
