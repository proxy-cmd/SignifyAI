try:
    import cv2
except Exception as ex:  # pragma: no cover - simple runtime helper
    raise SystemExit(
        "OpenCV import failed in the current interpreter.\n"
        "Install with: python -m pip install opencv-python\n"
        f"Details: {ex}"
    )


if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Failed to open camera")
        raise SystemExit(1)

    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if cv2.getWindowProperty("Camera Test", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
