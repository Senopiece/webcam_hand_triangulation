import sys
import cv2
import time
import threading

# Global flag to signal all threads to stop
stop_threads = False


def capture_camera(camera_id):
    global stop_threads
    frame_counter = 0
    fps = 0
    start_time = time.time()

    cap = cv2.VideoCapture(camera_id)

    # Set 60 fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Disable autofocus
    autofocus_supported = cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1
    if autofocus_supported:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Set manual focus value
    focus_value = 260
    focus_supported = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
    if not focus_supported:
        print(
            f"Camera {camera_id} does not support manual focus! (or an invalid focus value provided)",
            file=sys.stderr,
        )
        sys.exit(1)

    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {camera_id} failed to capture a frame.")
            break

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        frame_counter += 1

        if time.time() - start_time >= 1:
            fps = frame_counter
            frame_counter = 0
            start_time = time.time()

        cv2.imshow(f"Camera {camera_id} stream", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_threads = True
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_ids = [1, 2, 3, 4]
    threads = []

    for camera_id in camera_ids:
        thread = threading.Thread(target=capture_camera, args=(camera_id,))
        thread.start()
        threads.append(thread)

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop_threads = True
        for thread in threads:
            thread.join()
