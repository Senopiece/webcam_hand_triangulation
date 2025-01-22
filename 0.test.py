import numpy as np

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

cap = cv2.VideoCapture(4)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Set 60 fps
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
cap.set(cv2.CAP_PROP_FPS, 60)

# Disable autofocus
autofocus_supported = cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1
if autofocus_supported:
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

# Set manual focus value
focus_supported = cap.set(cv2.CAP_PROP_FOCUS, 260)
if not focus_supported:
    print(
        f"Camera {idx} does not support manual focus! (or invalid focus value provided)",
        file=sys.stderr,
    )
    sys.exit(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()