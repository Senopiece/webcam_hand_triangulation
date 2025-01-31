import multiprocessing
import multiprocessing.synchronize
import sys
from typing import Tuple
import cv2
import numpy as np

from wrapped import Wrapped
from fps_counter import FPSCounter
from models import CameraParams


def cap_reading(
        idx: int,
        stop_event: multiprocessing.synchronize.Event,
        my_last_frame: Wrapped[Tuple[np.ndarray, int] | None],
        cam_param: CameraParams,
    ):
    # Initialize video capture
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera {idx}", file=sys.stderr)
        sys.exit(1)

    # Set resolution and fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_param.size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_param.size[1])
    cap.set(cv2.CAP_PROP_FPS, cam_param.fps)

    # Try disabling autofocus
    autofocus_supported = cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1
    if autofocus_supported:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Set manual focus value
    focus_value = cam_param.focus
    focus_supported = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
    if not focus_supported:
        print(
            f"Camera {idx} does not support manual focus! (or invalid focus value)",
            file=sys.stderr,
        )
        sys.exit(1)

    # FPS tracking variables
    fps_counter = FPSCounter()

    while True:
        if stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read from camera {idx}", file=sys.stderr)
            break

        my_last_frame.set((frame, fps_counter.get_fps()))
        fps_counter.count()

    cap.release()
    print(f"Camera {idx} finished.")