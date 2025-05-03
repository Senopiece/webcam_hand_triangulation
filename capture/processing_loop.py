import cv2
import numpy as np
from typing import Any, Callable, List, Tuple
import mediapipe as mp

from .hand_utils import rm_th_base
from .hand_normalization import normalize_hand
from .kinematics import inverse_hand_angles_by_landmarks
from .models import CameraParams
from .finalizable_queue import EmptyFinalized, FinalizableQueue
from .draw_utils import draw_left_top, draw_origin_landmarks, draw_reprojected_landmarks
from .hand_triangulator import HandTriangulator

mp_hands = mp.solutions.hands  # type: ignore
num_landmarks = 21  # MediaPipe Hands has 21 landmarks


def processing_loop(
    landmark_transforms: List[Callable[..., Any]],
    to_draw_origin_landmarks: bool,
    desired_window_size: Tuple[int, int],
    cameras_params: List[CameraParams],
    coupled_frames_queue: FinalizableQueue,
    results_queue: FinalizableQueue,
    display_queues: List[FinalizableQueue],
):
    triangulator = HandTriangulator(landmark_transforms, cameras_params)

    while True:
        try:
            elem = coupled_frames_queue.get()
        except EmptyFinalized:
            break

        index: int = elem[0]
        indexed_frames: List[Tuple[np.ndarray, int]] = elem[1]
        coupling_fps: int = elem[2]

        cap_fps: List[int] = [item[1] for item in indexed_frames]
        frames: List[np.ndarray] = [item[0] for item in indexed_frames]

        del indexed_frames

        landmarks, chosen_cams, points_3d = triangulator.triangulate(frames)

        # Send to 3d visualization
        results_queue.put(
            (
                index,
                (
                    (
                        inverse_hand_angles_by_landmarks(
                            normalize_hand(rm_th_base(points_3d))
                        )
                        if points_3d
                        else None
                    ),
                    coupling_fps,
                ),
            )
        )

        # Resize frames before drawing
        for i, frame in enumerate(frames):
            frames[i] = cv2.resize(
                frame, desired_window_size, interpolation=cv2.INTER_AREA
            )

        # Draw original landmarks
        if to_draw_origin_landmarks:
            draw_origin_landmarks(landmarks, frames)

        # Draw reprojected landmarks
        draw_reprojected_landmarks(points_3d, frames, cameras_params, chosen_cams)

        # Draw cap fps for every pov
        for fps, frame in zip(cap_fps, frames):
            draw_left_top(0, f"Capture FPS: {fps}", frame)

        # Write results
        for display_queue, frame in zip(display_queues, frames):
            display_queue.put((index, frame))

        coupled_frames_queue.task_done()

    triangulator.close()

    print("A processing loop is finished.")
