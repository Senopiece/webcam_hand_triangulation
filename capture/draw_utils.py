from typing import Any, List, Tuple
import cv2
import numpy as np
import mediapipe as mp
from .projection import distorted_project
from .models import CameraParams

mp_hands = mp.solutions.hands  # type: ignore

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (255, 255, 255)  # Main text color
thickness = 1
background_color = (0, 0, 0)  # Black rectangle color


def draw_left_top(y: int, text: str, frame: np.ndarray):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = 10  # Padding from the left
    text_y = 20 + y * (text_size[1] + 7)  # Dynamic top offset

    # Calculate the rectangle coordinates
    rect_x1 = text_x - 5
    rect_y1 = text_y - text_size[1] - 5
    rect_x2 = text_x + text_size[0] + 5
    rect_y2 = text_y + 5

    # Draw the rectangle
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, -1)

    # Draw the text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)


def draw_right_bottom(y: int, text: str, frame: np.ndarray):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = frame.shape[1] - text_size[0] - 10  # Padding from the right
    text_y = frame.shape[0] - y * (text_size[1] + 7) - 10  # Dynamic bottom offset

    # Calculate the rectangle coordinates
    rect_x1 = text_x - 5
    rect_y1 = text_y - text_size[1] - 5
    rect_x2 = text_x + text_size[0] + 5
    rect_y2 = text_y + 5

    # Draw the rectangle
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, -1)

    # Draw the text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)


def draw_origin_landmarks(landmarks: List[List[Any]], frames: List[np.ndarray]):
    for origin_landmarks, frame in zip(landmarks, frames):
        if origin_landmarks is None:
            continue

        h, w, _ = frame.shape
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_pt = origin_landmarks[start_idx]
            end_pt = origin_landmarks[end_idx]
            cv2.line(
                frame,
                (int(start_pt.x * w), int(start_pt.y * h)),
                (int(end_pt.x * w), int(end_pt.y * h)),
                color=(0, 200, 200),
                thickness=1,
            )
        for lm in origin_landmarks:
            cv2.circle(
                frame,
                (int(lm.x * w), int(lm.y * h)),
                radius=3,
                color=(0, 200, 200),
                thickness=-1,
            )


def draw_reprojected_landmarks(
    points_3d: List[np.ndarray],
    frames: List[np.ndarray],
    cameras_params: List[CameraParams],
    chosen_cams: List[List[int]],
):
    if points_3d:
        for pov_i, (frame, params) in enumerate(zip(frames, cameras_params)):
            # Project 3D points onto each camera
            reprojected_lms: List[Tuple[int, int]] = []
            for point_3d in points_3d:
                x, y = distorted_project(
                    point_3d,
                    params.extrinsic.rvec,
                    params.extrinsic.T,
                    params.intrinsic.mtx,
                    params.intrinsic.dist_coeffs,
                )

                # camera pixel coordinates -> normalized coordinates
                x, y = x / params.size[0], y / params.size[1]

                # normalized coordinates -> real viewport pixel coordinates
                h, w, _ = frame.shape
                x, y = x * w, y * h

                # Clip to image size
                x = max(min(int(x), frame.shape[1] - 1), 0)
                y = max(min(int(y), frame.shape[0] - 1), 0)

                reprojected_lms.append((x, y))

            # Draw reptojected landmarks
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_pt = reprojected_lms[start_idx]
                end_pt = reprojected_lms[end_idx]
                cv2.line(
                    frame,
                    start_pt,
                    end_pt,
                    color=(255, 255, 255),
                    thickness=1,
                )
            for involved_in_triangulating_this_lm, lm in zip(
                chosen_cams, reprojected_lms
            ):
                if pov_i in involved_in_triangulating_this_lm:
                    color = (0, 255, 0)  # Chosen camera
                else:
                    color = (255, 0, 0)  # Others
                cv2.circle(frame, lm, radius=3, color=color, thickness=-1)
