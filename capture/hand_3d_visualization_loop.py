import cv2
import multiprocessing
import multiprocessing.synchronize
import numpy as np
from typing import Tuple
import mediapipe as mp

from .projection import compute_sphere_rotating_camera_projection_matrix, project
from .finalizable_queue import EmptyFinalized, FinalizableQueue
from .fps_counter import FPSCounter
from .draw_utils import draw_left_top, draw_right_bottom


mp_hands = mp.solutions.hands  # type: ignore


def hand_3d_visualization_loop(
    window_size: Tuple[int, int],
    stop_event: multiprocessing.synchronize.Event,
    hand_points_queue: FinalizableQueue,
):
    window_title = "3D visualization"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    fps_counter = FPSCounter()
    frame = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

    # Mouse interaction
    intertia_absorbtion_k = 0.95
    sensetivity = 0.01

    grabbed = False
    last_pos = None
    inertia = (0, 0)
    inertia_tracked_last_pos = None

    # Wheel interaction
    min_camera_distance_to_target = 10
    max_camera_distance_to_target = 1000
    camera_distance_delta = 10

    # Camera parameters
    camera_target = np.array([0, 70, 0])
    camera_fov = 60  # Field of view in degrees
    camera_near = 0.1  # Near clipping plane
    camera_far = 100.0  # Far clipping plane
    camera_distance_to_target = (
        300  # Distance from the camera to the target to maintain
    )
    camera_roll = 0
    camera_spherical_position = [0, 0]  # [azimuth, elevation] in radians

    def update_camera_spherical_position(delta):
        nonlocal camera_spherical_position
        camera_spherical_position[0] += delta[0]
        camera_spherical_position[1] += delta[1]
        camera_spherical_position[1] = np.clip(
            camera_spherical_position[1], -1.1, 1.1
        )  # Limit pitch

    def handle_mouse_event(event, x, y, flags, param):
        nonlocal grabbed, last_pos, inertia, inertia_tracked_last_pos, camera_distance_to_target, camera_spherical_position

        if event == cv2.EVENT_LBUTTONDOWN:
            grabbed = True
            last_pos = (x, y)
            inertia_tracked_last_pos = None
            inertia = (0, 0)

        elif event == cv2.EVENT_LBUTTONUP:
            grabbed = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                camera_distance_to_target = max(
                    min_camera_distance_to_target,
                    camera_distance_to_target - camera_distance_delta,
                )  # Decrease distance (zoom in)
            else:
                camera_distance_to_target = min(
                    max_camera_distance_to_target,
                    camera_distance_to_target + camera_distance_delta,
                )  # Increase distance (zoom out)

        elif event == cv2.EVENT_MOUSEMOVE:
            if last_pos is not None:
                delta = (x - last_pos[0]) * sensetivity, (y - last_pos[1]) * sensetivity
            else:
                delta = 0, 0

            if grabbed:
                update_camera_spherical_position(delta)

            last_pos = (x, y)

    cv2.setMouseCallback(window_title, handle_mouse_event)

    while True:
        try:
            result = hand_points_queue.get()
            hand_points, coupling_fps, debt_size = result
        except EmptyFinalized:
            break

        # Clear prev frame
        frame.fill(0)

        # Update and apply inertia
        if grabbed:
            # Accamulate inertia
            if last_pos is None:
                inertia = (0, 0)
            elif inertia_tracked_last_pos is not None:
                delta = (last_pos[0] - inertia_tracked_last_pos[0]) * sensetivity, (
                    last_pos[1] - inertia_tracked_last_pos[1]
                ) * sensetivity
                inertia = delta
            inertia_tracked_last_pos = last_pos
        else:
            # Release inertia
            inertia = (
                inertia[0] * intertia_absorbtion_k,
                inertia[1] * intertia_absorbtion_k,
            )
            update_camera_spherical_position(inertia)

        # calculate projection matrix
        P = compute_sphere_rotating_camera_projection_matrix(
            np.radians(camera_fov),
            camera_near,
            camera_far,
            camera_spherical_position,
            camera_roll,
            camera_distance_to_target,
            camera_target,
            frame.shape[1],
            frame.shape[0],
        )

        # Project points
        landmarks = [project(point_3d, P) for point_3d in hand_points]

        if len(landmarks) != 0:
            # Draw hand connections
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_pt, z_start = landmarks[start_idx]
                end_pt, z_end = landmarks[end_idx]

                # Skip drawing if either point is behind the camera
                if z_start <= 0 or z_end <= 0:
                    continue

                # Draw the line
                cv2.line(
                    frame,
                    (int(start_pt[0]), int(start_pt[1])),
                    (int(end_pt[0]), int(end_pt[1])),
                    color=(255, 255, 255),
                    thickness=1,
                )

            # Draw landmarks (circles)
            for lm, z in landmarks:
                # Skip drawing if the point is behind the camera
                if z <= 0:
                    continue

                # Draw the circle
                cv2.circle(
                    frame,
                    (int(lm[0]), int(lm[1])),
                    radius=3,
                    color=(0, 255, 0),
                    thickness=-1,
                )

        # Draw camera target with axis lines
        camera_target_proj, z_target = project(camera_target, P)
        if z_target > 0:  # Only draw the target if it's in front of the camera
            for axis_i in range(3):
                axis_end, z_axis = project(camera_target + np.eye(3)[axis_i] * 10, P)

                # Skip drawing if either end of the axis line is behind the camera
                if z_target <= 0 or z_axis <= 0:
                    continue

                cv2.line(
                    frame,
                    (int(camera_target_proj[0]), int(camera_target_proj[1])),
                    (int(axis_end[0]), int(axis_end[1])),
                    color=np.eye(3)[-axis_i - 1] * 255,
                    thickness=1,
                )

        # Draw coupling fps
        draw_right_bottom(1, f"Couple FPS: {coupling_fps}", frame)
        draw_right_bottom(0, f"Debt: {debt_size}", frame)

        # Draw FPS text on the frame
        fps_counter.count()
        draw_left_top(0, f"FPS: {fps_counter.get_fps()}", frame)

        # Update the frame
        cv2.imshow(window_title, frame)

        # Maybe stop
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            # Stop capturing loop
            stop_event.set()

        hand_points_queue.task_done()

    print("3D visualization loop finished.")
