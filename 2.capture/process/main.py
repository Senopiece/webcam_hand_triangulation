from math import pi
import multiprocessing
import multiprocessing.synchronize
import os
import threading

import numpy as np

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

from typing import Any, Dict, List, Tuple
import mediapipe as mp
import argparse
import sys
import time

from cam_conf import load_cameras_parameters
from models import CameraParams, ContextedLandmark
from wrapped import Wrapped
from triangulation import triangulate_lmcs
from projection import compute_sphere_rotating_camera_projection_matrix, distorted_project, project
from finalizable_queue import EmptyFinalized, FinalizableQueue, ProcessFinalizableQueue, ThreadFinalizableQueue
from fps_counter import FPSCounter
from draw_utils import draw_left_top, draw_right_bottom
from hand_normalization import normalize_hand

mp_hands = mp.solutions.hands
num_landmarks = 21  # MediaPipe Hands has 21 landmarks


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


def coupling_loop(
        stop_event: multiprocessing.synchronize.Event,
        last_frame: List[Wrapped[Tuple[np.ndarray, int] | None]],
        coupled_frames_queue: FinalizableQueue,
    ):
    target_frame_interval = 1 / 30.0  # ~30 FPS

    # Wait until at least one frame is available from all cameras
    while True:
        if all(a_last_frame.get() is not None for a_last_frame in last_frame):
            break
        time.sleep(0.1)

    fps_counter = FPSCounter()
    index = 0

    while True:
        start_time = time.time()

        if stop_event.is_set():
            break

        fps_counter.count()

        frames = []
        for frame in last_frame:
            frame, fps = frame.get()
            frames.append((cv2.flip(frame, 1), fps))

        # Send coupled frames
        coupled_frames_queue.put((index, frames, fps_counter.get_fps()))
        index += 1

        # Rate-limit to ~60 FPS
        elapsed_time = time.time() - start_time
        sleep_time = max(0, target_frame_interval - elapsed_time)
        time.sleep(sleep_time)
    
    coupled_frames_queue.finalize()
    print("Coupling loop finished.")


def processing_loop(
        draw_origin_landmarks: bool,
        desired_window_size: Tuple[float, float],
        cameras_params: List[CameraParams],
        coupled_frames_queue: FinalizableQueue,
        hand_points_queue: FinalizableQueue,
        out_queues: List[FinalizableQueue],
    ):
    processors = [
        mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9,
        ) for _ in range(len(cameras_params))
    ]

    while True:
        try:
            elem = coupled_frames_queue.get()
        except EmptyFinalized:
            break

        index: int = elem[0]
        frames: List[Tuple[np.ndarray, int]] = elem[1]
        coupling_fps: int = elem[2]

        cap_fps: List[int] = [item[1] for item in frames]
        frames: List[np.ndarray] = [item[0] for item in frames]

        # Find landmarks
        landmarks = []
        for processor, frame in zip(processors, frames):
            # Convert to RGB and process
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = processor.process(frame_rgb)

            # Convert MediaPipe landmarks to plain Python list
            if res.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    res.multi_hand_landmarks, 
                    res.multi_handedness
                ):
                    if handedness.classification[0].label == "Left":
                        landmarks.append(hand_landmarks.landmark)
                        break
        
        povs_with_landmarks = [i for i, landmarks in enumerate(landmarks) if landmarks is not None]

        # Triangulate points across the cameras
        chosen_cams = []
        points_3d = []
        if len(povs_with_landmarks) >= 2:
            for lm_id in range(num_landmarks):
                lmcs = []
                for pov_i in povs_with_landmarks:
                    pov_params = cameras_params[pov_i]

                    # Landmark to pixel coord
                    lm = landmarks[pov_i][lm_id]
                    h, w, _ = frames[pov_i].shape
                    pixel_pt = [lm.x * w, lm.y * h]

                    # Undistort pixel coord
                    intrinsics = pov_params.intrinsic
                    undistorted_lm = cv2.undistortPoints(
                        np.array([[pixel_pt]], dtype=np.float32),
                        intrinsics.mtx,
                        intrinsics.dist_coeffs,
                        P=intrinsics.mtx,
                    )[0][0]

                    # Append the result
                    lmcs.append(
                        ContextedLandmark(
                            cam_idx=pov_i,
                            P=pov_params.P,
                            lm=undistorted_lm,
                        )
                    )

                chosen, point_3d = triangulate_lmcs(lmcs)

                chosen_cams.append(chosen)
                points_3d.append(point_3d)
        
        # Send to 3d visualization
        hand_points_queue.put((index, (normalize_hand(points_3d) if points_3d else [], coupling_fps, coupled_frames_queue.qsize())))
        
        # Resize frames before drawing
        for i, frame in enumerate(frames):
            frames[i] = cv2.resize(
                frame, desired_window_size, interpolation=cv2.INTER_AREA
            )

        # Draw original landmarks
        if draw_origin_landmarks:
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
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), radius=3, color=(0, 200, 200), thickness=-1)
        
        # Draw reprojected landmarks
        if len(povs_with_landmarks) >= 2:
            for pov_i, (frame, params) in enumerate(zip(frames, cameras_params)):
                # Project 3D points onto each camera
                reprojected_lms: List[Tuple[float, float]] = []
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
                for involved_in_triangulating_this_lm, lm in zip(chosen_cams, reprojected_lms):
                    if pov_i in involved_in_triangulating_this_lm:
                        color = (0, 255, 0)  # Chosen camera
                    else:
                        color = (255, 0, 0)  # Others
                    cv2.circle(frame, lm, radius=3, color=color, thickness=-1)
        
        # Draw cap fps for every pov
        for fps, frame in zip(cap_fps, frames):
            draw_left_top(0, f"Capture FPS: {fps}", frame)
        
        # Write results
        for out_queue, frame in zip(out_queues, frames):
            out_queue.put((index, frame))
        
        coupled_frames_queue.task_done()
    
    for processor in processors:
        processor.close()
    
    print("A processing loop is finished.")


def ordering_loop(
        in_queue: FinalizableQueue,
        out_queue: FinalizableQueue,
    ):
    expecting = 0
    unordered: Dict[int, Any] = {}
    while True:
        try:
            elem = in_queue.get()
        except EmptyFinalized:
            break

        index: int = elem[0]
        data: Any = elem[1]

        if expecting == index:
            out_queue.put(data)
            while True:
                expecting += 1
                data = unordered.get(expecting, None)
                if data is None:
                    break
                else:
                    del unordered[expecting]
                    out_queue.put(data)
        else:
            unordered[index] = data
        
        in_queue.task_done()
    
    out_queue.finalize()
    print("A ordering loop finished.")



def hand_3d_visualization_loop(
        window_size: Tuple[float, float],
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
    camera_distance_to_target = 300  # Distance from the camera to the target to maintain
    camera_roll = 0
    camera_spherical_position = [0, 0]  # [azimuth, elevation] in radians

    def update_camera_spherical_position(delta):
        nonlocal camera_spherical_position
        camera_spherical_position[0] += delta[0]
        camera_spherical_position[1] += delta[1]
        camera_spherical_position[1] = np.clip(camera_spherical_position[1], -1.1, 1.1)  # Limit pitch

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
                camera_distance_to_target = max(min_camera_distance_to_target, camera_distance_to_target - camera_distance_delta)  # Decrease distance (zoom in)
            else:
                camera_distance_to_target = min(max_camera_distance_to_target, camera_distance_to_target + camera_distance_delta)  # Increase distance (zoom out)

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
            hand_points, coupling_fps, debt_size = hand_points_queue.get()
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
                delta = (last_pos[0] - inertia_tracked_last_pos[0]) * sensetivity, (last_pos[1] - inertia_tracked_last_pos[1]) * sensetivity
                inertia = delta
            inertia_tracked_last_pos = last_pos
        else:
            # Release inertia
            inertia = inertia[0] * intertia_absorbtion_k, inertia[1] * intertia_absorbtion_k
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
                    thickness=-1
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
                    thickness=1
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


def display_loop(
        idx: int,
        stop_event: multiprocessing.synchronize.Event,
        frame_queue: FinalizableQueue,
    ):
    cv2.namedWindow(f"Camera_{idx}", cv2.WINDOW_AUTOSIZE)

    fps_counter = FPSCounter()

    while True:
        try:
            frame = frame_queue.get()
        except EmptyFinalized:
            break

        # Draw FPS text on the frame
        fps_counter.count()
        draw_left_top(1, f"Display FPS: {fps_counter.get_fps()}", frame)

        # Update the frame
        cv2.imshow(f"Camera_{idx}", frame)
        
        # Maybe stop
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            # Stop capturing loop
            stop_event.set()

        frame_queue.task_done()
    
    print(f"Display {idx} loop finished.")


def main():
    parser = argparse.ArgumentParser(
        description="3D Hand Reconstruction using MediaPipe and Multiple Cameras"
    )
    parser.add_argument(
        "--cfile",
        type=str,
        default="cameras.calib.json5",
        help="Path to the cameras calibration file",
    )
    parser.add_argument(
        "--window_size",
        type=str,
        default="448x336",
        help="Size of a preview window",
    )
    parser.add_argument(
        "--division",
        type=int,
        default=8,
        help="Number of the hand tracking worker pool per camera",
    )
    parser.add_argument(
        "-ol",
        "--origin_landmarks",
        help="Draw origin landmarks",
        action="store_true"
    )
    args = parser.parse_args()
    desired_window_size = tuple(map(int, args.window_size.split("x")))
    division = args.division
    draw_origin_landmarks = args.origin_landmarks

    # Load camera parameters
    cameras_params = load_cameras_parameters(args.cfile)
    if len(cameras_params) < 2:
        print("Need at least two cameras with calibration data.")
        sys.exit(1)
    
    cameras_ids = list(cameras_params.keys())

    # Shared
    cams_stop_event = multiprocessing.Event()
    last_frame: List[Wrapped[Tuple[np.ndarray, int] | None]] = [
        Wrapped()
        for _ in cameras_ids
    ]

    # Capture cameras
    caps: List[threading.Thread] = [
        threading.Thread(
            target=cap_reading,
            args=(
                idx,
                cams_stop_event,
                my_last_frame,
                cam_param,
            ),
            daemon=True,
        ) for my_last_frame, (idx, cam_param) in zip(last_frame, cameras_params.items())
    ]
    for process in caps:
        process.start()
    
    # Couple frames
    coupled_frames_queue = ThreadFinalizableQueue()
    coupling_worker = threading.Thread(
        target=coupling_loop,
        args=(
            cams_stop_event,
            last_frame,
            coupled_frames_queue,
        ),
        daemon=True,
    )
    coupling_worker.start()
    
    # Processing workers
    hand_points_queue = ThreadFinalizableQueue()
    processed_queues = [ThreadFinalizableQueue() for _ in cameras_ids]
    processing_loops_pool = [
        threading.Thread(
            target=processing_loop,
            args=(
                draw_origin_landmarks,
                desired_window_size,
                list(cameras_params.values()),
                coupled_frames_queue,
                hand_points_queue,
                processed_queues,
            ),
            daemon=True,
        ) for _ in range(division)
    ]
    for process in processing_loops_pool:
        process.start()
    
    # Sort hand points
    ordered_hand_points_queue = ProcessFinalizableQueue()
    hand_points_sorter = threading.Thread(
        target=ordering_loop,
        args=(
            hand_points_queue,
            ordered_hand_points_queue,
        ),
        daemon=True,
    )
    hand_points_sorter.start()

    # Visualize 3d hand
    hand_3d_visualizer = multiprocessing.Process(
        target=hand_3d_visualization_loop,
        args=(
            desired_window_size,
            cams_stop_event,
            ordered_hand_points_queue,
        ),
        daemon=True,
    )
    hand_3d_visualizer.start()

    # Sort processing workers output
    ordered_processed_queues = [ThreadFinalizableQueue() for _ in cameras_ids]
    display_ordering_loops = [
        threading.Thread(
            target=ordering_loop,
            args=(
                in_queue,
                out_queue
            ),
            daemon=True,
        ) for in_queue, out_queue in zip(processed_queues, ordered_processed_queues)
    ]
    for process in display_ordering_loops:
        process.start()
    
    # Displaying loops
    display_loops = [
        threading.Thread(
            target=display_loop,
            args=(
                idx,
                cams_stop_event,
                frame_queue
            ),
            daemon=True,
        ) for idx, frame_queue in zip(cameras_ids, ordered_processed_queues)
    ]
    for process in display_loops:
        process.start()

    # Wait for a stop signal
    cams_stop_event.wait()

    # Free resources
    print("Freeing resources...")
    coupling_worker.join()
    
    for worker in caps:
        worker.join()
    
    print("Waiting for lag to process...")
    coupling_worker.join()

    for worker in processing_loops_pool:
        worker.join()

    hand_points_queue.finalize()
    for queue in processed_queues:
        queue.finalize()
    
    hand_points_sorter.join()
    for worker in display_ordering_loops:
        worker.join()

    hand_3d_visualizer.join()
    for worker in display_loops:
        worker.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
