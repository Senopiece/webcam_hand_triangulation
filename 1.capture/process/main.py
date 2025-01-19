import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

import asyncio
import multiprocessing
from multiprocessing import Manager
from typing import List, Tuple
import mediapipe as mp
import argparse
import sys
import time

import numpy as np

from workers.pool import AsyncWorkersPool
from hands import AsyncHandsThreadedBuildinSolution
from cam_conf import load_cameras_parameters
from models import ContextedLandmark, PoV
from landmark2pixel_coord import landmark_to_pixel_coord
from distortion import undistort_pixel_coord

from triangulation import triangulate_lmcs
from projection import distorted_project

mp_hands = mp.solutions.hands

num_landmarks = 21  # MediaPipe Hands has 21 landmarks

def cap_reading_process(idx: int, run_bool, last_frame, cam_param) -> None:
    """
    Capture frames from the camera in a dedicated process, and store them
    in last_frame[idx]. The while loop runs until run_bool is False.
    """
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
    fps_counter = 0
    fps_display_time = time.time()

    while run_bool.value:  # Keep capturing until signaled to stop
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read from camera {idx}", file=sys.stderr)
            break

        # Store frame in shared dict (pickling overhead!)
        # We do a .copy() so the data is stable when pickled.
        last_frame[idx] = frame.copy()

        # Simple FPS measure
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_display_time >= 1.0:
            print(f"Cam {idx} fps:", fps_counter)
            fps_display_time = current_time
            fps_counter = 0

    cap.release()
    print(f"Camera {idx} process finishing.")

async def main():
    parser = argparse.ArgumentParser(
        description="3D Hand Reconstruction using MediaPipe and Multiple Cameras"
    )
    parser.add_argument(
        "--dfile",
        type=str,
        default="cameras.def.json5",
        help="Path to the cameras declarations file",
    )
    parser.add_argument(
        "--cfile",
        type=str,
        default="cameras.calib.json5",
        help="Path to the cameras calibration file",
    )
    parser.add_argument(
        "--window_scale",
        type=float,
        default=0.7,
        help="Scale of a window",
    )
    parser.add_argument(
        "-r",
        "--render",
        help="Forward captured 3d points to render in the 3d view",
        action="store_true",
    )
    parser.add_argument(
        "--division",
        type=int,
        default=8,
        help="Number of the hand tracking worker pool per camera",
    )
    args = parser.parse_args()
    do_render = args.render
    division = args.division

    # Load camera parameters
    cameras_params = load_cameras_parameters(args.dfile, args.cfile)
    if len(cameras_params) < 2:
        print("Need at least two cameras with calibration data.")
        sys.exit(1)

    # Use a Manager to share data between processes
    manager = Manager()

    # Shared dictionary for the frames from each camera process
    # Initialize all to None
    last_frame = manager.dict({idx: None for idx in cameras_params.keys()})

    # Shared boolean value to control the capture loop
    run_bool = manager.Value('b', True)

    # Create PoV objects and start each capture as a separate process
    povs: List[PoV] = []
    for idx, cam_param in cameras_params.items():
        cv2.namedWindow(f"Camera_{idx}", cv2.WINDOW_AUTOSIZE)

        # Start camera capture in a new process
        cap_process = multiprocessing.Process(
            target=cap_reading_process,
            args=(idx, run_bool, last_frame, cam_param),
        )
        cap_process.start()

        # Initialize hands trackers for each camera
        tracker_pool = AsyncWorkersPool(
            [AsyncHandsThreadedBuildinSolution() for _ in range(division)]
        )
        # Notice the renamed attribute in PoV: cap_process instead of cap_thread
        povs.append(
            PoV(
                cam_idx=idx,
                cap_process=cap_process,  # Updated attribute name here
                tracker=tracker_pool,
                parameters=cam_param,
            )
        )

    async def feeding_loop():
        """
        This loop sends the latest frame from each camera to the appropriate
        AsyncHandsThreadedBuildinSolution worker for hand landmark detection.
        """
        target_frame_interval = 1 / 60.0  # ~60 FPS

        # Wait until at least one frame is available from all cameras
        while True:
            if all(last_frame[idx] is not None for idx in cameras_params.keys()):
                break
            await asyncio.sleep(0.1)

        fps_counter = 0
        fps_display_time = time.time()

        while True:
            start_time = time.time()

            tasks = []
            for pov in povs:
                idx = pov.cam_idx
                # last_frame is a Manager dict, so we do an explicit copy for local usage
                frame = last_frame[idx]
                if frame is not None:
                    frame_copy = frame.copy()  # local copy
                    tasks.append(pov.tracker.send(frame_copy))

            # Send frames to the tracking workers
            await asyncio.gather(*tasks)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                # Stop capturing loop
                break

            # Print send FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_display_time >= 1.0:
                print("Send fps:", fps_counter)
                print(
                    "Idle workers:",
                    [pov.tracker.idle_workers.qsize() for pov in povs],
                )
                print(
                    "Results queue size:",
                    [pov.tracker.results.qsize() for pov in povs],
                )
                fps_display_time = current_time
                fps_counter = 0

            # Rate-limit to ~60 FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, target_frame_interval - elapsed_time)
            await asyncio.sleep(sleep_time)

    async def consuming_loop():
        """
        This loop consumes the processed hand landmarks from each AsyncHandsThreadedBuildinSolution,
        performs triangulation, and visualizes results.
        """
        fps_counter = 0
        fps_display_time = time.time()

        # Keep running until signaled to stop, but also wait for any leftover results
        while (
            run_bool.value
            or any(not pov.tracker.idle_workers.full() for pov in povs)
            or any(not pov.tracker.results.empty() for pov in povs)
        ):
            # Gather results from each camera
            results: List[Tuple[np.ndarray, cv2.typing.MatLike]] = await asyncio.gather(
                *[pov.tracker.results.get() for pov in povs]
            )

            # Update each POV with the new landmarks
            for pov, (hand_landmarks, frame) in zip(povs, results):
                pov.hand_landmarks = hand_landmarks
                pov.frame = frame

            # Triangulate points across the cameras
            chosen_cams = []
            points_3d = []
            povs_with_landmarks = [pov for pov in povs if pov.hand_landmarks is not None]
            if len(povs_with_landmarks) >= 2:
                for lm_id in range(num_landmarks):
                    # Prepare landmark contexts
                    lmcs = []
                    for p in povs_with_landmarks:
                        pixel_pt = landmark_to_pixel_coord(
                            p.frame.shape, p.hand_landmarks[lm_id]
                        )
                        undistorted_lm = undistort_pixel_coord(
                            pixel_pt,
                            p.parameters.intrinsic.mtx,
                            p.parameters.intrinsic.dist_coeffs,
                        )
                        lmcs.append(
                            ContextedLandmark(
                                cam_idx=p.cam_idx,
                                P=p.parameters.P,
                                lm=undistorted_lm,
                            )
                        )

                    # Triangulate
                    chosen, point_3d = triangulate_lmcs(lmcs)
                    assert point_3d is not None

                    chosen_cams.append(chosen)
                    points_3d.append(point_3d)

            # Print consume FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_display_time >= 1.0:
                print("Consume fps:", fps_counter)
                fps_display_time = current_time
                fps_counter = 0

            # Draw on frames and display them
            for pov in povs:
                idx = pov.cam_idx
                frame = pov.frame

                if frame is not None and len(points_3d) == 21:
                    # Project 3D points onto each camera
                    reprojected_lms = {}
                    for lm_id, point_3d in enumerate(points_3d):
                        x, y = distorted_project(
                            point_3d,
                            pov.parameters.extrinsic.rvec,
                            pov.parameters.extrinsic.T,
                            pov.parameters.intrinsic.mtx,
                            pov.parameters.intrinsic.dist_coeffs,
                        )
                        # Clip to image size
                        x = max(min(int(x), frame.shape[1] - 1), 0)
                        y = max(min(int(y), frame.shape[0] - 1), 0)
                        reprojected_lms[lm_id] = (x, y)

                    # Draw connections first
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        if (
                            start_idx in reprojected_lms
                            and end_idx in reprojected_lms
                        ):
                            start_pt = reprojected_lms[start_idx]
                            end_pt = reprojected_lms[end_idx]
                            cv2.line(
                                frame,
                                start_pt,
                                end_pt,
                                color=(255, 255, 255),
                                thickness=2,
                            )

                    # Draw landmarks
                    for lm_id, point_3d in enumerate(points_3d):
                        x, y = reprojected_lms[lm_id]
                        if idx in chosen_cams[lm_id]:
                            color = (0, 255, 0)  # Chosen camera
                        else:
                            color = (255, 0, 0)  # Others
                        cv2.circle(frame, (x, y), radius=5, color=color, thickness=-1)

                # Resize for display
                if frame is not None:
                    frame_height, frame_width = frame.shape[:2]
                    new_width = int(frame_width * args.window_scale)
                    new_height = int(frame_height * args.window_scale)
                    resized_frame = cv2.resize(
                        frame, (new_width, new_height), interpolation=cv2.INTER_AREA
                    )
                    cv2.imshow(f"Camera_{idx}", resized_frame)

    # Launch the consuming loop as a background task
    consuming_task = asyncio.create_task(consuming_loop())

    # Run the feeding loop in the foreground; it exits when user presses 'q'
    await feeding_loop()

    # Signal all camera processes to stop
    run_bool.value = False

    # Wait for the consuming loop to finish
    await consuming_task

    # Join all camera processes
    for pov in povs:
        # Now we call join on pov.cap_process
        pov.cap_process.join()

    # Dispose of trackers
    await asyncio.gather(*[pov.tracker.dispose() for pov in povs])

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Required check on Windows to avoid spawning extra processes recursively
    asyncio.run(main())
