import asyncio
import threading
from typing import Dict, List, Tuple
import cv2
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
        help="Number of the hand tracking workers pool per camera",
    )
    args = parser.parse_args()
    do_render = args.render
    division = args.division

    # Load camera parameters
    cameras_params = load_cameras_parameters(args.dfile, args.cfile)
    if len(cameras_params) < 2:
        print("Need at least two cameras with calibration data.")
        sys.exit(1)
    
    run = True

    # Initialize
    last_frame: Dict[int, np.ndarray] = {idx: None for idx in cameras_params.keys()}
    povs: List[PoV] = []
    for idx, cam_param in cameras_params.items():
        # Make window for the pov
        cv2.namedWindow(f"Camera_{idx}", cv2.WINDOW_AUTOSIZE)

        def cap_reading_thread(idx: int):
            # Initialize video capture
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print(f"Error: Could not open camera {idx}", out=sys.stderr)
                sys.exit(1)

            # Set 60 fps
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_param.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_param.size[1])
            cap.set(cv2.CAP_PROP_FPS, cam_param.fps)

            # Disable autofocus
            autofocus_supported = cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1
            if autofocus_supported:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            # Set manual focus value
            focus_value = cam_param.focus
            focus_supported = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
            if not focus_supported:
                print(
                    f"Camera {idx} does not support manual focus! (or an invalid focus value provided)",
                    file=sys.stderr,
                )
                sys.exit(1)
            
            # FPS tracking variables
            fps_counter = 0
            fps_display_time = time.time()
            
            while run:
                ret, frame = cap.read()

                if not ret:
                    print(f"Error: Could not read from camera {idx}", out=sys.stderr)
                    sys.exit(1)
                
                last_frame[idx] = frame

                # FPS counter update every second
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_display_time >= 1.0:
                    print(f"Cam {idx} fps:", fps_counter)
                    fps_display_time = current_time
                    fps_counter = 0
            
            cap.release()
        
        cap_thread = threading.Thread(target=cap_reading_thread, args=(idx,))
        cap_thread.start()

        # Initialize hands trackers for each camera
        povs.append(PoV(
            cam_idx=idx,
            cap_thread=cap_thread,
            tracker=AsyncWorkersPool(
                [AsyncHandsThreadedBuildinSolution() for _ in range(division)],
            ),
            parameters=cam_param,
        ),)
    
    async def feeding_loop():
        target_frame_interval = 1 / 60  # Approximately 0.01667 seconds

        while True:
            if all(last_frame[idx] is not None for idx in cameras_params.keys()):
                break

            await asyncio.sleep(0.1)
        
        # FPS tracking variables
        fps_counter = 0
        fps_display_time = time.time()

        while True:
            start_time = time.time()
            
            tasks = []
            for pov in povs:
                idx = pov.cam_idx
                frame = last_frame[idx].copy()

                tasks.append(pov.tracker.send(frame))

            await asyncio.gather(*tasks)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

            # FPS counter update every second
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_display_time >= 1.0:
                print("Send fps:", fps_counter)
                print("Idle workers:", [pov.tracker.idle_workers.qsize() for pov in povs])
                print("Results queue size:", [pov.tracker.results.qsize() for pov in povs])
                fps_display_time = current_time
                fps_counter = 0

            # Rate limit to have at max 60 FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, target_frame_interval - elapsed_time)
            await asyncio.sleep(sleep_time)

    async def consuming_loop():
        # FPS tracking variables
        fps_counter = 0
        fps_display_time = time.time()

        # Loop untill said to stop but make sure to process what remains
        while (
            run
            or any(
                not pov.tracker.idle_workers.full() for pov in povs
            )  # any channel is in processing -> new results may arrive
            or any(
                not pov.tracker.results.empty() for pov in povs
            )  # any channel has non empty results -> need to process them
        ):
            # NOTE: it will hang freeing if channels got not equal amounts of .send calls
            results: List[Tuple[np.ndarray, cv2.typing.MatLike]] = await asyncio.gather(
                *[pov.tracker.results.get() for pov in povs]
            )

            for pov, (hand_landmarks, frame) in zip(povs, results):
                pov.hand_landmarks = hand_landmarks
                pov.frame = frame

            # Triangulate points
            chosen_cams = []
            points_3d = []
            povs_with_landmarks = [
                pov for pov in povs if pov.hand_landmarks is not None
            ]
            if len(povs_with_landmarks) >= 2:
                for lm_id in range(num_landmarks):
                    # Prepare landmarks contexts
                    lmcs = []
                    for pov in povs_with_landmarks:
                        point = landmark_to_pixel_coord(pov.frame.shape, pov.hand_landmarks[lm_id])
                        undistorted_lm = undistort_pixel_coord(point, pov.parameters.intrinsic.mtx, pov.parameters.intrinsic.dist_coeffs)
                        lmcs.append(ContextedLandmark(cam_idx=pov.cam_idx, P=pov.parameters.P, lm=undistorted_lm))

                    # Triangulate
                    chosen, point_3d = triangulate_lmcs(lmcs)
                    assert point_3d is not None

                    # Append results
                    chosen_cams.append(chosen)
                    points_3d.append(point_3d)
            
            # FPS counter update every second
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

                # Draw landmarks if can
                if len(points_3d) == 21:
                    # Store reprojected 2D landmarks for drawing connections
                    reprojected_lms = {}

                    for lm_id, point_3d in enumerate(points_3d):
                        x, y = distorted_project(
                            point_3d,
                            pov.parameters.extrinsic.rvec,
                            pov.parameters.extrinsic.T,
                            pov.parameters.intrinsic.mtx,
                            pov.parameters.intrinsic.dist_coeffs,
                        )
                        x, y = max(
                            min(int(x), frame.shape[1]),
                            0,
                        ), max(
                            min(int(y), frame.shape[0]),
                            0,
                        )
                        reprojected_lms[lm_id] = (x, y)

                    # Draw connections between landmarks first
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        if (
                            start_idx in reprojected_lms
                            and end_idx in reprojected_lms
                        ):
                            start_point = reprojected_lms[start_idx]
                            end_point = reprojected_lms[end_idx]
                            cv2.line(
                                frame,
                                start_point,
                                end_point,
                                color=(255, 255, 255),
                                thickness=2,
                            )

                    # Draw landmarks (circles) on top of connections
                    for lm_id, point_3d in enumerate(points_3d):
                        # Reproject the 3D point to 2D (reuse previously computed values)
                        x, y = reprojected_lms[lm_id]

                        # Check if this camera was chosen for this point
                        if idx in chosen_cams[lm_id]:
                            color = (0, 255, 0)  # Green for chosen cameras
                        else:
                            color = (255, 0, 0)  # Blue for other cameras

                        # Draw the landmark
                        cv2.circle(frame, (x, y), radius=5, color=color, thickness=-1)

                # Resize the frame before displaying
                frame_height, frame_width = frame.shape[:2]
                new_width = int(frame_width * args.window_scale)
                new_height = int(frame_height * args.window_scale)
                resized_frame = cv2.resize(
                    frame, (new_width, new_height), interpolation=cv2.INTER_AREA
                )

                # Display the resized frame
                cv2.imshow(f"Camera_{idx}", resized_frame)

            # Visualize 3d landmarks
            # if do_render:
            #     if len(points_3d) == 21:
            #         points_3d = np.array(points_3d)
            #         bones = points_3d_to_bones_rotations(points_3d)
            #         resp = requests.post("http://localhost:3000/api/bones", json=bones)
            #         if resp.status_code != 200:
            #             print(f"Failed to send data. Status code: {resp.status_code}")
            #             print(resp.text)
            #     else:
            #         print("Not enough data to reconstruct hand in 3D.")

    # Run loops: consume asyncronusly and join with feeding
    consuming_task = asyncio.create_task(consuming_loop())
    await feeding_loop()

    # Finalize
    run = False  # notify consuming to stop
    await consuming_task  # wait for it to finish

    # Release resources
    for pov in povs:
        pov.cap_thread.join()
    await asyncio.gather(*[pov.tracker.dispose() for pov in povs])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
