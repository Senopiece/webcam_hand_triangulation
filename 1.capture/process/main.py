import asyncio
from typing import List, NamedTuple, Set, Tuple
import cv2
import mediapipe as mp
import argparse
import sys
import time

from async_hands import AsyncHandsThreadedBuildinSolution, HandTrackersPool
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
        "--file",
        type=str,
        default="setup.json5",
        help="Path to the cameras declarations file",
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
        default=4,
        help="Number of the hand tracking workers pool per camera",
    )
    args = parser.parse_args()
    cameras_path = args.file
    do_render = args.render
    division = args.division

    # Load camera parameters
    cameras_params = load_cameras_parameters(cameras_path)
    if len(cameras_params) < 2:
        print("Need at least two cameras with calibration data.")
        sys.exit(1)

    # Initialize
    povs: Set[PoV] = set()
    for idx, cam_param in cameras_params.items():
        # Initialize video capture
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Error: Could not open camera {idx}", out=sys.stderr)
            sys.exit(1)

        # Set 60 fps
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)

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
        
        # Make window for the pov
        cv2.namedWindow(f"Camera_{idx}", cv2.WINDOW_AUTOSIZE)

        # Initialize hands trackers for each camera
        povs.add(PoV(
            idx=idx,
            cap=cap,
            tracker=HandTrackersPool(
                [AsyncHandsThreadedBuildinSolution() for _ in range(division)],
            ),
            parameters=cam_param,
        ),)
    
    async def feeding_loop():
        # NOTE: Using only .tracker (the send side) and .cap from PoV
        while True:
            tasks = []
            for pov in povs:
                ret, frame = pov.cap.read()

                if not ret:
                    print(f"Error: Could not read from camera {idx}", out=sys.stderr)
                    sys.exit(1)

                tasks.append(pov.tracker.send(frame))

            await asyncio.gather(*tasks)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

    run = True

    async def consuming_loop():
        # NOTE: Using .tracker (the recieve side), and all the other except for .cap from PoV

        # FPS tracking variables
        fps_counter = 0
        fps = 0
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
            results: List[Tuple[PoV, NamedTuple, cv2.typing.MatLike]] = await asyncio.gather(
                *[(pov, *pov.tracker.results.get()) for pov in povs]
            )

            # Extract landmarks and processed frames
            for pov, res, frame in results:
                pov.frame = frame
                pov.hand_landmarks = None

                if res.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                        res.multi_hand_landmarks, res.multi_handedness
                    ):
                        if (
                            handedness.classification[0].label == "Right"
                        ):  # actually left lol
                            pov.hand_landmarks = hand_landmarks.landmark
                            break

            # Triangulate points
            chosen_cams = []
            points_3d = []
            povs_with_landmarks = {
                pov for pov in povs if pov.hand_landmarks is not None
            }
            if len(povs_with_landmarks) >= 2:
                for lm_id in range(num_landmarks):
                    # Prepare landmarks contexts
                    lmcs = set()
                    for pov in povs_with_landmarks:
                        point = landmark_to_pixel_coord(pov.frame.shape, pov.hand_landmarks[lm_id])
                        undistorted_lm = undistort_pixel_coord(point, pov.parameters.intrinsic.mtx, pov.parameters.intrinsic.dist_coeffs)
                        lmcs.add(ContextedLandmark(cam_idx=pov.cam_idx, P=pov.parameters.P, lm=undistorted_lm))

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
                fps_display_time = current_time
                fps = fps_counter
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

                # Add FPS to the frame
                cv2.putText(
                    resized_frame,
                    f"FPS: {fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
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
        pov.cap.release()
    await asyncio.gather(*[pov.tracker.dispose() for pov in povs])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
