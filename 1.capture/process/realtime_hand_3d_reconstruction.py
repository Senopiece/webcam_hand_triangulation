import asyncio
from itertools import combinations
import cv2
import mediapipe as mp
import numpy as np
import json5
import argparse
import sys
import time

import requests

from kinematics import points_3d_to_bones_rotations
from async_hands import AsyncHandsThreadedBuildinSolution, HandTrackersPool

mp_hands = mp.solutions.hands


def load_camera_parameters(cameras_file):
    with open(cameras_file, "r") as f:
        cameras_confs = json5.load(f)

    cameras = {}
    for cam_conf in cameras_confs:
        idx = cam_conf["index"]
        if "intrinsic" not in cam_conf or "extrinsic" not in cam_conf:
            print(f"Camera {idx} does not have necessary calibration data.")
            continue

        # Intrinsic parameters
        intrinsic = cam_conf["intrinsic"]
        intrinsic_mtx = np.array(
            [
                [
                    intrinsic["focal_length_pixels"][0],
                    intrinsic["skew_coefficient"],
                    intrinsic["principal_point"][1],
                ],
                [
                    0,
                    intrinsic["focal_length_pixels"][1],
                    intrinsic["principal_point"][1],
                ],
                [0, 0, 1],
            ]
        )
        dist_coeffs = np.array(intrinsic["dist_coeffs"])

        # Extrinsic parameters
        extrinsic = cam_conf["extrinsic"]

        T = np.array([extrinsic["translation_mm"]], dtype=np.float64)
        if T.shape != (1, 3):
            raise ValueError(
                f"Invalid translation_mm shape for camera {idx}, expected 3x1."
            )
        T = T.swapaxes(1, 0)

        # R = np.array(extrinsic["rotation_matrix"], dtype=np.float64)
        # if R.shape != (3, 3):
        #     raise ValueError(
        #         f"Invalid rotation_matrix shape for camera {idx}, expected 3x3."
        #     )
        # rvec, _ = cv2.Rodrigues(R)

        rvec = np.array([extrinsic["rotation_rodrigues"]], dtype=np.float64)
        if rvec.shape != (1, 3):
            raise ValueError(
                f"Invalid rotation_rodrigues shape for camera {idx}, expected 1x3."
            )
        rvec = rvec.swapaxes(1, 0)
        R, _ = cv2.Rodrigues(rvec)

        # Make projection matrix
        RT = np.hstack((R, T))  # Rotation and translation
        P = intrinsic_mtx @ RT  # Projection matrix

        # Store parameters
        cameras[idx] = {
            "intrinsic_mtx": intrinsic_mtx,
            "dist": dist_coeffs,
            "rvec": rvec,
            "T": T,
            "P": P,
            "cap": None,
            "frame": None,
            "hand_landmarks": None,
            "tracker": None,  # To be initialized later
            "focus": cam_conf.get("focus", 0),  # Retrieve focus value
        }
    return cameras


def point_pixel_coords(cam, point_idx):
    lm = cam["hand_landmarks"][point_idx]

    # Convert normalized coordinates to pixel coordinates
    h, w, _ = cam["frame"].shape
    x = lm.x * w
    y = lm.y * h

    return x, y


def undistorted_point_pixel_coords(cam, point_idx):
    x, y = point_pixel_coords(cam, point_idx)

    # Undistort points
    undistorted = cv2.undistortPoints(
        np.array([[[x, y]]], dtype=np.float32),
        cam["intrinsic_mtx"],
        cam["dist"],
        P=cam["intrinsic_mtx"],
    )

    return undistorted[0][0]


def stereo_triangulate_point(cameras, point_idx):
    """
    Triangulate 3D point from exactly two camera views.
    Use this in prefer to triangulate_point if only two cameras are available for triangulation
    """
    assert len(cameras) == 2
    assert all(cam["hand_landmarks"] is not None for cam in cameras)

    # Take the first two cameras
    cam1, cam2 = cameras[0], cameras[1]

    # Get undistorted pixel coordinates from both cameras
    x1, y1 = undistorted_point_pixel_coords(cam1, point_idx)
    x2, y2 = undistorted_point_pixel_coords(cam2, point_idx)

    # Prepare input for cv2.triangulatePoints
    # points must be shaped as (2, N). Here N=1 since we have one point.
    pts1 = np.array([[x1], [y1]], dtype=np.float64)
    pts2 = np.array([[x2], [y2]], dtype=np.float64)

    # Triangulate
    pts4D = cv2.triangulatePoints(cam1["P"], cam2["P"], pts1, pts2)

    # Convert from homogeneous to Cartesian coordinates
    X = pts4D[:3, 0] / pts4D[3, 0]

    return X


def project(cam, point3d):
    projected_point, _ = cv2.projectPoints(
        point3d,
        cam["rvec"],
        cam["T"],
        cam["intrinsic_mtx"],
        cam["dist"],
    )
    projected_point = projected_point[0][0]
    return projected_point[0], projected_point[1]


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
    cameras = load_camera_parameters(cameras_path)
    if len(cameras) < 2:
        print("Need at least two cameras with calibration data.")
        sys.exit(1)

    def best_stereo_triangulate_point(cameras_ids, point_idx):
        """
        Triangulate 3D point from multiple camera views.
        Makes all stereo projections and chooses the best
        """
        # Iterate all pairs of cameras to find best triangulation for each point
        best_point = None
        best_score = float("+inf")
        best_cams = []

        for ids in combinations(
            filter(
                lambda idx: cameras[idx]["hand_landmarks"] is not None,
                cameras_ids,
            ),
            2,
        ):
            cams = list(map(cameras.get, ids))
            point_3d = stereo_triangulate_point(cams, point_idx)

            mean_reprojection_error = 0
            for cam in cams:
                x1, y1 = project(cam, point_3d)
                x0, y0 = point_pixel_coords(cam, point_idx)

                # Compute the error
                reprojection_error = np.linalg.norm(
                    np.array([x1, y1]) - np.array([x0, y0])
                )

                mean_reprojection_error += reprojection_error

            mean_reprojection_error /= len(cams)

            if mean_reprojection_error < best_score:
                best_point = point_3d
                best_score = mean_reprojection_error
                best_cams = ids

        return best_cams, best_point

    def triangulate_point(cameras_ids, point_idx):
        """
        Triangulate 3D point from multiple camera views.
        Alternative to best_stereo_triangulate_point using custom svg
        """
        cams = list(map(cameras.get, cameras_ids))

        projections = []
        points = []

        for cam in cams:
            if cam["hand_landmarks"] is not None:
                points.append(undistorted_point_pixel_coords(cam, point_idx))
                projections.append(cam["P"])

        if len(projections) >= 2:
            # Prepare matrices for triangulation
            A = []
            for i in range(len(projections)):
                P = projections[i]
                x, y = points[i]
                A.append(x * P[2, :] - P[0, :])
                A.append(y * P[2, :] - P[1, :])

            A = np.array(A)

            # Solve using SVD
            U, S, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X /= X[3]

            return cameras_ids, X[:3]
        else:
            return [], None

    # Initialize
    for idx, cam in cameras.items():
        # Initialize video capture
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Error: Could not open camera {idx}")
            sys.exit(1)

        # Disable autofocus
        autofocus_supported = cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1
        if autofocus_supported:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # Set manual focus value
        focus_value = cam["focus"]
        focus_supported = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
        if not focus_supported:
            print(
                f"Camera {idx} does not support manual focus! (or an invalid focus value provided)",
                file=sys.stderr,
            )
            sys.exit(1)
        cam["cap"] = cap
        cv2.namedWindow(f"Camera_{idx}", cv2.WINDOW_AUTOSIZE)

        # Initialize hands trackers for each camera
        cameras[idx]["tracker"] = HandTrackersPool(
            [AsyncHandsThreadedBuildinSolution() for _ in range(division)],
        )

    run = True

    async def consuming_loop():
        # FPS tracking variables
        fps_counter = 0
        fps = 0
        fps_display_time = time.time()

        # Loop untill said to stop but make sure to process what remains
        while (
            run
            or any(
                not cam["tracker"].idle_workers.full() for cam in cameras.values()
            )  # any channel is in processing -> new results may arrive
            or any(
                not cam["tracker"].results.empty() for cam in cameras.values()
            )  # any channel has non empty results -> need to process them
        ):
            # NOTE: it will hang freeing if channels got not equal amounts of .send calls
            results = await asyncio.gather(
                *[cam["tracker"].results.get() for cam in cameras.values()]
            )

            # Extract landmarks
            for cam, (res, frame) in zip(cameras.values(), results):
                cam["frame"] = frame
                cam["hand_landmarks"] = None

                if res.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                        res.multi_hand_landmarks, res.multi_handedness
                    ):
                        if (
                            handedness.classification[0].label == "Right"
                        ):  # actually left lol
                            cam["hand_landmarks"] = hand_landmarks.landmark
                            break

            # Triangulate points
            chosen_cameras = []
            points_3d = []
            cams_with_landmarks_ids = [
                idx for idx, cam in cameras.items() if cam["hand_landmarks"] is not None
            ]
            if len(cams_with_landmarks_ids) >= 2:
                for point_idx in range(num_landmarks):
                    chosen, point_3d = triangulate_point(
                        cams_with_landmarks_ids, point_idx
                    )
                    assert point_3d is not None

                    chosen_cameras.append(chosen)
                    points_3d.append(point_3d)

            # Draw landmarks
            for idx, cam in cameras.items():
                frame = cam["frame"]

                # Draw landmarks if can
                if len(points_3d) == 21:
                    # Store reprojected 2D landmarks for drawing connections
                    reprojected_points = {}

                    for point_idx, point_3d in enumerate(points_3d):
                        x, y = project(cam, point_3d)
                        x, y = max(min(int(x), frame.shape[1]), 0), max(
                            min(int(y), frame.shape[0]), 0
                        )
                        reprojected_points[point_idx] = (x, y)

                    # Draw connections between landmarks first
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        if (
                            start_idx in reprojected_points
                            and end_idx in reprojected_points
                        ):
                            start_point = reprojected_points[start_idx]
                            end_point = reprojected_points[end_idx]
                            cv2.line(
                                frame,
                                start_point,
                                end_point,
                                color=(255, 255, 255),
                                thickness=2,
                            )

                    # Draw landmarks (circles) on top of connections
                    for point_idx, point_3d in enumerate(points_3d):
                        # Reproject the 3D point to 2D (reuse previously computed values)
                        x, y = reprojected_points[point_idx]

                        # Check if this camera was chosen for this point
                        if idx in chosen_cameras[point_idx]:
                            color = (0, 255, 0)  # Green for chosen cameras
                        else:
                            color = (255, 0, 0)  # Blue for other cameras

                        # Draw the landmark
                        cv2.circle(frame, (x, y), radius=5, color=color, thickness=-1)

            # FPS counter update every second
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_display_time >= 1.0:
                fps_display_time = current_time
                fps = fps_counter
                fps_counter = 0

            # Display frames
            for idx, cam in cameras.items():
                frame = cam["frame"]

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
            if do_render:
                if len(points_3d) == 21:
                    points_3d = np.array(points_3d)
                    bones = points_3d_to_bones_rotations(points_3d)
                    resp = requests.post("http://localhost:3000/api/bones", json=bones)
                    if resp.status_code != 200:
                        print(f"Failed to send data. Status code: {resp.status_code}")
                        print(resp.text)
                else:
                    print("Not enough data to reconstruct hand in 3D.")

    async def feeding_loop():
        while True:
            tasks = [None for _ in range(len(cameras))]
            for i, (idx, cam) in enumerate(cameras.items()):
                cap = cam["cap"]
                ret, frame = cap.read()

                if not ret:
                    print(f"Error: Could not read from camera {idx}")
                    continue

                tasks[i] = cam["tracker"].send(frame)

            await asyncio.gather(*tasks)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

    # Run loops: consume asyncronusly and join with feeding
    consuming_task = asyncio.create_task(consuming_loop())
    await feeding_loop()

    # Finalize
    run = False  # notify consuming to stop
    await consuming_task  # wait for it to finish

    # Release resources
    for cam in cameras.values():
        cam["cap"].release()
    await asyncio.gather(*[cam["tracker"].dispose() for cam in cameras.values()])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
