# Merged Camera Calibration Script (Intrinsic and Extrinsic) - With Direct Pairwise Overlap Assertions

# Approach is fixed to specific camera positioning such that all cameras can see the pattern at the same time
# Also it ignores frames where a camera does not detect the pattern for simplicity

# TODO: Calibrate using global optimization (for now stereoCalibration is used that utilizes only information of each camera with the pivot camera, but with using global optimization we can utilize the data between other pairs to improve accuracy and consistency among the cameras between each other)

import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

import asyncio
import sys
import time
from typing import List, Tuple
import numpy as np
import json5
import argparse

from .models import PoV
from .async_cb import AsyncCBThreadedSolution, CBProcessingPool

# TODO: to get rid of separate .def and .calib and write the calibration back to the original file, manage somehow to keep the original formatting and comments
# TODO: rewrite to threads to be in style sync with the capture/process


async def main():
    # Set up argument parser to accept various parameters
    parser = argparse.ArgumentParser(description="Camera Calibration Script")
    parser.add_argument(
        "--ifile",
        type=str,
        default="cameras.def.json5",
        help="Path to the cameras declaration file",
    )
    parser.add_argument(
        "--ofile",
        type=str,
        default="cameras.calib.json5",
        help="Path to the calibration file to be written",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=12,
        help="Number of calibration images required per camera",
    )
    parser.add_argument(
        "--chessboard_size",
        type=str,
        default="9x13",
        help="Chessboard size as columns x rows (inner corners), e.g., '9x6'",
    )
    parser.add_argument(
        "--square_size",
        type=float,
        default=13,
        help="Size of a square in millimeters",
    )
    parser.add_argument(
        "--window_size",
        type=str,
        default="448x336",
        help="Size of a preview window",
    )
    parser.add_argument(
        "--pivot",
        type=int,
        default=None,
        help="Index of the pivot (reference) camera",
    )
    parser.add_argument(
        "--division",
        type=int,
        default=4,
        help="Number of workers to process frames (per camera)",
    )

    args = parser.parse_args()
    input_file_path = args.ifile
    output_file_path = args.ofile
    calibration_images_needed = args.n
    desired_window_size = tuple(map(int, args.window_size.split("x")))

    # Parse chessboard size argument
    try:
        chessboard_cols, chessboard_rows = map(
            int, args.chessboard_size.lower().split("x")
        )
        chessboard_size = (chessboard_cols, chessboard_rows)
    except ValueError:
        print("Error: Invalid chessboard_size format. Use 'colsxrows', e.g., '9x6'.")
        sys.exit(1)

    square_size = args.square_size

    # Load camera configurations from the JSON file
    with open(input_file_path, "r") as f:
        cameras_confs = json5.load(f)

    if len(set(conf["index"] for conf in cameras_confs)) < 2:
        print("Need at least two cameras.")
        sys.exit(1)

    # Initialize video captures
    print("\nInitalizing cameras...")
    povs: List[PoV] = []
    for camera_conf in cameras_confs:
        idx = camera_conf["index"]
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Error: Could not open camera {idx}", file=sys.stderr)
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))  # type: ignore

        # Set 60 fps
        size = list(map(int, camera_conf["size"].split("x")))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        cap.set(cv2.CAP_PROP_FPS, camera_conf["fps"])

        # Disable autofocus
        autofocus_supported = cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1
        if autofocus_supported:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # Set manual focus value
        focus_value = camera_conf["focus"]
        focus_supported = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
        if not focus_supported:
            print(
                f"Camera {idx} does not support manual focus! (or invalid focus value provided)",
                file=sys.stderr,
            )
            sys.exit(1)

        # Make window for the pov
        cv2.namedWindow(f"Camera_{idx}", cv2.WINDOW_AUTOSIZE)

        # Initialize camera data
        povs.append(
            PoV(
                cam_id=idx,
                cap=cap,
                processor=CBProcessingPool(
                    [
                        AsyncCBThreadedSolution(chessboard_size)
                        for _ in range(args.division)
                    ],
                ),
            )
        )

        print(f"Camera {idx} setup complete")

    # Collect camera indices
    camera_indices = [camera_conf["index"] for camera_conf in cameras_confs]

    # Validate pivot camera index
    if args.pivot is not None:
        if args.pivot not in camera_indices:
            print(
                f"Error: Specified pivot camera index {args.pivot} is not in the list of available cameras."
            )
            sys.exit(1)
        reference_idx = args.pivot
    else:
        # Default to the first camera index
        reference_idx = camera_indices[0]

    print(f"\nUsing camera {reference_idx} as the pivot (reference) camera.")

    print()
    print("=== Camera Calibration Script ===")
    print("Instructions:")
    print(
        f"1. Ensure that the calibration pattern ({chessboard_size[0]}x{chessboard_size[1]} chessboard of {square_size}mm squares) is visible in all cameras you want to calibrate."
    )
    print("2. Press 'c' to capture calibration images.")
    print("   The script will print which cameras detected the pattern.")
    print("3. Press 's' to perform calibration when ready.")
    print(
        f"   Calibration requires at least {calibration_images_needed} captures when all cameras detect the pattern."
        f"   Captures when a camera does not detect the pattern will be skipped."
    )
    print(
        "4. After calibration, the script will write the intrinsic and extrinsic parameters to the calibration file."
    )
    print()

    shots_count = 0

    async def feeding_loop():
        nonlocal shots_count
        while True:
            tasks = []
            for pov in povs:
                idx = pov.cam_id
                ret, frame = pov.cap.read()

                if not ret:
                    print(f"Error: Could not read from camera {idx}", file=sys.stderr)
                    sys.exit(1)

                tasks.append(pov.processor.send(cv2.flip(frame, 1)))

            await asyncio.gather(*tasks)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

            elif key & 0xFF == ord("c"):
                # Verify shot
                missing_cameras = [pov.cam_id for pov in povs if pov.corners is None]
                if missing_cameras:
                    print("Not all cameras have detected the pattern.")
                    print(f"+- Cameras missing pattern: {sorted(missing_cameras)}")
                    continue

                # Collect detected corners
                for pov in povs:
                    pov.shots.append(pov.corners)
                shots_count += 1

                # Print how many shots remains
                print(f"Captured {shots_count}/{calibration_images_needed}.")

            elif key & 0xFF == ord("s"):
                # Check if can proceed
                if shots_count < calibration_images_needed:
                    remaining_shots = calibration_images_needed - shots_count
                    print(
                        f"Not enough shots collected. Please capture {remaining_shots} more shots."
                    )
                    continue

                print("\nProceeding to calibration...")
                break

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
                not pov.processor.idle_workers.full() for pov in povs
            )  # any channel is in processing -> new results may arrive
            or any(
                not pov.processor.results.empty() for pov in povs
            )  # any channel has non empty results -> need to process them
        ):
            # NOTE: it will hang freeing if channels got not equal amounts of .send calls
            results: List[Tuple[np.ndarray, cv2.typing.MatLike]] = await asyncio.gather(
                *[pov.processor.results.get() for pov in povs]
            )

            # Display and detect corners
            for pov, (res, frame) in zip(povs, results):
                idx = pov.cam_id
                pov.frame = frame
                pov.corners = res

                # Resize the frame before displaying
                resized_frame = cv2.resize(
                    frame, desired_window_size, interpolation=cv2.INTER_AREA
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

            # FPS counter update every second
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_display_time >= 1.0:
                fps_display_time = current_time
                fps = fps_counter
                fps_counter = 0

    # Run loops: consume asyncronusly and join with feeding
    consuming_task = asyncio.create_task(consuming_loop())
    await feeding_loop()

    # Finalize
    run = False  # notify consuming to stop
    await consuming_task  # wait for it to finish

    # Release resources after loop
    for pov in povs:
        pov.cap.release()
    cv2.destroyAllWindows()

    # Skip if not enough data
    if shots_count < calibration_images_needed:
        return

    # Prepare object points based on the real-world dimensions of the calibration pattern
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(
        -1, 2
    )
    objp *= square_size

    # Perform intrinsic calibrations
    for pov in povs:
        idx = pov.cam_id

        cam_calib = next(calib for calib in cameras_confs if calib["index"] == idx)

        print(f"Performing intrinsic calibration for camera {idx}...")
        assert pov.frame is not None
        ret, mtx, dist_coeffs, _, _ = cv2.calibrateCamera(
            [objp for _ in range(shots_count)],
            pov.shots,
            pov.frame.shape[1::-1],
            np.zeros((3, 3)),
            np.zeros((5, 1)),
        )
        dist_coeffs = dist_coeffs.flatten()

        # Extract intrinsic parameters
        fx, fy = mtx[0, 0], mtx[1, 1]
        s, cx, cy = mtx[0, 1], mtx[0, 2], mtx[1, 2]

        # Store intrinsic parameters
        cam_calib["intrinsic"] = {
            "focal_length_pixels": [fx, fy],
            "skew_coefficient": s,
            "principal_point": [cx, cy],
            "dist_coeffs": dist_coeffs.tolist(),
        }

        # Store intrinsic parameters for later use
        pov.mtx = mtx
        pov.dist_coeffs = dist_coeffs

    print("\nComputing transformations relative to the pivot camera...")

    # Pivot camera extrinsic is identity
    pivot_cam_calib = next(
        calib for calib in cameras_confs if calib["index"] == reference_idx
    )
    pivot_cam_calib["extrinsic"] = {
        "translation_mm": [0, 0, 0],
        "rotation_rodrigues": [0, 0, 0],
        # "rotation_matrix": [
        #     [1, 0, 0],
        #     [0, 1, 0],
        #     [0, 0, 1],
        # ],
    }

    # Prepare shared data before transoformation compute
    objpoints = [objp for _ in range(shots_count)]

    ref_pov = next(pov for pov in povs if pov.cam_id == reference_idx)

    imgpoints1 = ref_pov.shots

    mtx1 = ref_pov.mtx
    dist1 = ref_pov.dist_coeffs

    assert ref_pov.frame is not None
    image_size = ref_pov.frame.shape[1::-1]

    # Compute transformations for each camera relative to the pivot
    for pov in povs:
        idx = pov.cam_id

        if idx == reference_idx:
            continue

        imgpoints2 = pov.shots

        mtx2 = pov.mtx
        dist2 = pov.dist_coeffs

        # Stereo calibration
        assert not (mtx1 is None or dist1 is None or mtx2 is None or dist2 is None)

        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints1,
            imgpoints2,
            mtx1,
            dist1,
            mtx2,
            dist2,
            image_size,
            criteria=(
                cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                100,
                1e-5,
            ),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )

        cam_calib = next(calib for calib in cameras_confs if calib["index"] == idx)
        cam_calib["extrinsic"] = {
            "translation_mm": T.flatten().tolist(),
            "rotation_rodrigues": cv2.Rodrigues(R)[0].flatten().tolist(),
            # "rotation_matrix": R.tolist(),
        }

        print(f"Computed transformation from camera {reference_idx} to camera {idx}.")

    # Save calibrations
    with open(output_file_path, "w") as f:
        json5.dump(cameras_confs, f, indent=4)
    print("\nCalibration file is written with intrinsic and extrinsic parameters.")


if __name__ == "__main__":
    asyncio.run(main())
