# calibrates just intrinsics for a set of cameras

import sys
import cv2
import numpy as np
import json5
import argparse

# Set up argument parser to accept various parameters
parser = argparse.ArgumentParser(description="Camera Calibration Script")
parser.add_argument(
    "--file",
    type=str,
    default="setup.json5",
    help="Path to the state declarations file",
)
parser.add_argument(
    "--cams",
    type=str,
    default=None,
    help="Comma-separated list of camera indices to use, e.g., '0,1,2'",
)
parser.add_argument(
    "--n",
    type=int,
    default=10,
    help="Number of calibration images required per camera",
)
parser.add_argument(
    "--chessboard_size",
    type=str,
    default="9x6",
    help="Chessboard size as columns x rows (inner corners), e.g., '9x6'",
)
parser.add_argument(
    "--square_size",
    type=float,
    default=8,
    help="Size of a square in millimeters",
)
parser.add_argument(
    "-f",
    "--force",
    help="Force overwrite calibrations",
    action="store_true",
)

args = parser.parse_args()
cameras_path = args.file
calibration_images_needed = args.n

# Parse chessboard size argument
try:
    chessboard_cols, chessboard_rows = map(int, args.chessboard_size.lower().split("x"))
    chessboard_size = (chessboard_cols, chessboard_rows)
except ValueError:
    print("Error: Invalid chessboard_size format. Use 'colsxrows', e.g., '9x6'.")
    sys.exit(1)

square_size = args.square_size

# Parse camera_indices argument
if args.cams is not None:
    try:
        specified_indices = set(map(int, args.cams.split(",")))
    except ValueError:
        print("Error: Invalid format for cams. Use a comma-separated list of integers.")
        sys.exit(1)
else:
    specified_indices = None

# Load camera configurations from the JSON file
with open(cameras_path, "r") as f:
    all_cameras_confs = json5.load(f)


# Filter cameras based on specified indices
if specified_indices is not None:
    actual_indices = set(camera_conf["index"] for camera_conf in all_cameras_confs)
    unknown_cameras = specified_indices - actual_indices

    if len(unknown_cameras) != 0:
        print(f"Error: Cameras {unknown_cameras} are not defined in the file")
        sys.exit(1)

    cameras_confs = [
        camera_conf
        for camera_conf in all_cameras_confs
        if camera_conf["index"] in specified_indices
    ]
else:
    cameras_confs = all_cameras_confs

# Notify if calibration already exist
if not args.force:
    new_cameras_confs = []

    for camera_conf in cameras_confs:
        idx = camera_conf["index"]

        if "intrinsic" in camera_conf:
            owerwrite = input(
                f"Intrincic calibration of camera {idx} already exists. Owerwrite? (y/n): "
            )
            if owerwrite.strip().lower().startswith("y"):
                print("Ok, will overwrite")
            else:
                print("Ok, skip this camera from calibration")
                continue

        new_cameras_confs.append(camera_conf)

    cameras_confs = new_cameras_confs

# Check if any cameras are loaded
if not cameras_confs:
    print(
        "Error: No cameras loaded. Please check your cameras.json file and camera_indices argument."
    )
    sys.exit(1)

# Initialize video captures
print("\nLaunching...")
cameras = []
for camera_conf in cameras_confs:
    cap = cv2.VideoCapture(camera_conf["index"])
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_conf['index']}")
        sys.exit(1)

    # Disable autofocus
    autofocus_supported = cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1
    if autofocus_supported:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Set manual focus value
    focus_value = camera_conf.get("focus", 0)
    focus_supported = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
    if not focus_supported:
        print(
            f"Camera {camera_conf['index']} does not support manual focus! (or an invalid focus value provided)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Remember the camera
    cameras.append(
        {
            "index": camera_conf["index"],
            "calibration_count": 0,
            "image_size": None,
            "imgpoints": [],  # 2D points in image plane
            "cap": cap,
        }
    )

del cameras_confs

print()
print("=== Camera Calibration Script ===")
print("Instructions:")
print(
    f"1. Ensure that the calibration pattern ({chessboard_size[0]}x{chessboard_size[1]} chessboard of {square_size}mm squares) is available."
)
print("2. For each camera, display the calibration pattern within its field of view.")
print("3. Press 'c' to capture calibration images for all cameras.")
print(f"   Collect at least {calibration_images_needed} images per camera.")
print(
    "4. After enough images per camera was collected, script will perform calibration and write the results back to the cameras file."
)

# Set up windows for each camera feed
for camera in cameras:
    cv2.namedWindow(f"Camera_{camera['index']}", cv2.WINDOW_NORMAL)

# Capture frames until any camera has not enough images for calibration
while any(
    camera["calibration_count"] < calibration_images_needed for camera in cameras
):
    for camera in cameras:
        cap = camera["cap"]
        idx = camera["index"]

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read from camera {idx}")
            continue

        camera["frame"] = frame
        cv2.imshow(f"Camera_{idx}", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        print("Exiting calibration script.")
        break

    elif key & 0xFF == ord("c"):
        print("Capturing calibration images...")
        for camera in cameras:
            idx = camera["index"]
            frame = camera["frame"]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                # Refine corner locations
                corners2 = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    ),
                )
                camera["imgpoints"].append(corners2)
                camera["calibration_count"] += 1

                # Update image size and ensiure it's consistent
                assert (
                    camera["image_size"] is None
                    or camera["image_size"] == gray.shape[::-1]
                )
                camera["image_size"] = gray.shape[::-1]

                # Draw and display the corners
                print(
                    f"Calibration image {camera['calibration_count']} collected for camera {idx}."
                )
            else:
                print(f"Calibration pattern not found in camera {idx}.")

# Release windows
for camera in cameras:
    camera["cap"].release()
cv2.destroyAllWindows()

# Prepare object points based on the real-world dimensions of the calibration pattern
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Perform calibrations
for camera in cameras:
    idx = camera["index"]

    print(f"Performing intrinsic calibration for camera {idx}...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp for _ in camera["imgpoints"]],
        camera["imgpoints"],
        camera["image_size"],
        None,
        None,
    )
    # Extract intrinsic parameters
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    s = mtx[0, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    dist_coeffs = dist.flatten().tolist()

    # Calculate Reprojection Error (e.g the inverse of how confident is the script that the determined parameters are correct)
    total_r_error = 0
    for i in range(len(camera["imgpoints"])):
        imgpoints2, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(camera["imgpoints"][i], imgpoints2, cv2.NORM_L2) / len(
            imgpoints2
        )
        total_r_error += error
    mean_r_error = total_r_error / len(camera["imgpoints"])

    # Modify the camera confs
    camera_conf = [conf for conf in all_cameras_confs if conf["index"] == idx][0]

    # Store intrinsic parameters in the desired format
    camera_conf["intrinsic"] = {
        "focal_length_pixels": {"x": fx, "y": fy},
        "skew_coefficient": s,
        "principal_point": {"x": cx, "y": cy},
        "dist_coeffs": dist_coeffs,
    }

    print(f"Intrinsic calibration completed for camera {idx}.")
    print(f"Mean reprojection error for camera {idx}: {mean_r_error:.4f} pixels")

# Save calibrations
with open(cameras_path, "w") as f:
    json5.load(all_cameras_confs, f, indent=4)
print("Cameras file updated.")
