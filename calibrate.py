# a all in one solution (extrinsics and intrinsics) for our specific case to make less work for calibration

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
    "--n",
    type=int,
    default=12,
    help="Number of calibration images required per camera",
)
parser.add_argument(
    "--chessboard_size",
    type=str,
    default="5x7",
    help="Chessboard size as columns x rows (inner corners), e.g., '9x6'",
)
parser.add_argument(
    "--square_size",
    type=float,
    default=13.7,
    help="Size of a square in millimeters",
)
parser.add_argument(
    "-f",
    "--force",
    help="Force overwrite calibrations",
    action="store_true",
)
parser.add_argument(
    "--window_scale",
    type=float,
    default=0.7,
    help="Scale of a window",
)
parser.add_argument(
    "--use_existing_intrinsics",
    help="Use existing intrinsics, dont overwrite them",
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

# Load camera configurations from the JSON file
with open(cameras_path, "r") as f:
    cameras_confs = json5.load(f)

# Notify if calibration already exists
if not args.force and any(
    "extrinsic" in cam or (("intrinsic" in cam) and not args.use_existing_intrinsics)
    for cam in cameras_confs
):
    print("Some calibration already exists. Use --force to overwrite.", file=sys.stderr)
    sys.exit(1)

# Initialize video captures
print("\nLaunching...")
cameras = []
for camera_conf in cameras_confs:
    cap = cv2.VideoCapture(camera_conf["index"])
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_conf['index']}", file=sys.stderr)
        sys.exit(1)

    autofocus_supported = cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1
    if autofocus_supported:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    focus_value = camera_conf.get("focus", 0)
    focus_supported = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
    if not focus_supported:
        print(
            f"Camera {camera_conf['index']} does not support manual focus!",
            file=sys.stderr,
        )
        sys.exit(1)

    cameras.append(
        {
            "index": camera_conf["index"],
            "image_size": None,
            "imgpoints": [],
            "cap": cap,
        }
    )

print()
print("=== Camera Calibration Script ===")
print("Instructions:")
print(
    f"1. Ensure that the calibration pattern ({chessboard_size[0]}x{chessboard_size[1]} chessboard of {square_size}mm squares) is available."
)
print(
    "2. Display the calibration pattern within the field of view of all cameras simultaneously."
)
print("3. Press 'c' to capture calibration images for all cameras.")
print(f"   Collect at least {calibration_images_needed} images.")
print("4. A camera bacomes center of the world.")
print("5. Calibration will be performed, and results saved.")

for camera in cameras:
    cv2.namedWindow(f"Camera_{camera['index']}", cv2.WINDOW_AUTOSIZE)

# Capture frames and display recognized pattern points
calibration_count = 0
while calibration_count < calibration_images_needed:
    frames = []
    cameras_imgpoints = {}  # Dictionary to store valid corners for each camera

    for camera in cameras:
        cap = camera["cap"]
        idx = camera["index"]

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read from camera {idx}")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners, meta = cv2.findChessboardCornersSBWithMeta(
            gray,
            chessboard_size,
            flags=cv2.CALIB_CB_MARKER,
        )

        if ret:
            if meta.shape[0] != chessboard_rows:
                corners = corners.reshape(-1, 2)
                corners = corners.reshape(*chessboard_size, 2)
                corners = corners.transpose(1, 0, 2)
                corners = corners.reshape(-1, 2)
                corners = corners[:, np.newaxis, :]
            cameras_imgpoints[idx] = corners
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

        if camera["image_size"] is None:
            camera["image_size"] = gray.shape[::-1]

        #  Resize the frame before displaying
        frame_height, frame_width = frame.shape[:2]
        new_width = int(frame_width * args.window_scale)
        new_height = int(frame_height * args.window_scale)
        resized_frame = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        # Display the resized frame
        cv2.imshow(f"Camera_{idx}", resized_frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        print("Exiting calibration script.")
        sys.exit(0)

    elif key & 0xFF == ord("c"):
        # Check if the pattern was visible in all cameras simultaneously
        if len(cameras_imgpoints) == len(cameras):
            calibration_count += 1
            for idx, corners2 in cameras_imgpoints.items():
                camera = next(cam for cam in cameras if cam["index"] == idx)
                camera["imgpoints"].append(corners2)
            print(
                f"Calibration images captured for all cameras. Remaining {calibration_images_needed - calibration_count}."
            )
        else:
            print("Pattern is not visible to all cameras")

# Release resources after loop
for camera in cameras:
    camera["cap"].release()
cv2.destroyAllWindows()

# Prepare object points (same for all images)
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Perform intrinsic calibrations
for camera in cameras:
    idx = camera["index"]
    print(f"Processing intrinsic calibration for camera {idx}...")
    cam_conf = next(conf for conf in cameras_confs if conf["index"] == idx)

    if args.use_existing_intrinsics and "intrinsic" in cam_conf:
        print(f"Using existing intrinsic parameters for camera {idx}.")
        # Reconstruct camera matrix and distortion coefficients
        intrinsic_conf = cam_conf["intrinsic"]
        fx = intrinsic_conf["focal_length_pixels"]["x"]
        fy = intrinsic_conf["focal_length_pixels"]["y"]
        s = intrinsic_conf["skew_coefficient"]
        cx = intrinsic_conf["principal_point"]["x"]
        cy = intrinsic_conf["principal_point"]["y"]
        mtx = np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]])
        dist_coeffs = np.array(intrinsic_conf["dist_coeffs"])
    else:
        print(f"Performing intrinsic calibration for camera {idx}...")
        ret, mtx, dist_coeffs, _, _ = cv2.calibrateCamera(
            [objp for _ in camera["imgpoints"]],
            camera["imgpoints"],
            camera["image_size"],
            None,
            None,
        )
        dist_coeffs = dist_coeffs.flatten()
        # Extract intrinsic parameters
        fx, fy = mtx[0, 0], mtx[1, 1]
        s, cx, cy = mtx[0, 1], mtx[0, 2], mtx[1, 2]
        # Store intrinsic parameters
        cam_conf["intrinsic"] = {
            "focal_length_pixels": {"x": fx, "y": fy},
            "skew_coefficient": s,
            "principal_point": {"x": cx, "y": cy},
            "dist_coeffs": dist_coeffs.tolist(),
        }

    # Store intrinsic parameters in the camera dictionary for later use
    camera["mtx"] = mtx
    camera["dist_coeffs"] = dist_coeffs

# Perform stereo calibration between each pair of cameras
from itertools import combinations

print("\n=== Performing Stereo Calibration ===")
camera_pairs = list(combinations(cameras, 2))

for cam1, cam2 in camera_pairs:
    idx1 = cam1["index"]
    idx2 = cam2["index"]
    print(f"\nStereo calibration between Camera {idx1} and Camera {idx2}...")

    # Ensure that the number of image points is the same for both cameras
    num_images = len(cam1["imgpoints"])
    if num_images != len(cam2["imgpoints"]):
        print(
            f"Error: Number of calibration images for cameras {idx1} and {idx2} do not match."
        )
        continue

    # Prepare object points and image points for stereo calibration
    objpoints = [objp for _ in range(num_images)]
    imgpoints1 = cam1["imgpoints"]
    imgpoints2 = cam2["imgpoints"]

    # Retrieve intrinsic parameters
    mtx1 = cam1["mtx"]
    dist1 = cam1["dist_coeffs"]
    mtx2 = cam2["mtx"]
    dist2 = cam2["dist_coeffs"]

    # Stereo calibration flags
    stereo_flags = (
        cv2.CALIB_FIX_INTRINSIC
    )  # Assume intrinsic parameters are known and fixed

    # Perform stereo calibration
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1,
        imgpoints2,
        mtx1,
        dist1,
        mtx2,
        dist2,
        cam1["image_size"],
        criteria=(
            cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
            100,
            1e-5,
        ),
        flags=stereo_flags,
    )

    print(f"Stereo calibration between cameras {idx1} and {idx2} completed.")
    print(f"Rotation matrix:\n{R}")
    print(f"Translation vector:\n{T}")

    # Store extrinsic parameters in the camera configurations
    # Since we have multiple cameras, we'll define the world coordinate system
    # with respect to the first camera (you can choose any reference)

    # Compute yaw, pitch, roll from rotation matrix
    from math import atan2, asin

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x_angle = atan2(R[2, 1], R[2, 2])
        y_angle = atan2(-R[2, 0], sy)
        z_angle = atan2(R[1, 0], R[0, 0])
    else:
        x_angle = atan2(-R[1, 2], R[1, 1])
        y_angle = atan2(-R[2, 0], sy)
        z_angle = 0

    # Convert to degrees
    yaw_deg = np.degrees(z_angle)
    pitch_deg = np.degrees(y_angle)
    roll_deg = np.degrees(x_angle)

    # Convert to radians
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    roll_rad = np.radians(roll_deg)

    # Convert translation vector to centimeters
    T_cm = T.flatten() * 0.1  # Assuming square_size is in millimeters

    # Update the configuration for cam2 relative to cam1
    cam2_conf = next(conf for conf in cameras_confs if conf["index"] == idx2)
    cam2_conf["extrinsic"] = {
        "translation_centimeters": {
            "x": float(T_cm[0]),
            "y": float(T_cm[1]),
            "z": float(T_cm[2]),
        },
        "rotation_radians": {
            "yaw": float(yaw_rad),
            "pitch": float(pitch_rad),
            "roll": float(roll_rad),
        },
    }

    # Optionally, you can also update cam1's extrinsic parameters (assuming it's the reference)
    cam1_conf = next(conf for conf in cameras_confs if conf["index"] == idx1)
    if "extrinsic" not in cam1_conf:
        # Set cam1 as the origin
        cam1_conf["extrinsic"] = {
            "translation_centimeters": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation_radians": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        }

print("\nStereo calibration completed for all camera pairs.")

# Save calibrations
with open(cameras_path, "w") as f:
    json5.dump(cameras_confs, f, indent=4)
print("Cameras file updated.")
