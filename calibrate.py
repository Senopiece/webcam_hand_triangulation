print("Launching...")

import sys
import cv2
import numpy as np
import json
import argparse

# Set up argument parser to accept various parameters
parser = argparse.ArgumentParser(description="Camera Calibration Script")
parser.add_argument(
    "--input",
    type=str,
    default="cameras.json",
    help="Path to the cameras declarations file",
)
parser.add_argument(
    "--cams",
    type=str,
    default=None,
    help="Comma-separated list of camera indices to use, e.g., '0,1,2'",
)
parser.add_argument(
    "--output",
    type=str,
    default="calibration.json",
    help="Name of the output JSON file for calibration data",
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
    default=25.0,
    help="Size of a square in millimeters",
)

args = parser.parse_args()
cameras_path = args.input
output_file = args.output
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
    all_cameras = json.load(f)

# Filter cameras based on specified indices
if specified_indices is not None:
    cameras = [camera for camera in all_cameras if camera["index"] in specified_indices]
    if not cameras:
        print("Error: No matching cameras found for the specified indices.")
        sys.exit(1)
else:
    cameras = all_cameras

# Check if any cameras are loaded
if not cameras:
    print(
        "Error: No cameras loaded. Please check your cameras.json file and camera_indices argument."
    )
    sys.exit(1)

for camera in cameras:
    assert isinstance(camera["index"], int), "Camera index must be an integer"

# Initialize video captures
for camera in cameras:
    cap = cv2.VideoCapture(camera["index"])
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera['index']}")
        sys.exit(1)
    camera["cap"] = cap

    # Disable autofocus
    autofocus_supported = cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1
    if autofocus_supported:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Set manual focus value
    focus_value = camera.get("focus", 0)
    focus_supported = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
    if not focus_supported:
        print(
            f"Camera {camera['index']} does not support manual focus! (or an invalid focus value provided)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize calibration data
    camera["calibration_count"] = 0
    camera["image_size"] = None
    camera["imgpoints"] = []  # 2D points in image plane

print()
print("=== Camera Calibration Script ===")
print("Instructions:")
print(
    f"1. Ensure that the calibration pattern ({chessboard_size[0]}x{chessboard_size[1]} chessboard of {square_size}mm squares) is available."
)
print("2. For each camera, display the calibration pattern within its field of view.")
print("3. Press 'c' to capture calibration images for all cameras.")
print(f"   Collect at least {calibration_images_needed} images per camera.")
print("4. The script will notify you when enough images are collected for each camera.")
print(f"5. Calibration parameters will be saved to '{output_file}'.")

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

    # Store intrinsic parameters in the desired format
    camera["intrinsic"] = {
        "focal_length_pixels": {"x": fx, "y": fy},
        "skew_coefficient": s,
        "principal_point": {"x": cx, "y": cy},
        "dist_coeffs": dist_coeffs,
    }

    print(f"Intrinsic calibration completed for camera {idx}.")
    print(f"Calibration parameters saved for camera {idx}.")

# Save calibrations
calibration_data = [
    {
        "index": camera["index"],
        "intrinsic": camera["intrinsic"],
    }
    for camera in cameras
]
with open(output_file, "w") as f:
    json.dump(calibration_data, f, indent=4)
print(f"All cameras calibrated. Calibration data saved to '{output_file}'.")
print("You can now use this calibration data in your main application.")
