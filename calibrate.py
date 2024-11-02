import sys
import cv2
import numpy as np
import json
import argparse

# Set up argument parser to accept various parameters
parser = argparse.ArgumentParser(description="Camera Calibration Script")
parser.add_argument(
    "--file",
    type=str,
    default="setup.json",
    help="Path to the state declarations file",
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

# Load camera configurations from the JSON file
with open(cameras_path, "r") as f:
    cameras_confs = json.load(f)

# Notify if calibration already exists
if not args.force and any(
    "extrinsic" in cam or "intrinsic" in cam for cam in cameras_confs
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
print("4. Calibration will be performed, and results saved.")

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
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If corners are found, refine and store them, and draw them on the frame
        if ret:
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
            cameras_imgpoints[idx] = corners2  # Store the refined corners for later use
            cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)

        # Set image size once for each camera
        if camera["image_size"] is None:
            camera["image_size"] = gray.shape[::-1]

        # Display the frame with drawn corners if detected
        cv2.imshow(f"Camera_{idx}", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        print("Exiting calibration script.")
        break

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


objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Perform calibrations
for camera in cameras:
    idx = camera["index"]
    print(f"Performing calibration for camera {idx}...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp for _ in camera["imgpoints"]],
        camera["imgpoints"],
        camera["image_size"],
        None,
        None,
    )

    # Extract intrinsic parameters
    fx, fy = mtx[0, 0], mtx[1, 1]
    s, cx, cy = mtx[0, 1], mtx[0, 2], mtx[1, 2]
    dist_coeffs = dist.flatten().tolist()

    # Extrinsic parameters based on the first image
    rvec, tvec = rvecs[0], tvecs[0]
    R, _ = cv2.Rodrigues(rvec)
    T_cm = tvec * 0.1  # Convert to centimeters
    yaw, pitch, roll = cv2.decomposeProjectionMatrix(np.hstack((R, tvec)))[6]

    # Calculate mean reprojection error
    total_r_error = 0
    for i in range(len(camera["imgpoints"])):
        imgpoints2, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(camera["imgpoints"][i], imgpoints2, cv2.NORM_L2) / len(
            imgpoints2
        )
        total_r_error += error
    mean_r_error = total_r_error / len(camera["imgpoints"])

    # Store calibration results in camera configuration
    cam_conf = next(conf for conf in cameras_confs if conf["index"] == idx)
    cam_conf["intrinsic"] = {
        "focal_length_pixels": {"x": fx, "y": fy},
        "skew_coefficient": s,
        "principal_point": {"x": cx, "y": cy},
        "dist_coeffs": dist_coeffs,
    }
    cam_conf["extrinsic"] = {
        "translation_centimeters": {
            "x": float(T_cm[0]),
            "y": float(T_cm[1]),
            "z": float(T_cm[2]),
        },
        "rotation_degrees": {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
        },
    }
    cam_conf["reprojection_error"] = mean_r_error
    print(
        f"Calibration for camera {idx} complete with mean reprojection error: {mean_r_error}."
    )

# Save calibrations
with open(cameras_path, "w") as f:
    json.dump(cameras_confs, f, indent=4)
print("Cameras file updated.")
