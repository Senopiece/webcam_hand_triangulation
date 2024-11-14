import sys
import cv2
import numpy as np
import json5
import argparse
import itertools

# Set up argument parser to accept various parameters
parser = argparse.ArgumentParser(description="Camera Extrinsic Calibration Script")
parser.add_argument(
    "--file",
    type=str,
    default="setup.json5",
    help="Path to the cameras declarations file",
)
parser.add_argument(
    "--n",
    type=int,
    default=10,
    help="Number of calibration images required per camera pair",
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
    "--pivot",
    type=int,
    default=None,
    help="Index of the pivot (reference) camera",
)
parser.add_argument(
    "--window_scale",
    type=float,
    default=0.7,
    help="Scale of a window",
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

# Check for intrinsic parameters and collect camera indices
camera_indices = []
for camera_conf in cameras_confs:
    idx = camera_conf["index"]
    if "intrinsic" not in camera_conf:
        print(
            f"Error: Camera {idx} does not have intrinsic parameters. Please run intrinsic calibration first."
        )
        sys.exit(1)
    camera_indices.append(idx)

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

# Initialize video captures
print("\nLaunching...")
cameras = {}
for camera_conf in cameras_confs:
    idx = camera_conf["index"]
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera {idx}")
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
            f"Camera {idx} does not support manual focus! (or an invalid focus value provided)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize camera data
    cameras[idx] = {
        "index": idx,
        "cap": cap,
        "image_size": None,
    }

# Prepare data structures
pair_imgpoints = {pair: [] for pair in itertools.combinations(camera_indices, 2)}
pair_objpoints = {pair: [] for pair in itertools.combinations(camera_indices, 2)}
pair_transformations = {}  # To store relative transformations (R, T)

print()
print("=== Camera Extrinsic Calibration Script ===")
print("Instructions:")
print(
    f"1. Ensure that the calibration pattern ({chessboard_size[0]}x{chessboard_size[1]} chessboard of {square_size}mm squares) is visible in as many cameras as possible."
)
print("2. Press 'c' to capture calibration images from all cameras.")
print("   The script will print which cameras detected the pattern.")
print("3. Press 's' to perform calibration when ready.")
print(
    f"   Calibration requires at least {calibration_images_needed} common images per camera pair, and each camera must be involved in at least one such pair."
)
print(
    "4. After calibration, the script will write the extrinsic parameters back to the cameras file."
)

# Set up windows for each camera feed
for idx in cameras:
    cv2.namedWindow(f"Camera_{idx}", cv2.WINDOW_NORMAL)

# Prepare object points based on the real-world dimensions of the calibration pattern
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Capture images
while True:
    # Read frames from all cameras
    for idx in cameras:
        cap = cameras[idx]["cap"]
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read from camera {idx}")
            continue
        cameras[idx]["frame"] = frame

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
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

        assert (
            cameras[idx]["image_size"] is None
            or cameras[idx]["image_size"] == gray.shape[::-1]
        )
        cameras[idx]["image_size"] = gray.shape[::-1]

        cameras[idx]["corners"] = corners

        # Resize the frame before displaying
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
        sys.exit()

    elif key & 0xFF == ord("c"):
        print("Capturing calibration images...")
        # Convert frames to grayscale and find chessboard corners
        detected_cameras = []
        corners_dict = {}

        for idx in cameras:
            corners = cameras[idx]["corners"]
            if corners is not None:
                corners_dict[idx] = corners
                detected_cameras.append(idx)

        if detected_cameras:
            print(f"Pattern detected in cameras: {detected_cameras}")
            # Update counts for camera pairs
            for pair in itertools.combinations(detected_cameras, 2):
                sorted_pair = tuple(sorted(pair))
                pair_imgpoints[sorted_pair].append(
                    (corners_dict[sorted_pair[0]], corners_dict[sorted_pair[1]])
                )
                pair_objpoints[sorted_pair].append(objp)

            # Print for observation
            counts = {k: len(v) for k, v in pair_imgpoints.items()}
            print("Current counts per camera pair:")
            for pair in counts:
                print(f"Cameras {pair[0]} and {pair[1]}: {counts[pair]} frames")
        else:
            print("Pattern not detected in any camera.")

    elif key & 0xFF == ord("s"):
        # Check if each camera is involved in at least one pair with sufficient images
        cameras_with_sufficient_pairs = set()
        for pair, elems in pair_imgpoints.items():
            count = len(elems)
            if count >= calibration_images_needed:
                cameras_with_sufficient_pairs.update(pair)
        missing_cameras = set(camera_indices) - cameras_with_sufficient_pairs
        if not missing_cameras:
            print("\nProceeding to calibration...")
            print("Number of frames in common for each camera pair:")
            for pair, elems in pair_imgpoints.items():
                count = len(elems)
                print(f"Cameras {pair[0]} and {pair[1]}: {count} frames")
            break
        else:
            print("\nNot all cameras have sufficient data for calibration.")
            print(f"Cameras needing more images: {sorted(missing_cameras)}")
            print("Current counts per camera pair:")
            for pair, elems in pair_imgpoints.items():
                count = len(elems)
                print(f"Cameras {pair[0]} and {pair[1]}: {count} frames")
            print(f"Need at least {calibration_images_needed} frames per camera pair.")
            continue

# Release windows
for idx in cameras:
    cameras[idx]["cap"].release()
cv2.destroyAllWindows()

# Perform stereo calibration for all pairs with sufficient data
print("\nPerforming stereo calibration for all camera pairs with sufficient data...")
for pair, elems in pair_imgpoints.items():
    if len(elems) < calibration_images_needed:
        print(f"Not enough data to calibrate cameras {pair[0]} and {pair[1]}.")
        continue
    idx1, idx2 = pair
    objpoints = pair_objpoints[pair]
    imgpoints1 = [imgpair[0] for imgpair in elems]
    imgpoints2 = [imgpair[1] for imgpair in elems]

    # Get intrinsic parameters
    cam1_conf = next(conf for conf in cameras_confs if conf["index"] == idx1)
    cam2_conf = next(conf for conf in cameras_confs if conf["index"] == idx2)

    mtx1 = np.array(
        [
            [
                cam1_conf["intrinsic"]["focal_length_pixels"]["x"],
                cam1_conf["intrinsic"]["skew_coefficient"],
                cam1_conf["intrinsic"]["principal_point"]["x"],
            ],
            [
                0,
                cam1_conf["intrinsic"]["focal_length_pixels"]["y"],
                cam1_conf["intrinsic"]["principal_point"]["y"],
            ],
            [0, 0, 1],
        ]
    )
    dist1 = np.array(cam1_conf["intrinsic"]["dist_coeffs"])

    mtx2 = np.array(
        [
            [
                cam2_conf["intrinsic"]["focal_length_pixels"]["x"],
                cam2_conf["intrinsic"]["skew_coefficient"],
                cam2_conf["intrinsic"]["principal_point"]["x"],
            ],
            [
                0,
                cam2_conf["intrinsic"]["focal_length_pixels"]["y"],
                cam2_conf["intrinsic"]["principal_point"]["y"],
            ],
            [0, 0, 1],
        ]
    )
    dist2 = np.array(cam2_conf["intrinsic"]["dist_coeffs"])

    # Stereo calibration
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1,
        imgpoints2,
        mtx1,
        dist1,
        mtx2,
        dist2,
        cameras[idx1]["image_size"],
        criteria=(
            cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
            100,
            1e-5,
        ),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    # Store the relative transformation
    pair_transformations[pair] = {"R": R, "T": T}

    # Compute reprojection error (optional)
    print(f"Stereo calibration between cameras {idx1} and {idx2} completed.")

# Build camera graph
camera_graph = {idx: [] for idx in camera_indices}
for pair in pair_transformations:
    idx1, idx2 = pair
    camera_graph[idx1].append(idx2)
    camera_graph[idx2].append(idx1)


# Function to find paths from pivot to other cameras
def find_transformation(pivot_idx, target_idx, visited=None):
    if visited is None:
        visited = set()
    visited.add(pivot_idx)
    if pivot_idx == target_idx:
        return {"R": np.eye(3), "T": np.zeros((3, 1))}
    for neighbor in camera_graph[pivot_idx]:
        if neighbor in visited:
            continue
        pair = tuple(sorted((pivot_idx, neighbor)))
        rel_trans = pair_transformations.get(pair)
        if rel_trans is None:
            continue
        res = find_transformation(neighbor, target_idx, visited)
        if res is not None:
            # Compose transformations
            if pivot_idx < neighbor:
                # Transformation from pivot to neighbor
                R_pn = rel_trans["R"]
                T_pn = rel_trans["T"]
            else:
                # Need to invert the transformation
                R_pn = rel_trans["R"].T
                T_pn = -rel_trans["R"].T @ rel_trans["T"]
            # Compose
            R = res["R"] @ R_pn
            T = res["R"] @ T_pn + res["T"]
            return {"R": R, "T": T}
    return None


# Compute transformations from pivot camera to all other cameras
print("\nComputing transformations relative to the pivot camera...")
for idx in camera_indices:
    if idx == reference_idx:
        # Pivot camera, identity transformation
        cam_conf = next(conf for conf in cameras_confs if conf["index"] == idx)
        cam_conf["extrinsic"] = {
            "translation_centimeters": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            },
            "rotation_radians": {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
            },
            "reprojection_error": 0.0,
        }
        continue
    # Find transformation path
    res = find_transformation(reference_idx, idx)
    if res is None:
        print(f"Could not find a path from camera {reference_idx} to camera {idx}")
        continue
    R = res["R"]
    T = res["T"]

    # Convert rotation matrix to Euler angles (yaw, pitch, roll)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        yaw = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0

    # Convert translation to centimeters
    T_cm = T.flatten() * 0.1  # Assuming T is in millimeters

    # Store extrinsic parameters
    cam_conf = next(conf for conf in cameras_confs if conf["index"] == idx)
    cam_conf["extrinsic"] = {
        "translation_centimeters": {
            "x": float(T_cm[0]),
            "y": float(T_cm[1]),
            "z": float(T_cm[2]),
        },
        "rotation_radians": {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
        },
    }

    print(f"Computed transformation from camera {reference_idx} to camera {idx}.")

# Save calibrations
with open(cameras_path, "w") as f:
    json5.dump(cameras_confs, f, indent=4)
print("\nCameras file updated with extrinsic parameters.")
