# Merged Camera Calibration Script (Intrinsic and Extrinsic)

import sys
import cv2
import numpy as np
import json5
import argparse
import itertools

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
    help="Use existing intrinsics, don't overwrite them",
    action="store_true",
)
parser.add_argument(
    "--pivot",
    type=int,
    default=None,
    help="Index of the pivot (reference) camera",
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
cameras = {}
for camera_conf in cameras_confs:
    idx = camera_conf["index"]
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera {idx}", file=sys.stderr)
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
        "imgpoints": [],  # For intrinsic calibration
        "corners": None,
    }

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

# Initialize counts
camera_image_counts = {idx: 0 for idx in camera_indices}

# Prepare data structures
pair_imgpoints = {pair: [] for pair in itertools.combinations(camera_indices, 2)}
pair_objpoints = {pair: [] for pair in itertools.combinations(camera_indices, 2)}
pair_transformations = {}  # To store relative transformations (R, T)

print()
print("=== Camera Calibration Script ===")
print("Instructions:")
print(
    f"1. Ensure that the calibration pattern ({chessboard_size[0]}x{chessboard_size[1]} chessboard of {square_size}mm squares) is visible in as many cameras as possible."
)
print("2. Press 'c' to capture calibration images from all cameras.")
print("   The script will print which cameras detected the pattern.")
print("3. Press 's' to perform calibration when ready.")
print(
    f"   Calibration requires at least {calibration_images_needed} images per camera for intrinsic calibration, and sufficient images per camera pair."
)
print(
    "4. After calibration, the script will write the intrinsic and extrinsic parameters back to the cameras file."
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

    # Display frames in windows
    for idx in cameras:
        frame = cameras[idx]["frame"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners, meta = cv2.findChessboardCornersSBWithMeta(
            gray,
            chessboard_size,
            flags=(
                cv2.CALIB_CB_MARKER
                | cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            ),
        )

        if ret_corners:
            if meta.shape[0] != chessboard_rows:
                corners = corners.reshape(-1, 2)
                corners = corners.reshape(*chessboard_size, 2)
                corners = corners.transpose(1, 0, 2)
                corners = corners.reshape(-1, 2)
                corners = corners[:, np.newaxis, :]
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret_corners)
            cameras[idx]["corners"] = corners
        else:
            cameras[idx]["corners"] = None

        if cameras[idx]["image_size"] is None:
            cameras[idx]["image_size"] = gray.shape[::-1]
        else:
            assert cameras[idx]["image_size"] == gray.shape[::-1]

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
        # Collect detected corners
        detected_cameras = []
        corners_dict = {}

        for idx in cameras:
            corners = cameras[idx]["corners"]
            if corners is not None:
                corners_dict[idx] = corners
                detected_cameras.append(idx)
                cameras[idx]["imgpoints"].append(corners)
                camera_image_counts[idx] += 1

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
            # Print counts per camera
            print("Current counts per camera:")
            for idx in camera_indices:
                print(f"Camera {idx}: {camera_image_counts[idx]} frames")
        else:
            print("Pattern not detected in any camera.")

    elif key & 0xFF == ord("s"):
        # Check if each camera has enough images for intrinsic calibration
        cameras_with_sufficient_images = [
            idx
            for idx in camera_indices
            if camera_image_counts[idx] >= calibration_images_needed
        ]
        missing_cameras = set(camera_indices) - set(cameras_with_sufficient_images)
        if missing_cameras:
            print("\nNot all cameras have sufficient data for intrinsic calibration.")
            print(f"Cameras needing more images: {sorted(missing_cameras)}")
            print("Current counts per camera:")
            for idx in camera_indices:
                print(f"Camera {idx}: {camera_image_counts[idx]} frames")
            print(f"Need at least {calibration_images_needed} frames per camera.")
            continue
        else:
            print("\nProceeding to calibration...")
            print("Number of frames collected per camera:")
            for idx in camera_indices:
                print(f"Camera {idx}: {camera_image_counts[idx]} frames")
            break

# Release resources after loop
for idx in cameras:
    cameras[idx]["cap"].release()
cv2.destroyAllWindows()

# Perform intrinsic calibrations
for idx in cameras:
    camera = cameras[idx]
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

    mtx1 = cameras[idx1]["mtx"]
    dist1 = cameras[idx1]["dist_coeffs"]

    mtx2 = cameras[idx2]["mtx"]
    dist2 = cameras[idx2]["dist_coeffs"]

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
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
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
print("\nCameras file updated with intrinsic and extrinsic parameters.")
