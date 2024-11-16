import json
import cv2
import mediapipe as mp
import numpy as np
import json5
import argparse
import sys

from kinematics import mp2bones


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
        mtx = np.array(
            [
                [
                    intrinsic["focal_length_pixels"]["x"],
                    intrinsic["skew_coefficient"],
                    intrinsic["principal_point"]["x"],
                ],
                [
                    0,
                    intrinsic["focal_length_pixels"]["y"],
                    intrinsic["principal_point"]["y"],
                ],
                [0, 0, 1],
            ]
        )
        dist_coeffs = np.array(intrinsic["dist_coeffs"])
        # Extrinsic parameters
        extrinsic = cam_conf["extrinsic"]
        T_cm = extrinsic["translation_centimeters"]
        R_rad = extrinsic["rotation_radians"]
        # Convert rotation from Euler angles to rotation matrix
        yaw = R_rad["yaw"]
        pitch = R_rad["pitch"]
        roll = R_rad["roll"]
        R = euler_angles_to_rotation_matrix(yaw, pitch, roll)
        T = (
            np.array([[T_cm["x"]], [T_cm["y"]], [T_cm["z"]]]) * 10
        )  # Convert to millimeters
        # Store parameters
        cameras[idx] = {
            "mtx": mtx,
            "dist": dist_coeffs,
            "R": R,
            "T": T,
            "cap": None,
            "frame": None,
            "hand_landmarks": None,
            "hands_tracker": None,  # To be initialized later
            "focus": cam_conf.get("focus", 0),  # Retrieve focus value
        }
    return cameras


def euler_angles_to_rotation_matrix(yaw, pitch, roll):
    """
    Converts Euler angles (in radians) to a rotation matrix.
    """
    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    R = R_z @ R_y @ R_x
    return R


def triangulate_points(cameras, point_idx):
    """
    Triangulate 3D point from multiple camera views.
    """
    projections = []
    points = []
    for idx in cameras:
        cam = cameras[idx]
        if cam["hand_landmarks"] is not None:
            lm = cam["hand_landmarks"][point_idx]
            # Convert normalized coordinates to pixel coordinates
            h, w, _ = cam["frame"].shape
            x = lm.x * w
            y = lm.y * h
            # Undistort points
            undistorted = cv2.undistortPoints(
                np.array([[[x, y]]], dtype=np.float32),
                cam["mtx"],
                cam["dist"],
                P=cam["mtx"],
            )
            points.append(undistorted[0][0])
            # Compute projection matrix
            RT = np.hstack((cam["R"], cam["T"]))
            P = cam["mtx"] @ RT
            projections.append(P)
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
        return X[:3]
    else:
        return None


def main():
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
    args = parser.parse_args()
    cameras_path = args.file

    # Load camera parameters
    cameras = load_camera_parameters(cameras_path)
    if len(cameras) < 2:
        print("Need at least two cameras with calibration data.")
        sys.exit(1)

    # Initialize video captures and MediaPipe Hands trackers
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    for idx in cameras:
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
        focus_value = cameras[idx]["focus"]
        focus_supported = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
        if not focus_supported:
            print(
                f"Camera {idx} does not support manual focus! (or an invalid focus value provided)",
                file=sys.stderr,
            )
            sys.exit(1)
        cameras[idx]["cap"] = cap
        cv2.namedWindow(f"Camera_{idx}", cv2.WINDOW_AUTOSIZE)
        # Initialize MediaPipe Hands tracker for each camera
        cameras[idx]["hands_tracker"] = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9,
        )

    # Start capturing frames
    while True:
        # Capture frames
        for idx in cameras:
            cap = cameras[idx]["cap"]
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read from camera {idx}")
                continue
            cameras[idx]["frame"] = frame

        # Process frames with MediaPipe and display
        for idx in cameras:
            frame = cameras[idx]["frame"]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = cameras[idx]["hands_tracker"]
            results = hands.process(rgb_frame)
            cameras[idx]["hand_landmarks"] = None
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    if handedness.classification[0].label == "Right":
                        cameras[idx]["hand_landmarks"] = hand_landmarks.landmark
                        # Draw landmarks using the provided code
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(0, 0, 255), thickness=2, circle_radius=4
                            ),
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        )
                        break  # Only consider one right hand

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
            break
        elif key & 0xFF == ord("s"):
            # Collect landmarks from all cameras
            points_3d = []
            num_landmarks = 21  # MediaPipe Hands has 21 landmarks
            valid_indices = []
            for point_idx in range(num_landmarks):
                point_3d = triangulate_points(cameras, point_idx)
                if point_3d is not None:
                    points_3d.append(point_3d)
                    valid_indices.append(point_idx)
            if points_3d:
                points_3d = np.array(points_3d)
                bones = mp2bones(points_3d)
                with open("rotation_data.json", "w") as f:
                    json.dump(bones, f, indent=4)
            else:
                print("Not enough data to reconstruct hand in 3D.")

    # Release resources
    for idx in cameras:
        cameras[idx]["cap"].release()
        cameras[idx]["hands_tracker"].close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
