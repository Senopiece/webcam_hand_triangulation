import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime, timedelta
import argparse
import json
import matplotlib.pyplot as plt

from hand_nature import calculate_natureness

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up argument parser to accept a path to the cameras declarations file
parser = argparse.ArgumentParser(
    description="Multi-camera hand detection and 3D reconstruction"
)
parser.add_argument(
    "--cameras",
    type=str,
    default="cameras.json",
    help="Path to the cameras declarations file",
)

args = parser.parse_args()
cameras_path = args.cameras

# Load camera configurations from the JSON file
with open(cameras_path, "r") as f:
    cameras = json.load(f)

for camera in cameras:
    assert isinstance(camera["latency"], int), "Latency must be an integer"
    assert isinstance(camera["index"], int), "Camera index must be an integer"

# Initialize video captures and buffers
for camera in cameras:
    cap = cv2.VideoCapture(camera["index"])
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera['index']}")
        exit()
    camera["cap"] = cap
    camera["buff"] = []
    camera["fps"] = 0
    camera["fps_label"] = 0
    camera["show_frame"] = None
    camera["hands"] = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.95,
        min_tracking_confidence=0.95,
        model_complexity=1
    )

# Set up windows for each camera feed
for camera in cameras:
    cv2.namedWindow(f"Camera_{camera['index']}", cv2.WINDOW_NORMAL)

# Calibration parameters
calibration_mode = True
calibration_count = 0
calibration_images_needed = 10  # Number of calibration images required
chessboard_size = (9, 6)  # Number of inner corners per chessboard row and column
square_size = 25.0  # Size of a square in your defined unit (e.g., millimeters)

# Prepare object points based on the real-world dimensions of the calibration pattern
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = {
    camera["index"]: [] for camera in cameras
}  # 2D points in image plane for each camera

# Initialize variables for calibration
camera_matrices = {}
dist_coeffs = {}
extrinsics = {}

# Initialize 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plt.ion()
plt.show()

last_fps_measure = datetime.now()

while True:
    if datetime.now() - last_fps_measure >= timedelta(seconds=1):
        for camera in cameras:
            camera["fps_label"] = camera["fps"]
            camera["fps"] = 0
        last_fps_measure = datetime.now()

    for camera in cameras:
        cap = camera["cap"]
        idx = camera["index"]
        latency = camera["latency"]
        buff = camera["buff"]

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read from camera {idx}")
            continue
        camera["fps"] += 1

        # Add latency per camera
        buff.append([frame, datetime.now()])
        show_frame = None
        while len(buff) != 0:
            frame, timestamp = buff[0]

            if datetime.now() - timestamp < timedelta(microseconds=latency):
                break

            buff.pop(0)
            show_frame = frame

        if show_frame is not None:
            cv2.putText(
                show_frame,
                f"FPS: {camera["fps_label"]:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            camera["show_frame"] = show_frame

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

    if key & 0xFF == ord("c") and calibration_mode:
        # Capture frames from all cameras for calibration
        calibration_success = True
        gray_images = {}
        temp_imgpoints = {}
        for camera in cameras:
            idx = camera["index"]
            frame = camera["show_frame"]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_images[idx] = gray

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
                temp_imgpoints[idx] = corners2
                # Draw and display the corners
                cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
                cv2.imshow(f"Camera_{idx}", frame)
            else:
                print(f"Calibration pattern not found in camera {idx}")
                calibration_success = False
                break  # Cannot proceed without all cameras detecting the pattern

        if calibration_success:
            objpoints.append(objp)
            for idx in temp_imgpoints:
                imgpoints[idx].append(temp_imgpoints[idx])
            calibration_count += 1
            print(f"Calibration image {calibration_count} collected.")
        else:
            print("Calibration failed for this set of images.")

        if calibration_count >= calibration_images_needed:
            print("Enough calibration images collected. Performing calibration...")
            # Perform calibration for each camera
            for camera in cameras:
                idx = camera["index"]
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints,
                    imgpoints[idx],
                    gray_images[idx].shape[::-1],
                    None,
                    None,
                )
                camera_matrices[idx] = mtx
                dist_coeffs[idx] = dist

            # Choose one camera as the reference
            reference_idx = cameras[0]["index"]
            extrinsics[reference_idx] = (np.eye(3), np.zeros((3, 1)))
            # Perform stereo calibration between reference camera and others
            for camera in cameras:
                idx = camera["index"]
                if idx == reference_idx:
                    continue
                ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                    objpoints,
                    imgpoints[reference_idx],
                    imgpoints[idx],
                    camera_matrices[reference_idx],
                    dist_coeffs[reference_idx],
                    camera_matrices[idx],
                    dist_coeffs[idx],
                    gray_images[idx].shape[::-1],
                    flags=cv2.CALIB_FIX_INTRINSIC,
                )
                extrinsics[idx] = (R, T)
                # For debugging, print the projection matrices
                P = camera_matrices[idx] @ np.hstack((R, T))
                print(f"Camera {idx} Projection Matrix:\n{P}")

            # Now we have intrinsic and extrinsic parameters
            calibration_mode = False
            print("Calibration completed.")

    elif not calibration_mode:
        # Hand detection and 3D reconstruction
        landmarks_2d = {}
        for camera in cameras:
            hands = camera["hands"]
            idx = camera["index"]
            frame = camera["show_frame"]
            # Convert the BGR frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Improve performance by making the frame read-only
            image_rgb.flags.writeable = False

            # Process the frame and find hand landmarks
            results = hands.process(image_rgb)

            image_rgb.flags.writeable = True
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=(0, 0, 255), thickness=2, circle_radius=4
                        ),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    )
                    # Collect 2D landmarks
                    landmark_list = []
                    for lm in hand_landmarks.landmark:
                        x = lm.x * frame.shape[1]
                        y = lm.y * frame.shape[0]
                        landmark_list.append([x, y])
                    landmarks_2d[idx] = landmark_list
            # Display the frame
            cv2.imshow(f"Camera_{idx}", frame)

        # Triangulate if landmarks from all cameras are available
        if len(landmarks_2d) == len(cameras):
            num_landmarks = len(landmarks_2d[reference_idx])
            landmarks_3d = []
            for i in range(num_landmarks):
                # Collect the 2D observations
                points_2d = []
                projection_matrices = []
                for camera in cameras:
                    idx = camera["index"]
                    x, y = landmarks_2d[idx][i]
                    points_2d.append([x, y])
                    R, T = extrinsics[idx]
                    P = camera_matrices[idx] @ np.hstack((R, T))
                    projection_matrices.append(P)

                # Triangulate the 3D point
                def triangulate_point(points_2d, projection_matrices):
                    num_views = len(points_2d)
                    A = []
                    for j in range(num_views):
                        x, y = points_2d[j]
                        P = projection_matrices[j]
                        A.append(x * P[2, :] - P[0, :])
                        A.append(y * P[2, :] - P[1, :])
                    A = np.array(A)
                    _, _, Vt = np.linalg.svd(A)
                    X = Vt[-1]
                    X = X / X[3]  # Normalize homogeneous coordinate
                    return X[:3]

                point_3d = triangulate_point(points_2d, projection_matrices)
                landmarks_3d.append(point_3d)

            # Plot the 3D hand
            ax.clear()
            X = [pt[0] for pt in landmarks_3d]
            Y = [pt[1] for pt in landmarks_3d]
            Z = [pt[2] for pt in landmarks_3d]
            ax.scatter(X, Y, Z, c="r", marker="o")

            # Optionally, draw the hand connections
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d):
                    ax.plot(
                        [X[start_idx], X[end_idx]],
                        [Y[start_idx], Y[end_idx]],
                        [Z[start_idx], Z[end_idx]],
                        "b-",
                    )
            plt.draw()
            plt.pause(0.001)

            natureness_score = calculate_natureness(landmarks_3d)
            print(f"Natureness score: {natureness_score:.2f}")

    else:
        for camera in cameras:
            idx = camera["index"]
            frame = camera["show_frame"]
            if frame is not None:
                cv2.imshow(f"Camera_{idx}", frame)

# Release resources
for camera in cameras:
    camera["cap"].release()
cv2.destroyAllWindows()
