from datetime import datetime, timedelta
import cv2
import mediapipe as mp
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up argument parser to accept camera index and latency as command-line arguments
parser = argparse.ArgumentParser(
    description="Select which camera to use and set latency"
)
parser.add_argument("--camera", type=int, default=0, help="Camera index (default is 0)")
parser.add_argument(
    "--latency",
    type=float,
    default=0,
    help="Additional camera latency",
)

# Parse the command-line arguments
args = parser.parse_args()
camera_index = args.camera
latency = args.latency

buff = []

# Open the selected camera
cap = cv2.VideoCapture(camera_index)

# Check if the camera is opened successfully
if not cap.isOpened():
    print(f"Error: Could not open camera {camera_index}")
    exit()

# Set up a window name for the camera feed
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

# Initialize variables for FPS calculation
prev_time = datetime.now()
fps = 0

with mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:
    while True:
        # Capture frame-by-frame from the selected camera
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print(f"Error: Could not read from camera {camera_index}")
            break

        # Convert the BGR frame to RGB.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Improve performance by making the frame read-only.
        image_rgb.flags.writeable = False

        # Process the frame and find hand landmarks.
        results = hands.process(image_rgb)

        # Draw hand landmarks on the original frame.
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 0, 255), thickness=2, circle_radius=4
                    ),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )

                # Get the confidence score.
                score = handedness.classification[0].score

                # Get the first landmark's coordinates to position the text.
                x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y = int(hand_landmarks.landmark[0].y * frame.shape[0])

                # Display the confidence score on the frame.
                cv2.putText(
                    frame,
                    f"{score:.2f}",
                    (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

        # Get the current time
        current_time = datetime.now()

        # Calculate FPS
        fps = 1000000 / (current_time - prev_time).microseconds
        prev_time = current_time

        # Display the FPS on the frame
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Add latency (delay) between frames
        buff.append([frame, current_time])

        # Display the resulting frame
        while len(buff) != 0:
            frame, timestamp = buff[0]
            if current_time - timestamp < timedelta(microseconds=latency):
                break
            buff.pop(0)
            cv2.imshow("Camera", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# When everything is done, release the capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
