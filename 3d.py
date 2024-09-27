import math
import matplotlib

from hand_nature import calculate_natureness

matplotlib.use("Qt5Agg")

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# Initialize MediaPipe Hands.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize Matplotlib for 3D plotting.
plt.ion()  # Turn on interactive mode.
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Show the plot window.
plt.show(block=False)

# Capture video from the webcam.
cap = cv2.VideoCapture(0)

# **User Input: Measure your hand length in centimeters (wrist to middle fingertip)**
real_hand_length_cm = 18.5  # Replace this with your actual measurement

with mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame from camera.")
            continue

        # Flip the frame horizontally for a selfie-view display.
        frame = cv2.flip(frame, 1)
        # Convert the BGR frame to RGB.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Improve performance by making the frame read-only.
        image_rgb.flags.writeable = False

        # Process the frame and find hand landmarks.
        results = hands.process(image_rgb)

        # Draw hand landmarks on the original frame.
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Clear the axes for the new plot.
        ax.cla()

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
                x0 = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y0 = int(hand_landmarks.landmark[0].y * frame.shape[0])

                # Display the confidence score on the frame.
                cv2.putText(
                    frame,
                    f"Confidence: {score:.2f}",
                    (x0, y0 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                # Extract landmark coordinates.
                x = np.array([lm.x for lm in hand_landmarks.landmark])
                y = np.array([lm.y for lm in hand_landmarks.landmark])
                z = np.array([lm.z for lm in hand_landmarks.landmark])

                # **Compute the normalized distance between wrist (0) and middle fingertip (12)**
                idx_wrist = 0
                idx_middle_tip = 12

                # Coordinates of wrist and middle fingertip
                wrist = np.array([x[idx_wrist], y[idx_wrist], z[idx_wrist]])
                middle_tip = np.array(
                    [x[idx_middle_tip], y[idx_middle_tip], z[idx_middle_tip]]
                )

                # Euclidean distance in normalized coordinates
                normalized_distance = np.linalg.norm(middle_tip - wrist)

                # **Calculate the scaling factor**
                scale_factor = real_hand_length_cm / normalized_distance

                # **Scale all coordinates to real-world units (centimeters)**
                x_real = x * scale_factor
                y_real = y * scale_factor
                z_real = z * scale_factor

                # Prepare the 3D landmarks for the natureness calculation
                landmarks_3d = list(zip(x_real, y_real, z_real))

                # **Calculate the natureness score**
                natureness_score = calculate_natureness(landmarks_3d)
                print(f"Natureness score: {natureness_score:.2f}")

                # Display the natureness score on the frame.
                cv2.putText(
                    frame,
                    f"Natureness: {natureness_score:.2f}",
                    (x0, y0 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                # Plot the landmarks in real-world coordinates.
                ax.scatter(x_real, y_real, z_real, c="red", marker="o")

                # Plot the connections.
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    x_values = [x_real[start_idx], x_real[end_idx]]
                    y_values = [y_real[start_idx], y_real[end_idx]]
                    z_values = [z_real[start_idx], z_real[end_idx]]
                    ax.plot(x_values, y_values, z_values, c="blue")

                # Set plot limits and labels.
                ax.set_xlim([np.min(x_real) - 5, np.max(x_real) + 5])
                ax.set_ylim([np.min(y_real) - 5, np.max(y_real) + 5])
                ax.set_zlim([np.min(z_real) - 5, np.max(z_real) + 5])
                ax.set_xlabel("X (cm)")
                ax.set_ylabel("Y (cm)")
                ax.set_zlabel("Z (cm)")
                ax.set_title("3D Hand Landmarks (Real-World Coordinates)")

            # Draw the updated plot.
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

        else:
            # If no hands are detected, set the title accordingly.
            ax.set_title("No Hand Detected")

        # Display the frame with landmarks, confidence score, and natureness score.
        cv2.imshow("MediaPipe Hands with Natureness Score", frame)

        # Exit when 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close the Matplotlib figure.
    plt.close()

# Release the video capture object.
cap.release()
cv2.destroyAllWindows()
