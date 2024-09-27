import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Capture video from the webcam.
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame from camera.")
            continue

        # Flip the frame horizontally for a later selfie-view display.
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

        # Display the frame with landmarks and confidence score.
        cv2.imshow("MediaPipe Hands with Confidence", frame)

        # Exit when 'q' key is pressed.
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

# Release the video capture object.
cap.release()
cv2.destroyAllWindows()
