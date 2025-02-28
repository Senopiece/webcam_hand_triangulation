import cv2
import numpy as np
from typing import Any, Callable, List
import mediapipe as mp

from .models import CameraParams, ContextedLandmark
from .triangulation import triangulate_lmcs


mp_hands = mp.solutions.hands  # type: ignore
num_landmarks = 21  # MediaPipe Hands has 21 landmarks


class HandTriangulator:
    def __init__(
        self,
        landmark_transforms: List[Callable[..., Any]],
        cameras_params: List[CameraParams],
    ):
        assert len(landmark_transforms) == len(cameras_params)
        self.landmark_transforms = landmark_transforms
        self.cameras_params = cameras_params
        self.processors = [
            mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.9,
                min_tracking_confidence=0.9,
            )
            for _ in range(len(cameras_params))
        ]

    def triangulate(self, frames: List[np.ndarray]):
        assert len(self.landmark_transforms) == len(frames)
        # Find landmarks
        landmarks = [[] for _ in range(num_landmarks)]  # lm = landmarks[lm_id][pov_id]
        for landmark_transform, processor, frame in zip(
            self.landmark_transforms, self.processors, frames
        ):
            # Convert to RGB and process
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = processor.process(frame_rgb)

            # Extract landmarks
            pov_landmarks = [None for _ in range(num_landmarks)]
            if res.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    res.multi_hand_landmarks, res.multi_handedness
                ):
                    if handedness.classification[0].label == "Left":
                        pov_landmarks = landmark_transform(hand_landmarks.landmark)
                        break

            # Store landmarks regrouping by landmark id first
            assert len(pov_landmarks) == num_landmarks
            for i, lm in enumerate(pov_landmarks):
                landmarks[i].append(lm)

        # Triangulate points across the cameras (if all landmarks are present at least on two cameras)
        # so that the full triangulation is dropped if cannot triangulate a landmark
        chosen_cams = []  # chosen_cams[lm_id]
        points_3d = []  # points_3d[lm_id]
        if all(
            sum(1 for lm in lm_povs if lm is not None) >= 2 for lm_povs in landmarks
        ):
            for lm_povs in landmarks:
                lmcs = []
                for pov_i, (frame, pov_params, lm) in enumerate(
                    zip(frames, self.cameras_params, lm_povs)
                ):
                    # Skip if lm is not present
                    if lm is None:
                        continue

                    # Landmark to pixel coord
                    h, w, _ = frame.shape
                    pixel_pt = [lm.x * w, lm.y * h]

                    # Undistort pixel coord
                    intrinsics = pov_params.intrinsic
                    undistorted_lm = cv2.undistortPoints(
                        np.array([[pixel_pt]], dtype=np.float32),
                        intrinsics.mtx,
                        intrinsics.dist_coeffs,
                        P=intrinsics.mtx,
                    )[0][0]

                    # Append the result
                    lmcs.append(
                        ContextedLandmark(
                            cam_idx=pov_i,
                            P=pov_params.P,
                            lm=undistorted_lm,
                        )
                    )

                chosen, point_3d = triangulate_lmcs(lmcs)

                chosen_cams.append(chosen)
                points_3d.append(point_3d)

        return landmarks, chosen_cams, points_3d

    def close(self):
        for processor in self.processors:
            processor.close()
        self.processors = []
        self.landmark_transforms = []
        self.cameras_params = []
