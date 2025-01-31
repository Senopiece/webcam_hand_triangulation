import cv2
import numpy as np
from typing import Any, Callable, List, Tuple
import mediapipe as mp

from models import CameraParams, ContextedLandmark
from triangulation import triangulate_lmcs
from projection import distorted_project
from finalizable_queue import EmptyFinalized, FinalizableQueue
from draw_utils import draw_left_top
from hand_normalization import normalize_hand


mp_hands = mp.solutions.hands
num_landmarks = 21  # MediaPipe Hands has 21 landmarks


def processing_loop(
        landmark_transforms: List[Callable[..., Any]],
        draw_origin_landmarks: bool,
        desired_window_size: Tuple[float, float],
        cameras_params: List[CameraParams],
        coupled_frames_queue: FinalizableQueue,
        hand_points_queue: FinalizableQueue,
        out_queues: List[FinalizableQueue],
    ):
    processors = [
        mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9,
        ) for _ in range(len(cameras_params))
    ]

    while True:
        try:
            elem = coupled_frames_queue.get()
        except EmptyFinalized:
            break

        index: int = elem[0]
        frames: List[Tuple[np.ndarray, int]] = elem[1]
        coupling_fps: int = elem[2]

        cap_fps: List[int] = [item[1] for item in frames]
        frames: List[np.ndarray] = [item[0] for item in frames]

        # Find landmarks
        landmarks = [[] for _ in range(num_landmarks)] # lm = landmarks[lm_id][pov_id]
        for landmark_transform, processor, frame in zip(landmark_transforms, processors, frames):
            # Convert to RGB and process
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = processor.process(frame_rgb)

            # Extract landmarks
            pov_landmarks = [None for _ in range(num_landmarks)]
            if res.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    res.multi_hand_landmarks, 
                    res.multi_handedness
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
        chosen_cams = [] # chosen_cams[lm_id]
        points_3d = [] # points_3d[lm_id]
        if all(
            sum(
                1 for lm in lm_povs if lm is not None
            ) >= 2 for lm_povs in landmarks
        ):
            for lm_povs in landmarks:
                lmcs = []
                for pov_i, (frame, pov_params, lm) in enumerate(zip(frames, cameras_params, lm_povs)):
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
        
        # Send to 3d visualization
        hand_points_queue.put((index, (normalize_hand(points_3d) if points_3d else [], coupling_fps, coupled_frames_queue.qsize())))
        
        # Resize frames before drawing
        for i, frame in enumerate(frames):
            frames[i] = cv2.resize(
                frame, desired_window_size, interpolation=cv2.INTER_AREA
            )

        # Draw original landmarks
        if draw_origin_landmarks:
            for origin_landmarks, frame in zip(landmarks, frames):
                if origin_landmarks is None:
                    continue

                h, w, _ = frame.shape
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_pt = origin_landmarks[start_idx]
                    end_pt = origin_landmarks[end_idx]
                    cv2.line(
                        frame,
                        (int(start_pt.x * w), int(start_pt.y * h)),
                        (int(end_pt.x * w), int(end_pt.y * h)),
                        color=(0, 200, 200),
                        thickness=1,
                    )
                for lm in origin_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), radius=3, color=(0, 200, 200), thickness=-1)
        
        # Draw reprojected landmarks
        if points_3d:
            for pov_i, (frame, params) in enumerate(zip(frames, cameras_params)):
                # Project 3D points onto each camera
                reprojected_lms: List[Tuple[float, float]] = []
                for point_3d in points_3d:
                    x, y = distorted_project(
                        point_3d,
                        params.extrinsic.rvec,
                        params.extrinsic.T,
                        params.intrinsic.mtx,
                        params.intrinsic.dist_coeffs,
                    )

                    # camera pixel coordinates -> normalized coordinates
                    x, y = x / params.size[0], y / params.size[1]

                    # normalized coordinates -> real viewport pixel coordinates
                    h, w, _ = frame.shape
                    x, y = x * w, y * h

                    # Clip to image size
                    x = max(min(int(x), frame.shape[1] - 1), 0)
                    y = max(min(int(y), frame.shape[0] - 1), 0)

                    reprojected_lms.append((x, y))

                # Draw reptojected landmarks
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_pt = reprojected_lms[start_idx]
                    end_pt = reprojected_lms[end_idx]
                    cv2.line(
                        frame,
                        start_pt,
                        end_pt,
                        color=(255, 255, 255),
                        thickness=1,
                    )
                for involved_in_triangulating_this_lm, lm in zip(chosen_cams, reprojected_lms):
                    if pov_i in involved_in_triangulating_this_lm:
                        color = (0, 255, 0)  # Chosen camera
                    else:
                        color = (255, 0, 0)  # Others
                    cv2.circle(frame, lm, radius=3, color=color, thickness=-1)
        
        # Draw cap fps for every pov
        for fps, frame in zip(cap_fps, frames):
            draw_left_top(0, f"Capture FPS: {fps}", frame)
        
        # Write results
        for out_queue, frame in zip(out_queues, frames):
            out_queue.put((index, frame))
        
        coupled_frames_queue.task_done()
    
    for processor in processors:
        processor.close()
    
    print("A processing loop is finished.")