import os
import threading

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

import multiprocessing
import multiprocessing.synchronize
from typing import Any, Dict, List, Tuple
import mediapipe as mp
import argparse
import sys
import time

from cam_conf import load_cameras_parameters
from models import CameraParams, ContextedLandmark
from landmark2pixel_coord import landmark_to_pixel_coord
from distortion import undistort_pixel_coord
from threading_shared_numpy_array import SharedNumpyArray
from triangulation import triangulate_lmcs
from projection import distorted_project
from finalizable_thread_queue import EmptyFinalized, FinalizableQueue


mp_hands = mp.solutions.hands
num_landmarks = 21  # MediaPipe Hands has 21 landmarks


# TODO: maybe mv directly to coupling_loop and do `frames = [cap.read() for cap in caps]`
def cap_reading(
        idx: int,
        stop_event: multiprocessing.synchronize.Event,
        my_last_frame: SharedNumpyArray,
        cam_param: CameraParams,
    ):
    # Initialize video capture
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera {idx}", file=sys.stderr)
        sys.exit(1)

    # Set resolution and fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_param.size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_param.size[1])
    cap.set(cv2.CAP_PROP_FPS, cam_param.fps)

    # Try disabling autofocus
    autofocus_supported = cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1
    if autofocus_supported:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Set manual focus value
    focus_value = cam_param.focus
    focus_supported = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
    if not focus_supported:
        print(
            f"Camera {idx} does not support manual focus! (or invalid focus value)",
            file=sys.stderr,
        )
        sys.exit(1)

    # FPS tracking variables
    fps_counter = 0
    fps_display_time = time.time()

    while True:
        if stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read from camera {idx}", file=sys.stderr)
            break

        my_last_frame.set(frame)

        # Simple FPS measure
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_display_time >= 1.0:
            print(f"Cam {idx} fps:", fps_counter)
            fps_display_time = current_time
            fps_counter = 0

    cap.release()
    print(f"Camera {idx} finished.")


def coupling_loop(
        stop_event: multiprocessing.synchronize.Event,
        last_frame: List[SharedNumpyArray],
        coupled_frames_queue: FinalizableQueue,
    ):
    target_frame_interval = 1 / 60.0  # ~60 FPS

    # Wait until at least one frame is available from all cameras
    while True:
        if all(a_last_frame.get() is not None for a_last_frame in last_frame):
            break
        time.sleep(0.1)

    fps_counter = 0
    fps_display_time = time.time()
    index = 0

    while True:
        start_time = time.time()

        if stop_event.is_set():
            break

        # Print send FPS
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_display_time >= 1.0:
            print("Send fps:", fps_counter)
            print("Coupled frames queue size:", coupled_frames_queue.qsize(), "/", index)
            fps_display_time = current_time
            fps_counter = 0

        frames = []
        for frame in last_frame:
            frames.append(frame.get())

        # Send coupled frames
        coupled_frames_queue.put((index, frames))
        index += 1

        # Rate-limit to ~60 FPS
        elapsed_time = time.time() - start_time
        sleep_time = max(0, target_frame_interval - elapsed_time)
        time.sleep(sleep_time)
    
    coupled_frames_queue.finalize()
    print("Coupling loop finished.")


def processing_loop(
        scale: float,
        cameras_params: List[CameraParams],
        coupled_frames_queue: FinalizableQueue,
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
        frames: List[cv2.typing.MatLike] = elem[1]

        # Find landmarks
        landmarks: List[Any] = []
        for processor, frame in zip(processors, frames):
            # Convert to RGB and process
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = processor.process(frame_rgb)

            # Convert MediaPipe landmarks to plain Python list
            if res.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    res.multi_hand_landmarks,
                    res.multi_handedness
                ):
                    if handedness.classification[0].label == "Right":
                        landmarks.append(hand_landmarks.landmark)
                        break
        
        povs_with_landmarks = [i for i, landmarks in enumerate(landmarks) if landmarks is not None]
        if len(povs_with_landmarks) >= 2:
            # Triangulate points across the cameras
            chosen_cams = []
            points_3d = []
            for lm_id in range(num_landmarks):
                lmcs = []
                for pov_i in povs_with_landmarks:
                    pov_params = cameras_params[pov_i]
                    pixel_pt = landmark_to_pixel_coord(
                        frames[pov_i].shape, landmarks[pov_i][lm_id]
                    )
                    undistorted_lm = undistort_pixel_coord(
                        pixel_pt,
                        pov_params.intrinsic.mtx,
                        pov_params.intrinsic.dist_coeffs,
                    )
                    lmcs.append(
                        ContextedLandmark(
                            cam_idx=pov_i,
                            P=pov_params.P,
                            lm=undistorted_lm,
                        )
                    )

                chosen, point_3d = triangulate_lmcs(lmcs)
                assert point_3d is not None

                chosen_cams.append(chosen)
                points_3d.append(point_3d)

            # Draw on frames
            for i, (frame, params) in enumerate(zip(frames, cameras_params)):
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
                    # Clip to image size
                    x = max(min(int(x), frame.shape[1] - 1), 0)
                    y = max(min(int(y), frame.shape[0] - 1), 0)
                    reprojected_lms.append((x, y))

                # Draw connections first
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    if (
                        start_idx in reprojected_lms
                        and end_idx in reprojected_lms
                    ):
                        start_pt = reprojected_lms[start_idx]
                        end_pt = reprojected_lms[end_idx]
                        cv2.line(
                            frame,
                            start_pt,
                            end_pt,
                            color=(255, 255, 255),
                            thickness=2,
                        )

                # Draw landmarks
                for lm_id, point_3d in enumerate(points_3d):
                    x, y = reprojected_lms[lm_id]
                    if i in chosen_cams[lm_id]:
                        color = (0, 255, 0)  # Chosen camera
                    else:
                        color = (255, 0, 0)  # Others
                    cv2.circle(frame, (x, y), radius=5, color=color, thickness=-1)

        # Resize before display
        for i, frame in enumerate(frames):
            frame_height, frame_width = frame.shape[:2]
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            resized_frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            out_queues[i].put((index, resized_frame))
        
        coupled_frames_queue.task_done()
    
    for processor in processors:
        processor.close()
    
    print("A processing loop is finished.")


def ordering_loop(
        in_queue: FinalizableQueue,
        out_queue: FinalizableQueue,
    ):
    expecting = 0
    unordered: Dict[int, Any] = {}
    while True:
        try:
            elem = in_queue.get()
        except EmptyFinalized:
            break

        index: int = elem[0]
        data: Any = elem[1]

        if expecting == index:
            out_queue.put(data)
            while True:
                expecting += 1
                data = unordered.get(expecting, None)
                if data is None:
                    break
                else:
                    del unordered[expecting]
                    out_queue.put(data)
        else:
            unordered[index] = data
        
        in_queue.task_done()
    
    out_queue.finalize()
    print("A ordering loop finished.")


def display_loop(
        idx: int,
        stop_event: multiprocessing.synchronize.Event,
        frame_queue: FinalizableQueue
    ):
    cv2.namedWindow(f"Camera_{idx}", cv2.WINDOW_AUTOSIZE)

    fps_counter = 0
    fps_display_time = time.time()

    while True:
        try:
            frame = frame_queue.get()
        except EmptyFinalized:
            break

        cv2.imshow(f"Camera_{idx}", frame)

        # Print FPS
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_display_time >= 1.0:
            print(f"Display {idx} fps:", fps_counter)
            fps_display_time = current_time
            fps_counter = 0
        
        # Maybe stop
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            # Stop capturing loop
            stop_event.set()

        frame_queue.task_done()
    
    print(f"Display {idx} loop finished.")


def main():
    parser = argparse.ArgumentParser(
        description="3D Hand Reconstruction using MediaPipe and Multiple Cameras"
    )
    parser.add_argument(
        "--dfile",
        type=str,
        default="cameras.def.json5",
        help="Path to the cameras declarations file",
    )
    parser.add_argument(
        "--cfile",
        type=str,
        default="cameras.calib.json5",
        help="Path to the cameras calibration file",
    )
    parser.add_argument(
        "--window_scale",
        type=float,
        default=0.7,
        help="Scale of a window",
    )
    parser.add_argument(
        "--division",
        type=int,
        default=12,
        help="Number of the hand tracking worker pool per camera",
    )
    args = parser.parse_args()
    window_scale = args.window_scale
    division = args.division

    # Load camera parameters
    cameras_params = load_cameras_parameters(args.dfile, args.cfile)
    if len(cameras_params) < 2:
        print("Need at least two cameras with calibration data.")
        sys.exit(1)
    
    cameras_ids = list(cameras_params.keys())

    # Shared
    cams_stop_event = threading.Event()
    last_frame: List[SharedNumpyArray] = [
        SharedNumpyArray()
        for _ in cameras_ids
    ]

    # Capture cameras
    cap_processes: List[threading.Thread] = [
        threading.Thread(
            target=cap_reading,
            args=(idx, cams_stop_event, my_last_frame, cam_param),
            daemon=True,
        ) for my_last_frame, (idx, cam_param) in zip(last_frame, cameras_params.items())
    ]
    for process in cap_processes:
        process.start()
    
    # Processing workers
    coupled_frames_queue = FinalizableQueue(300)
    processed_queues = [FinalizableQueue(300) for _ in cameras_ids]
    processing_loops_pool = [
        threading.Thread(
            target=processing_loop,
            args=(window_scale, list(cameras_params.values()), coupled_frames_queue, processed_queues),
            daemon=True,
        ) for _ in range(division)
    ]
    for process in processing_loops_pool:
        process.start()

    # Sort processing workers output
    ordered_processed_queues = [FinalizableQueue(300) for _ in cameras_ids]
    ordering_loops = [
        threading.Thread(
            target=ordering_loop,
            args=(in_queue, out_queue),
            daemon=True,
        ) for in_queue, out_queue in zip(processed_queues, ordered_processed_queues)
    ]
    for process in ordering_loops:
        process.start()
    
    # Displaying loops
    display_loops = [
        threading.Thread(
            target=display_loop,
            args=(idx, cams_stop_event, frame_queue),
            daemon=True,
        ) for idx, frame_queue in zip(cameras_ids, ordered_processed_queues)
    ]
    for process in display_loops:
        process.start()

    # Run the feeding loop in the foreground; it exits when user presses 'q'
    coupling_worker = threading.Thread(
        target=coupling_loop,
        args=(cams_stop_event, last_frame, coupled_frames_queue),
        daemon=True,
    )
    coupling_worker.start()

    # Wait for a stop signal
    cams_stop_event.wait()

    # Free resources
    print("Freeing resources...")
    coupling_worker.join()
    
    for process in cap_processes:
        process.join()
    
    print("Waiting for lag to process...")
    coupling_worker.join()

    for process in processing_loops_pool:
        process.join()

    for queue in processed_queues:
        queue.finalize()

    for process in ordering_loops:
        process.join()

    for process in display_loops:
        process.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
