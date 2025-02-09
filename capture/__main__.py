import os
import sys

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

import multiprocessing
import multiprocessing.synchronize
import threading
import numpy as np
from typing import Dict, List, Tuple
import argparse

from cam_conf import load_cameras_parameters
from wrapped import Wrapped
from models import CameraParams
from finalizable_queue import ThreadFinalizableQueue, ProcessFinalizableQueue
from cap_reading_loop import cap_reading
from coupling_loop import coupling_loop
from processing_loop import processing_loop
from ordering_loop import ordering_loop
from hand_3d_visualization_loop import hand_3d_visualization_loop
from display_loop import display_loop
from landmark_transforms import landmark_transforms


def main(
    cameras_params: Dict[int, CameraParams],
    couple_fps: int,
    desired_window_size: Tuple[int, int],
    division: int,
    draw_origin_landmarks: bool,
):
    # Check camera parameters
    if len(cameras_params) < 2:
        print("Need at least two cameras with calibration data.")
        return

    cameras_ids = list(cameras_params.keys())

    # Shared
    cams_stop_event = multiprocessing.Event()
    last_frame: List[Wrapped[Tuple[np.ndarray, int] | None]] = [
        Wrapped() for _ in cameras_ids
    ]

    # Capture cameras
    caps: List[threading.Thread] = [
        threading.Thread(
            target=cap_reading,
            args=(
                idx,
                cams_stop_event,
                my_last_frame,
                cam_param,
            ),
            daemon=True,
        )
        for my_last_frame, (idx, cam_param) in zip(last_frame, cameras_params.items())
    ]
    for process in caps:
        process.start()

    # Couple frames
    coupled_frames_queue = ThreadFinalizableQueue()
    coupling_worker = threading.Thread(
        target=coupling_loop,
        args=(
            couple_fps,
            cams_stop_event,
            last_frame,
            coupled_frames_queue,
        ),
        daemon=True,
    )
    coupling_worker.start()

    # Processing workers
    hand_points_queue = ThreadFinalizableQueue()
    processed_queues = [ThreadFinalizableQueue() for _ in cameras_ids]
    processing_loops_pool = [
        threading.Thread(
            target=processing_loop,
            args=(
                [landmark_transforms[cp.track] for cp in cameras_params.values()],
                draw_origin_landmarks,
                desired_window_size,
                list(cameras_params.values()),
                coupled_frames_queue,
                hand_points_queue,
                processed_queues,
            ),
            daemon=True,
        )
        for _ in range(division)
    ]
    for process in processing_loops_pool:
        process.start()

    # Sort hand points
    ordered_hand_points_queue = ProcessFinalizableQueue()
    hand_points_sorter = threading.Thread(
        target=ordering_loop,
        args=(
            hand_points_queue,
            ordered_hand_points_queue,
        ),
        daemon=True,
    )
    hand_points_sorter.start()

    # Visualize 3d hand
    hand_3d_visualizer = multiprocessing.Process(
        target=hand_3d_visualization_loop,
        args=(
            desired_window_size,
            cams_stop_event,
            ordered_hand_points_queue,
        ),
        daemon=True,
    )
    hand_3d_visualizer.start()

    # Sort processing workers output
    ordered_processed_queues = [ProcessFinalizableQueue() for _ in cameras_ids]
    display_ordering_loops = [
        threading.Thread(
            target=ordering_loop,
            args=(
                in_queue,
                out_queue,
            ),
            daemon=True,
        )
        for in_queue, out_queue in zip(processed_queues, ordered_processed_queues)
    ]
    for process in display_ordering_loops:
        process.start()

    # Displaying loops
    display_loops = [
        multiprocessing.Process(
            target=display_loop,
            args=(
                idx,
                cams_stop_event,
                frame_queue,
            ),
            daemon=True,
        )
        for idx, frame_queue in zip(cameras_ids, ordered_processed_queues)
    ]
    for process in display_loops:
        process.start()

    # Wait for a stop signal
    cams_stop_event.wait()

    # Free resources
    print("Freeing resources...")
    coupling_worker.join()

    for worker in caps:
        worker.join()

    print("Waiting for lag to process...")
    coupling_worker.join()

    for worker in processing_loops_pool:
        worker.join()

    hand_points_queue.finalize()
    for queue in processed_queues:
        queue.finalize()

    hand_points_sorter.join()
    for worker in display_ordering_loops:
        worker.join()

    hand_3d_visualizer.join()
    for worker in display_loops:
        worker.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Hand Reconstruction using MediaPipe and Multiple Cameras"
    )
    parser.add_argument(
        "--cfile",
        type=str,
        default="cameras.calib.json5",
        help="Path to the cameras calibration file",
    )
    parser.add_argument(
        "--window_size",
        type=str,
        default="448x336",
        help="Size of a preview window",
    )
    parser.add_argument(
        "--division",
        type=int,
        default=8,
        help="Number of the hand tracking worker pool per camera",
    )
    parser.add_argument(
        "--couple_fps",
        type=int,
        default=30,
        help="Rate at which frames from cameras will be coupled",
    )
    parser.add_argument(
        "-ol", "--origin_landmarks", help="Draw origin landmarks", action="store_true"
    )
    args = parser.parse_args()

    desired_window_size = tuple(map(int, args.window_size.split("x")))
    if len(desired_window_size) != 2:
        print("Error: window_size must be a AxB value", file=sys.stderr)
        sys.exit(1)

    main(
        cameras_params=load_cameras_parameters(args.cfile),
        couple_fps=args.couple_fps,
        desired_window_size=desired_window_size,
        division=args.division,
        draw_origin_landmarks=args.origin_landmarks,
    )
