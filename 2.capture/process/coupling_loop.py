import multiprocessing
import multiprocessing.synchronize
import time
from typing import List, Tuple
import cv2
import numpy as np


from wrapped import Wrapped
from fps_counter import FPSCounter
from finalizable_queue import FinalizableQueue

def coupling_loop(
        stop_event: multiprocessing.synchronize.Event,
        last_frame: List[Wrapped[Tuple[np.ndarray, int] | None]],
        coupled_frames_queue: FinalizableQueue,
    ):
    target_frame_interval = 1 / 30.0  # ~30 FPS

    # Wait until at least one frame is available from all cameras
    while True:
        if all(a_last_frame.get() is not None for a_last_frame in last_frame):
            break
        time.sleep(0.1)

    fps_counter = FPSCounter()
    index = 0

    while True:
        start_time = time.time()

        if stop_event.is_set():
            break

        fps_counter.count()

        frames = []
        for frame in last_frame:
            frame, fps = frame.get()
            frames.append((cv2.flip(frame, 1), fps))

        # Send coupled frames
        coupled_frames_queue.put((index, frames, fps_counter.get_fps()))
        index += 1

        # Rate-limit to ~60 FPS
        elapsed_time = time.time() - start_time
        sleep_time = max(0, target_frame_interval - elapsed_time)
        time.sleep(sleep_time)
    
    coupled_frames_queue.finalize()
    print("Coupling loop finished.")