import multiprocessing
import multiprocessing.synchronize
import cv2

from finalizable_queue import EmptyFinalized, FinalizableQueue
from fps_counter import FPSCounter
from draw_utils import draw_left_top


def display_loop(
        idx: int,
        stop_event: multiprocessing.synchronize.Event,
        frame_queue: FinalizableQueue,
    ):
    cv2.namedWindow(f"Camera_{idx}", cv2.WINDOW_AUTOSIZE)

    fps_counter = FPSCounter()

    while True:
        try:
            frame = frame_queue.get()
        except EmptyFinalized:
            break

        # Draw FPS text on the frame
        fps_counter.count()
        draw_left_top(1, f"Display FPS: {fps_counter.get_fps()}", frame)

        # Update the frame
        cv2.imshow(f"Camera_{idx}", frame)
        
        # Maybe stop
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            # Stop capturing loop
            stop_event.set()

        frame_queue.task_done()
    
    print(f"Display {idx} loop finished.")