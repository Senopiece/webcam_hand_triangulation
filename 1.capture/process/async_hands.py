from abc import ABC, abstractmethod
import multiprocessing
import asyncio
import threading
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
from pydantic import BaseModel

class Landmark(BaseModel):
    x: float
    y: float

def spawn_buildin_hand_processor():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9,
    )

    def send(frame):
        # Convert to RGB and process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(frame_rgb)

        # Convert MediaPipe landmarks to plain Python list
        res_hand_landmarks = None
        if res.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                res.multi_hand_landmarks, 
                res.multi_handedness
            ):
                if handedness.classification[0].label == "Right":
                    # Convert all landmarks to a list of (x, y)
                    res_hand_landmarks = [
                        Landmark(x=lm.x, y=lm.y) for lm in hand_landmarks.landmark
                    ]
                    break
        
        return (res_hand_landmarks, frame)

    return send, hands.close

class AsyncHands(ABC):
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.dispose()

    @abstractmethod
    async def dispose(self):
        pass

    @abstractmethod
    async def send(self, frame):
        pass

    @abstractmethod
    def is_busy(self):
        pass

    @abstractmethod
    async def wait_free(self):
        pass


class _ThreadedAsyncHands(AsyncHands):
    def __init__(self, spawn):
        # NOTE: Assumed for .send to be called in the same loop the object is created

        self._spawn = spawn
        self.loop = asyncio.get_running_loop()
        self.in_data_event = threading.Event()
        self.out_data_event = asyncio.Event()
        self.out_data_event.set()
        self.disposed = False
        self.frame = None
        self.landmarks = None

        # Spawn a worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    async def dispose(self):
        await self.out_data_event.wait()
        self.disposed = True
        self.in_data_event.set()

    def is_busy(self):
        return not self.out_data_event.is_set()

    async def wait_free(self):
        await self.out_data_event.wait()

    def _worker(self):
        send, dispose = self._spawn()

        while True:
            # Wait for input data
            self.in_data_event.wait()

            if self.disposed:
                dispose()
                break

            # Process the frame
            self.landmarks = send(self.frame)

            self.in_data_event.clear()  # drop anything that came while we were processing

            self.loop.call_soon_threadsafe(self.out_data_event.set)

    async def send(self, frame):
        # NOTE: Unstable behavior if calling next send without avaiting the previous one
        # Meaning the next frame will surely be throwed, but the result of the next frame send
        # may be from previous one as well from the next sucesefully processed frame

        if self.out_data_event.is_set():
            self.out_data_event.clear()
            self.frame = frame
            self.in_data_event.set()

        await self.out_data_event.wait()
        return self.landmarks

class AsyncHandsThreadedBuildinSolution(_ThreadedAsyncHands):
    def __init__(self):
        super().__init__(spawn_buildin_hand_processor)

class _MultiprocessingAsyncHands(AsyncHands):
    def __init__(self, spawn):
        self.loop = asyncio.get_running_loop()
        
        # Two queues:
        #   - input_queue: main process => child process
        #   - output_queue: child process => main process
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        # Start the dedicated worker process
        self.worker_process = multiprocessing.Process(
            target=self._worker,
            args=(self.input_queue, self.output_queue, spawn),
            daemon=True,
        )
        self.worker_process.start()

        # Concurrency-related flags
        self._disposed = False
        self._busy_lock = asyncio.Lock()  # used to manage single concurrency
        self._current_result_future: Optional[asyncio.Future] = None

        # Unique ID for each request so we know which result corresponds
        self._next_request_id = 0
        self._futures_by_id = {}

        # Start a background thread to read results from output_queue
        self._result_thread = threading.Thread(
            target=self._result_collector, daemon=True
        )
        self._result_thread.start()
    
    @staticmethod
    def _worker(input_queue, output_queue, spawn):
        send, dispose = spawn()

        while True:
            item = input_queue.get()  # blocking read
            if item is None:
                break

            (request_id, frame) = item
            if request_id is None or frame is None:
                break

            res = send(frame)

            output_queue.put((request_id, res))

        dispose()

    def _result_collector(self):
        """
        Runs in a background thread:
          - continuously reads results from `output_queue` (blocking)
          - for each result, finds the correct future in the main loop and sets it
        """
        while True:
            try:
                result = self.output_queue.get()
            except (EOFError, OSError):
                # If the queue is somehow closed, just exit
                break

            if result is None:
                # Sentinel => child is done
                break

            request_id, res = result

            # Switch back to the main thread (event loop) to set the future
            def _set_result_in_main():
                future = self._futures_by_id.pop(request_id, None)
                if future and not future.done():
                    future.set_result(res)

            self.loop.call_soon_threadsafe(_set_result_in_main)

    async def dispose(self):
        """
        Dispose the child process. Must not be reused after disposal.
        """
        if self._disposed:
            return

        # Wait for current operation to finish
        if self._current_result_future is not None:
            try:
                await self._current_result_future
            except:
                pass

        self._disposed = True

        # Send sentinel to worker => it will exit its loop
        self.input_queue.put(None)

        # Join the worker process
        self.worker_process.join(timeout=2.0)

        # Finally, also put sentinel into result_queue (in case the result thread is blocking)
        self.output_queue.put(None)

        # Attempt to close the queues
        self.input_queue.close()
        self.output_queue.close()

    def is_busy(self) -> bool:
        """
        We allow one `send` to be 'in flight' at a time.
        If locked => busy. If unlocked => free.
        """
        return self._busy_lock.locked()

    async def wait_free(self):
        """
        Wait until the concurrency lock is free.
        """
        # If the lock is free, we can immediately proceed. If locked, we wait.
        # Implementation detail: we do an acquire+release trick.
        async with self._busy_lock:
            pass

    async def send(self, frame: np.ndarray) -> Tuple[Optional[list], np.ndarray]:
        """
        Sends a frame to be processed. Returns a tuple: (landmarks, frame).
        - landmarks: either a list of 21 Landmarks or None if no right-hand found
        - frame: the original frame (for convenience).
        """
        if self._disposed:
            raise RuntimeError("Cannot send frame after dispose() was called.")

        # Wait until we're free to send the next frame
        await self._busy_lock.acquire()

        # Create a future that will eventually hold (landmarks, frame)
        future = self.loop.create_future()

        # Register the future using a new request_id
        request_id = self._next_request_id
        self._next_request_id += 1
        self._futures_by_id[request_id] = future

        # Put (request_id, frame) onto the input queue for the worker process
        self.input_queue.put((request_id, frame))

        # Keep track so we can await in dispose if we want
        self._current_result_future = future

        def _release_lock(_):
            # Once the future completes (success or exception),
            # release the concurrency lock
            if self._busy_lock.locked():
                self._busy_lock.release()

        future.add_done_callback(_release_lock)

        # Return the result
        return await future

class AsyncHandsMultiprocessingBuildinSolution(_MultiprocessingAsyncHands):
    def __init__(self):
        super().__init__(spawn_buildin_hand_processor)

class HandTrackersPool:
    def __init__(self, pool: List[AsyncHands]):
        self.pool = pool
        self.results = asyncio.Queue()  # Filled in order of .send calls
        self.idle_workers = asyncio.Queue(len(pool))
        self.last_task = None

        # Add all workers to the queue initially
        for worker in self.pool:
            self.idle_workers.put_nowait(worker)

    async def dispose(self):
        await asyncio.gather(*[worker.dispose() for worker in self.pool])

    async def _send(self, prev_task, worker: AsyncHands, frame):
        # NOTE: will hang if exception rises somewhere there

        res = await worker.send(frame)

        self.idle_workers.put_nowait(worker)

        # Ensure queue is filled in order
        if prev_task is not None:
            await prev_task

        self.results.put_nowait(res)

    async def send(self, frame):
        """
        Waits for an available worker and sends the frame to it.
        NOTE: will return immediately if a worker is available, get the result from the results queue
              otherwise will block for the fist available worker, still get the result from the results queue
        """
        # Wait for a free worker from the queue
        worker = await self.idle_workers.get()
        self.last_task = asyncio.create_task(self._send(self.last_task, worker, frame))
