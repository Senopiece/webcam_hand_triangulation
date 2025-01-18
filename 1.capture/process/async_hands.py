from abc import ABC, abstractmethod
import asyncio
import threading
from typing import List
import cv2
import mediapipe as mp


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
    def __init__(self, send, dispose):
        # NOTE: Assumed for .send to be called in the same loop the object is created

        self._send = send
        self._dispose = dispose
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
        while True:
            # Wait for input data
            self.in_data_event.wait()

            if self.disposed:
                self._dispose()
                break

            # Process the frame
            self.landmarks = self._send(self.frame)

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
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9,
        )

        def send(frame):
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            res_hand_landmarks = None
            if res.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    res.multi_hand_landmarks, res.multi_handedness
                ):
                    if (
                        handedness.classification[0].label == "Right"
                    ):  # actually left lol
                        res_hand_landmarks = hand_landmarks.landmark
                        break
            
            return (res_hand_landmarks, frame)

        super().__init__(send, hands.close)


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
