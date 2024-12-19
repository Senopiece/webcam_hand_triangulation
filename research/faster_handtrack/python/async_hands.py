from abc import ABC, abstractmethod
import asyncio
import threading
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


# NOTE: hand_landmarker.task can be download from https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task


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


class LiveStreamAsyncHands(AsyncHands):
    def __init__(
        self,
        model_asset_path="hand_landmarker.task",
        gpu=False,
    ):
        # NOTE: Assumed for .send to be called in the same loop the object is created

        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_asset_path,
            delegate=(
                mp.tasks.BaseOptions.Delegate.GPU
                if gpu
                else mp.tasks.BaseOptions.Delegate.CPU
            ),
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.9,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.9,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._result_callback,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.out_data_event = asyncio.Event()
        self.out_data_event.set()
        self.loop = asyncio.get_running_loop()
        self.landmarks = None
        self.ts = 0

    async def dispose(self):
        self.detector.close()
        self.detector = None
        await self.out_data_event.wait()

    def _result_callback(self, landmarks, img, ts):
        self.landmarks = landmarks
        self.loop.call_soon_threadsafe(self.out_data_event.set)

    async def send(self, frame):
        # NOTE: Unstable behavior if calling next send without avaiting the previous one
        # Meaning the next frame will surely be throwed, but the result of the next frame send
        # may be from previous one as well from the next sucesefully processed frame

        if self.out_data_event.is_set():
            self.out_data_event.clear()
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            self.detector.detect_async(image, self.ts)
            self.ts += 16

        await self.out_data_event.wait()
        return self.landmarks


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
        self.disposed = True
        self.in_data_event.set()
        await self.out_data_event.wait()
        self.worker_thread.join()
        self.in_data_event = None

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
            return hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        super().__init__(send, hands.close)


class AsyncHandsThreadedVideo(_ThreadedAsyncHands):
    def __init__(
        self,
        model_asset_path="hand_landmarker.task",
        gpu=False,
    ):
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_asset_path,
            delegate=(
                mp.tasks.BaseOptions.Delegate.GPU
                if gpu
                else mp.tasks.BaseOptions.Delegate.CPU
            ),
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.9,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.9,
            running_mode=vision.RunningMode.VIDEO,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        self.ts = 0

        def send(frame):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            res = detector.detect_for_video(image, self.ts)
            self.ts += 16
            return res

        super().__init__(send, detector.close)


class AsyncHandsThreadedImage(_ThreadedAsyncHands):
    def __init__(
        self,
        model_asset_path="hand_landmarker.task",
        gpu=False,
    ):
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_asset_path,
            delegate=(
                mp.tasks.BaseOptions.Delegate.GPU
                if gpu
                else mp.tasks.BaseOptions.Delegate.CPU
            ),
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.9,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.9,
            running_mode=vision.RunningMode.IMAGE,
        )
        detector = vision.HandLandmarker.create_from_options(options)

        def send(frame):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            return detector.detect(image)

        super().__init__(send, detector.close)
