import cv2
import mediapipe as mp

from models import Landmark
from workers.abstract import SyncWorker, SyncWorkerDelegate
from workers.multiprocessed import MultiprocessedAsyncWorker, MultiprocessedHands
from workers.threaded import ThreadedAsyncWorker

mp_hands = mp.solutions.hands

def spawn_buildin_hand_processor() -> SyncWorker:
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
        
        return res_hand_landmarks

    return SyncWorkerDelegate(send, hands.close)


class AsyncHandsThreadedBuildinSolution(ThreadedAsyncWorker):
    def __init__(self):
        super().__init__(spawn_buildin_hand_processor)

class AsyncHandsMultiprocessedBuildinSolution(MultiprocessedAsyncWorker):
    def __init__(self):
        super().__init__(spawn_buildin_hand_processor)

class SyncHandsMultiprocessedBuildinSolution(MultiprocessedHands):
    def __init__(self):
        super().__init__(spawn_buildin_hand_processor)