import numpy as np
from linal_utils import is_point_in_smoothed_polygon

_fingers_ids = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]


def _fingers(landmarks):
    return [None if i not in _fingers_ids else lm for i, lm in enumerate(landmarks)]


def _palm(landmarks):
    return [None if i in _fingers_ids else lm for i, lm in enumerate(landmarks)]


def _back(landmarks):
    """
    Set finger landmarks to None if they are outside the smoothed polygon defined by palm landmarks.
    :param landmarks: List of (x, y) tuples or None, representing landmarks.
    :return: Modified landmarks list with some points set to None.
    """
    if any(lm is None for lm in landmarks):
        return [None for _ in landmarks]

    # Extract palm landmarks (polygon vertices)
    palm_lms = np.array([[lm.x, lm.y] for i, lm in enumerate(landmarks) if i not in _fingers_ids])

    SMOOTHING_DISTANCE = 0.06
    updated_landmarks = landmarks[:]

    # Finger landmark may be thrown individually
    for i, lm in enumerate(landmarks):
        # Skip palm landmarks
        if i not in _fingers_ids:
            continue

        # Check if the landmark is in the smoothed polygon - e.g. covered by palm
        if is_point_in_smoothed_polygon(np.array([lm.x, lm.y]), palm_lms, SMOOTHING_DISTANCE):
            updated_landmarks[i] = None

    return updated_landmarks


landmark_transforms = {
    "full": lambda x: x,
    "fingers": _fingers,
    "palm": _palm,
    "back": _back,
    # TODO: front landmarks
    # TODO: maybe some variations with weights on landmarks
}