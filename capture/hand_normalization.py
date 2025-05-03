from typing import List
import numpy as np
from .linal_utils import rotation_matrix_from_vectors

# Bone connections
BONE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),  # Thumb
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 7),  # Index
    (0, 8),
    (8, 9),
    (9, 10),
    (10, 11),  # Middle
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),  # Ring
    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),  # Pinky
]


def normalize_hand(hand_3d_points: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize hand landmark positions using NumPy:
      1. Translate wrist to origin
      2. Align middle finger MCP (idx 8) to +Y axis
      3. Scale middle finger bone to length 1
      4. Rotate around Y axis so index MCP (idx 4) lies in +Z half-plane
    Input: list of 20 (3,) numpy arrays
    Output: list of 20 (3,) numpy arrays
    """
    hand = [pt.copy() for pt in hand_3d_points]

    # 1. Translate wrist (0) to origin
    T = -hand[0]
    hand = [pt + T for pt in hand]

    # 2. Rotate so middle MCP (8) aligns with +Y
    R1 = rotation_matrix_from_vectors(hand[8], np.array([0, 1, 0]))
    hand = [R1 @ pt for pt in hand]

    # 3. Scale so middle MCP is at y=1
    y_len = hand[8][1] + 1e-6
    hand = [pt / y_len for pt in hand]

    # 4. Rotate around Y so index MCP (5) lies in +Z half-plane
    v = hand[4]
    xz = np.array([v[0], v[2]])
    norm = np.linalg.norm(xz)
    if norm >= 1e-6:
        sinA = xz[0] / norm
        cosA = xz[1] / norm
        R2 = np.array([[cosA, 0.0, -sinA], [0.0, 1.0, 0.0], [sinA, 0.0, cosA]])
        hand = [R2 @ pt for pt in hand]

    return hand
