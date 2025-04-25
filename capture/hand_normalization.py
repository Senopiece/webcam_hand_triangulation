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

# Desired bone lengths
BONE_LENGTHS = {
    (0, 1): 7.0,
    (1, 2): 3.5,
    (2, 3): 2.5,  # Thumb
    (0, 4): 9.0,
    (4, 5): 4.0,
    (5, 6): 2.5,
    (6, 7): 2.0,  # Index
    (0, 8): 9.0,
    (8, 9): 5.0,
    (9, 10): 3.0,
    (10, 11): 2.0,  # Middle
    (0, 12): 8.5,
    (12, 13): 5.0,
    (13, 14): 3.0,
    (14, 15): 2.0,  # Ring
    (0, 16): 8.0,
    (16, 17): 4.0,
    (17, 18): 2.5,
    (18, 19): 2.0,  # Pinky
}


def fix_hand_landmarks_anatomy(joints: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize bone lengths of hand landmarks (non-batched, NumPy version) so that it becomes more anatomically correct.
    Returns normalized hand joints

    Input:
        joints: (20, 3) array of hand joint coordinates
    Output:
        (20, 3) array with normalized bone lengths
    """
    fixed = normalize_hand(joints)

    for p, c in BONE_LENGTHS:
        vec = joints[c] - joints[p]  # (3,)
        length = np.linalg.norm(vec)
        if length > 0:
            direction = vec / length
        else:
            direction = np.zeros(3)
        target_length = BONE_LENGTHS[(p, c)]
        fixed[c] = fixed[p] + direction * target_length

    return normalize_hand(fixed)


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
