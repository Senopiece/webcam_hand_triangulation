import numpy as np
from linal_utils import rotation_matrix_from_vectors


def normalize_hand(hand_3d_points):
    # Translate to origin
    T = -hand_3d_points[0]
    hand_3d_points = [pt + T for pt in hand_3d_points]

    # Rotate the virtual main bone (WHIRST - MIDDLE_FINGER_MCP) to align with the y-axis
    R = rotation_matrix_from_vectors(hand_3d_points[9], np.array([0, 1, 0]))
    hand_3d_points = [R @ pt for pt in hand_3d_points]

    # Rotate the whole hand around the y-axis so that the virtual secondary bone (WHIRST - INDEX_FINGER_MCP) is lying on the yz-plane with z > 0
    v = hand_3d_points[5]
    v = np.array([v[0], v[2]])
    v = v / np.linalg.norm(v)
    sinA = v[0]
    cosA = v[1]
    R = np.array([[cosA, 0, -sinA], [0, 1, 0], [sinA, 0, cosA]])
    hand_3d_points = [R @ pt for pt in hand_3d_points]

    return hand_3d_points
