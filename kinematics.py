import numpy as np


def direction2euler(p1: np.ndarray, p2: np.ndarray):
    assert p1.shape == (3,) and p2.shape == (3,), "Input points must have shape (3,)."

    direction = p2 - p1
    direction /= np.linalg.norm(direction)
    yaw = np.arctan2(direction[1], direction[0])
    pitch = np.arcsin(direction[2])

    return yaw, pitch


def bone_mp_point_corresondence():
    idx = 0
    for label in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        for i in range(4):
            idx += 1
            yield f"{label}.{i}", 0 if i == 0 else idx - 1, idx


def mp2bones(points_3d: np.ndarray):
    assert points_3d.shape == (21, 3)

    res = {}

    for label, p1i, p2i in bone_mp_point_corresondence():
        p1 = points_3d[p1i]
        p2 = points_3d[p2i]

        yaw, pitch = direction2euler(p1, p2)
        res[label] = {"x": 0, "y": pitch, "z": yaw}

    return res
