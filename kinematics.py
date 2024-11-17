import numpy as np


def decart2polar(p: np.ndarray):
    assert p.shape == (3,), "Input point must have shape (3,)."

    direction = p.copy()

    # Compute the radial distance (magnitude of the direction vector)
    r = np.linalg.norm(direction)
    if r == 0:
        raise ValueError("Points p1 and p2 are identical; direction is undefined.")

    # Compute azimuth (angle in the xy-plane from the x-axis)
    azimuth = np.arctan2(direction[1], direction[0])

    # Compute elevation (angle from the xy-plane)
    elevation = np.arcsin(direction[2] / r)

    return r, azimuth, elevation


def polar2decart(r, azimuth, elevation):
    return np.asarray(
        [
            r * np.cos(elevation) * np.cos(azimuth),
            r * np.cos(elevation) * np.sin(azimuth),
            r * np.sin(elevation),
        ]
    )


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

        _, azimuth, elevation = decart2polar(p2 - p1)
        res[label] = {"azimuth": azimuth, "elevation": elevation}

    return res


if __name__ == "__main__":
    p = np.array([9, 8, 3])
    r, azimuth, elevation = decart2polar(p)
    print(polar2decart(r, azimuth, elevation), p)
