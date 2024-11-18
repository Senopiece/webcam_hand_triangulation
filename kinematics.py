import numpy as np


def bone_mp_point_corresondence():
    idx = 0
    for label in ["Index", "Middle", "Ring", "Pinky"]:
        for i in range(4):
            idx += 1
            yield f"{label}.{i}", 0 if i == 0 else idx - 1, idx

    yield "Thumb.0", 0, 2
    yield "Thumb.1", 2, 3
    yield "Thumb.2", 3, 4


def points_3d_to_bones_rotations(points_3d: np.ndarray):
    assert points_3d.shape == (21, 3)

    res = {}

    for label, p1i, p2i in bone_mp_point_corresondence():
        p1 = points_3d[p1i]
        p2 = points_3d[p2i]

        res[label] = calculate_euler_angles(p2 - p1)

    return res


def calculate_euler_angles(target_point):
    # Define the initial vector
    initial_vector = np.array([0, 1, 0])

    # Normalize both vectors
    initial_vector = initial_vector / np.linalg.norm(initial_vector)
    target_point = target_point / np.linalg.norm(target_point)

    # Compute the rotation axis (cross product)
    rotation_axis = np.cross(initial_vector, target_point)

    # Compute the angle between the vectors (dot product)
    angle = np.arccos(np.clip(np.dot(initial_vector, target_point), -1.0, 1.0))

    # Handle the case where the vectors are aligned (no rotation needed)
    if np.linalg.norm(rotation_axis) < 1e-6:
        return np.array([0.0, 0.0, 0.0])  # No rotation

    # Normalize the rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Compute the rotation matrix using Rodrigues' rotation formula
    K = np.array(
        [
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0],
        ]
    )
    I = np.eye(3)
    rotation_matrix = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # Extract Euler angles (XYZ sequence) from the rotation matrix
    pitch = np.arcsin(-rotation_matrix[1, 2])  # sin(pitch) = -m[1,2]
    roll = np.arctan2(
        rotation_matrix[0, 2], rotation_matrix[2, 2]
    )  # roll = atan2(m[0,2], m[2,2])
    yaw = np.arctan2(
        rotation_matrix[1, 0], rotation_matrix[1, 1]
    )  # yaw = atan2(m[1,0], m[1,1])

    return np.array([pitch, roll, yaw])


if __name__ == "__main__":
    p = np.array([-1, 1, 1])
    q = calculate_euler_angles(p)
    print(list(q))
