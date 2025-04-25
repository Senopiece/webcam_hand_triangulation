import numpy as np
from typing import List, Tuple

DEFAULT_MORPHOLOGY = np.array(
    [
        # thumb
        [
            # bones lengths
            7.5329e-01,
            3.7025e-01,
            2.5811e-01,
            # start point (x, y, z)
            7.3117e-03,
            1.1166e-02,
            4.2211e-04,
            # starting rotation axis angle (radians)
            -3.5689e00,
        ],
        # index
        [
            # bones lengths
            4.3777e-01,
            2.7459e-01,
            2.2351e-01,
            # starting point (x, y, z)
            -1.6304e-03,
            9.7066e-01,
            2.2563e-01,
            # starting rotation axis angle (radians)
            1.5067e00,
        ],
        # middle
        [
            # bones lengths
            5.3909e-01,
            3.2731e-01,
            2.2931e-01,
            # starting point (x, y, z)
            -9.7050e-04,
            9.9966e-01,
            -1.7299e-02,
            # starting rotation axis angle (radians)
            -1.1042e01,
        ],
        # ring
        [
            # bones lengths
            5.3695e-01,
            3.2312e-01,
            2.3646e-01,
            # starting point (x, y, z)
            -3.7696e-02,
            9.1871e-01,
            -2.2384e-01,
            # starting rotation axis angle (radians)
            1.4743e00,
        ],
        # pinky
        [
            # bones lengths
            4.6647e-01,
            2.6924e-01,
            2.3053e-01,
            # starting point (x, y, z)
            -9.2299e-02,
            7.6455e-01,
            -3.9918e-01,
            # starting rotation axis angle (radians)
            1.3717e00,
        ],
    ],
    dtype=np.float32,
)


def rot(
    v: np.ndarray, p: np.ndarray, alpha: float, beta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate vector `v` and perpendicular vector `p` using angles alpha and beta.
    """
    sinA, cosA = np.sin(alpha), np.cos(alpha)
    sinB, cosB = np.sin(beta), np.cos(beta)

    q = np.cross(v, p)

    v_hat = q * sinA + v * cosA * cosB + p * sinB
    p_hat = p * cosB - v * sinB

    return v_hat, p_hat


def hand_landmarks_by_angles(
    angles: np.ndarray,
    morphology: np.ndarray = DEFAULT_MORPHOLOGY,  # shape (20,)  # shape (5, 7)
) -> List[np.ndarray]:
    """
    Compute hand landmarks from angles and morphology.
    Returns a list of 20 (3,) np.ndarrays
    """
    landmarks = [np.zeros(3, dtype=np.float32) for _ in range(20)]

    angles = angles.reshape(5, 4)

    for i in range(1, 5):
        idx = 4 * i
        landmarks[idx] = morphology[i, 3:6].copy()

    V = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    P = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    for finger_index in range(5):
        morph = morphology[finger_index]
        local_angles = angles[finger_index]

        base_idx = finger_index * 4
        bone_lengths = morph[0:3]
        joint = morph[3:6].copy()
        gamma = morph[6]

        a0, b0, a2, a3 = local_angles
        chain_angles = [
            (a0, b0),
            (a2, 0.0),
            (a3, 0.0),
        ]

        v = joint / (np.linalg.norm(joint) + 1e-8)

        sinB = np.dot(v, P)
        cosB = np.sqrt(1.0 - sinB**2)
        p = P * cosB - V * sinB

        # Rotate p around v by gamma
        sinG, cosG = np.sin(gamma), np.cos(gamma)
        p = p * cosG + np.cross(p, v) * sinG

        for j in range(3):
            l = bone_lengths[j]
            alpha, beta = chain_angles[j]
            v, p = rot(v, p, alpha, beta)
            joint = joint + v * l
            landmarks[base_idx + j + 1] = joint.copy()

    return landmarks


def irot(
    v: np.ndarray, p: np.ndarray, v_hat: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    Inverse rotation from v, p, v_hat to alpha, beta and p_hat
    """
    q = np.cross(v, p)

    sinA = np.clip(np.dot(v_hat, q), -1 + 1e-6, 1 - 1e-6)
    sinB = np.clip(np.dot(v_hat, p), -1 + 1e-6, 1 - 1e-6)

    alpha = np.arcsin(sinA)
    beta = np.arcsin(sinB)

    cosB = np.sqrt(max(1.0 - sinB**2, 1e-7))
    p_hat = p * cosB - v * sinB

    return alpha, beta, p_hat


def inverse_hand_angles_by_landmarks(
    landmarks: List[np.ndarray],  # list of 20 (3,) arrays
    morphology: np.ndarray = DEFAULT_MORPHOLOGY,  # shape (5, 7)
) -> np.ndarray:
    """
    Returns a (20,) array of angles
    """
    angles = np.zeros((5, 4), dtype=np.float32)

    V = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    P = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    for i in range(5):
        morph = morphology[i]
        joint = morph[3:6].copy()
        gamma = morph[6]

        v = joint / (np.linalg.norm(joint) + 1e-8)

        sinB = np.dot(v, P)
        cosB = np.sqrt(1.0 - sinB**2)
        p = P * cosB - V * sinB

        sinG, cosG = np.sin(gamma), np.cos(gamma)
        p = p * cosG + np.cross(p, v) * sinG

        idx = [4 * i + j for j in range(4)]

        for j in range(3):
            delta = landmarks[idx[j + 1]] - landmarks[idx[j]]
            target = delta / (np.linalg.norm(delta) + 1e-8)

            alpha, beta, p = irot(v, p, target)
            v = target

            if j == 0:
                angles[i, 0] = alpha
                angles[i, 1] = beta
            else:
                angles[i, j + 1] = alpha

    return angles.flatten()
