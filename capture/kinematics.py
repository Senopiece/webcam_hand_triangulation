import numpy as np
from typing import List

DEFAULT_MORPHOLOGY = np.array(
    [
        [
            0.3502,
            0.3765,
            0.2691,
            -0.2615,
            0.2654,
            0.3382,
            0.7531,
            0.0717,
            0.9498,
            1.2573,
        ],
        [
            0.4468,
            0.2765,
            0.2192,
            -0.0019,
            0.9693,
            0.2282,
            -0.3828,
            0.3692,
            -0.2930,
            -0.1947,
        ],
        [
            0.5582,
            0.3288,
            0.2200,
            -0.0020,
            0.9970,
            -0.0124,
            -0.5517,
            0.1407,
            -0.2910,
            -0.1613,
        ],
        [
            0.5533,
            0.3325,
            0.2214,
            -0.0453,
            0.9214,
            -0.2073,
            -0.1670,
            0.7382,
            -0.3557,
            -1.0503,
        ],
        [
            0.4466,
            0.2851,
            0.2230,
            -0.1164,
            0.7822,
            -0.3913,
            -0.1512,
            0.5448,
            -0.3952,
            -0.9645,
        ],
    ],
    dtype=np.float32,
)

E = 1e-7
V = np.array([0, 1, 0])
P = np.array([0, 0, 1])


def irot_full(
    v: np.ndarray,
    p: np.ndarray,
    v_hat: np.ndarray,
    eps: float,
    fallback_beta: float,
):
    """
    Inverse rotation function.

    v, p, v_hat: (3,) arrays
    eps: float
    sigma: float
    fallback_beta: float
    Returns: alpha, beta, v_hat, p_hat
    """
    q = np.cross(v, p)

    dot_v = np.dot(v_hat, v)  # cosB*cosA
    dot_p = np.dot(v_hat, p)  # sinB*cosA
    dot_q = np.dot(v_hat, q)  # sinA
    dot_q = np.clip(dot_q, -1 + E, 1 - E)

    alpha = np.arcsin(dot_q)

    cosA = np.sqrt(1 - dot_q**2)

    beta = np.arctan2(dot_p / cosA, dot_v / cosA) if cosA >= eps else fallback_beta

    p_hat = p * np.cos(beta) - v * np.sin(beta)

    return alpha, beta, v_hat, p_hat


def irot_alpha(
    v: np.ndarray,
    p: np.ndarray,
    v_hat: np.ndarray,
):
    """
    Inverse rotation function.

    v, p, v_hat: (3,) arrays
    Returns: alpha, beta, v_hat, p_hat
    """
    q = np.cross(v, p)

    cosA = np.dot(v_hat, v)
    cosA = np.clip(cosA, -1 + E, 1 - E)
    alpha = np.sign(np.dot(v_hat, q)) * np.arccos(cosA)
    beta = 0
    p_hat = p

    return alpha, beta, v_hat, p_hat


def inverse_hand_angles_by_landmarks(
    landmarks: List[np.ndarray],
    morphology: np.ndarray = DEFAULT_MORPHOLOGY,
    eps: float = 0.4,
):
    """
    landmarks: List[np.ndarray] - known hand landmarks in 3D space
    morphology: (5, 10) - hand morphology (not batched)
    eps: float
    sigma: float

    Returns:
    angles: (20,) - recovered joint angles
    """
    angles = np.zeros((5, 4), dtype=landmarks[0].dtype)

    # NOTE: assuming 0, 4, 8, 12, 16 landmarks are existing in the morphology
    # 20 landmarks -> 21 landmarks, restoring thumb base from the morphology
    # so that now assuming 0, 1, 5, 9, 13, 17 are existing in the morphology
    thumb_base = morphology[0][3:6]

    # Insert thumb_base at the correct position
    landmarks = landmarks[:1] + [thumb_base] + landmarks[1:]

    for i, morph in enumerate(morphology):
        base_idx = i * 4 + 1
        alpha, beta, gamma, fallback_beta = morph[6:10]

        v, p = rot(V, P, alpha, beta)

        # Rotate p around v according to parameter gamma
        sinG, cosG = np.sin(gamma), np.cos(gamma)
        p = p * cosG + np.cross(p, v) * sinG

        target = (landmarks[base_idx + 1] - landmarks[base_idx]) / np.linalg.norm(
            landmarks[base_idx + 1] - landmarks[base_idx]
        )
        alpha, beta, v, p = irot_full(v, p, target, eps, fallback_beta)
        angles[i, 0] = alpha
        angles[i, 1] = beta

        for j in range(1, 3):
            target = (
                landmarks[base_idx + j + 1] - landmarks[base_idx + j]
            ) / np.linalg.norm(landmarks[base_idx + j + 1] - landmarks[base_idx + j])
            alpha, beta, v, p = irot_alpha(v, p, target)
            angles[i, j + 1] = alpha

    return angles.flatten()


def rot(
    v: np.ndarray,
    p: np.ndarray,
    alpha: float,
    beta: float,
):
    """
    Rotation function.

    v, p: (3,) arrays representing vector and the first axis of rotation
    alpha, beta: floats representing angles to use for rotation
    """
    sinA = np.sin(alpha)
    cosA = np.cos(alpha)
    sinB = np.sin(beta)
    cosB = np.cos(beta)

    q = np.cross(v, p)

    p_hat = p * cosB - v * sinB
    v_hat = q * sinA + v * cosA * cosB + p * cosA * sinB

    return v_hat, p_hat


def hand_landmarks_by_angles(
    angles: np.ndarray,
    morphology: np.ndarray = DEFAULT_MORPHOLOGY,
):
    """
    angles: (20,)
    morphology: (5, 10) - not batched
    Returns: (20, 3)
    """
    landmarks = np.zeros((21, 3), dtype=angles.dtype)
    angles = angles.reshape(5, 4)

    # Write morphological const points
    for i in range(5):
        landmarks[4 * i + 1, :] = morphology[i, 3:6]

    for finger_index, (morph, local_angles) in enumerate(zip(morphology, angles)):
        base_idx = finger_index * 4 + 1
        bone_lengths = morph[0:3]
        joint = morph[3:6].copy()
        alpha, beta, gamma = morph[6:9]

        a0, b0, a2, a3 = local_angles
        chain_angles = np.array([[a0, b0], [a2, 0], [a3, 0]])

        v, p = rot(V, P, alpha, beta)

        # Rotate p around v according to parameter gamma
        sinG, cosG = np.sin(gamma), np.cos(gamma)
        p = p * cosG + np.cross(p, v) * sinG

        # Iterate over the chain and write the 3d points
        for j in range(3):
            l = bone_lengths[j]
            alpha, beta = chain_angles[j, 0], chain_angles[j, 1]
            v, p = rot(v, p, alpha, beta)
            joint = joint + v * l
            landmarks[base_idx + j + 1, :] = joint

    return np.concatenate([landmarks[:1, :], landmarks[2:, :]], axis=0)
