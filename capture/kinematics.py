import numpy as np
from typing import List

# DEFAULT_MORPHOLOGY = np.array(
#     [
#         # thumb
#         [
#             # bones lengths
#             5.5556e-01,
#             3.8889e-01,
#             2.7778e-01,
#             # start point (x, y, z)
#             2.6502e-01,
#             2.5895e-01,
#             4.9295e-01,
#             # starting rotation axis angle (radians)
#             -4.6933e-02,
#         ],
#         # index
#         [
#             # bones lengths
#             4.4444e-01,
#             2.7778e-01,
#             2.2222e-01,
#             # starting point (x, y, z)
#             -9.9807e-04,
#             9.7040e-01,
#             2.3527e-01,
#             # starting rotation axis angle (radians)
#             -4.8198e-02,
#         ],
#         # middle
#         [
#             # bones lengths
#             5.5556e-01,
#             3.3333e-01,
#             2.2222e-01,
#             # starting point (x, y, z)
#             -9.3504e-04,
#             9.9172e-01,
#             -3.7544e-03,
#             # starting rotation axis angle (radians)
#             -1.8250e-02,
#         ],
#         # ring
#         [
#             # bones lengths
#             5.5556e-01,
#             3.3333e-01,
#             2.2222e-01,
#             # starting point (x, y, z)
#             -3.4443e-02,
#             9.1400e-01,
#             -2.0179e-01,
#             # starting rotation axis angle (radians)
#             -5.2318e-02,
#         ],
#         # pinky
#         [
#             # bones lengths
#             4.4444e-01,
#             2.7778e-01,
#             2.2222e-01,
#             # starting point (x, y, z)
#             -9.3163e-02,
#             7.8096e-01,
#             -3.8304e-01,
#             # starting rotation axis angle (radians)
#             -1.4947e-01,
#         ],
#     ],
#     dtype=np.float32,
# )

DEFAULT_MORPHOLOGY = np.array(
    [
        [0.6405, 0.3992, 0.2436, 0.0031, 0.0016, 0.4138, -0.1902],
        [0.4387, 0.2822, 0.2187, -0.0033, 0.9732, 0.2296, -0.0940],
        [0.5536, 0.3408, 0.2169, -0.0026, 0.9941, -0.0107, -0.0486],
        [0.5416, 0.3476, 0.2187, -0.0478, 0.9216, -0.2068, -0.0588],
        [0.4298, 0.2986, 0.2228, -0.1193, 0.7854, -0.3895, -0.1599],
    ],
    dtype=np.float32,
)


def rot(
    v: np.ndarray,
    p: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
):
    """
    Rotation function.

    v, p: (3,) arrays representing vector and the first axis of rotation
    alpha, beta: floats representing angles to use for rotation
    """
    v = v / np.linalg.norm(v)
    p = p / np.linalg.norm(p)

    sinA = np.sin(alpha)
    cosA = np.cos(alpha)
    sinB = np.sin(beta)
    cosB = np.cos(beta)

    q = np.cross(v, p)

    p_hat = p * cosB - v * sinB
    v_hat = q * sinA + v * cosA * cosB + p * cosA * sinB

    return v_hat, p_hat


P = np.array([0, 0, 1])


def hand_landmarks_by_angles(
    angles: np.ndarray,
    morphology: np.ndarray = DEFAULT_MORPHOLOGY,
):
    """
    angles: (20,) array
    morphology: (5, 7) array - not batched
    Returns: (20, 3) array
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
        gamma = morph[6]

        a0, b0, a2, a3 = local_angles
        chain_angles = np.array([[a0, b0], [a2, 0], [a3, 0]])

        v = joint / np.linalg.norm(joint)

        # Get p perpendicular to v
        p_proj = v * np.dot(v, P)
        p = P - p_proj
        p = p / np.linalg.norm(p)

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

    landmarks = np.concatenate(
        [landmarks[:1, :], landmarks[2:, :]], axis=0
    )  # cut thumb base

    return [l for l in landmarks]  # unroll landmarks


E = 1e-7


def irot_full_merged(
    v: np.ndarray,
    p: np.ndarray,
    v_hat: np.ndarray,
    v_hat_hat: np.ndarray,
    eps=0.7,
    eps_fallback=0.01,
):
    """
    Inverse rotation function with improved fallback.

    v, p, v_hat, v_hat_hat: (3,) arrays
    Returns: alpha, beta, v_hat_update, p_hat
    """
    v = v / np.linalg.norm(v)
    p = p / np.linalg.norm(p)
    q = np.cross(v, p)

    # Attempt primary p_hat from cross product of v_hat and v_hat_hat
    p_hat_candidate = np.cross(v_hat, v_hat_hat)
    low_norm_mask = np.linalg.norm(p_hat_candidate) < eps

    # Fallback p_hat using irot_full1 logic
    dot_v = np.dot(v_hat, v)
    dot_p = np.dot(v_hat, p)
    sinA = np.dot(v_hat, q)
    sinA = np.clip(sinA, -1 + E, 1 - E)

    cosA = np.sqrt(1 - sinA**2)
    beta_fallback = (
        np.arctan2(dot_p / cosA, dot_v / cosA) if cosA >= eps_fallback else 0
    )
    p_hat_fallback = p * np.cos(beta_fallback) - v * np.sin(beta_fallback)

    # Combine: use fallback p_hat only where primary one is invalid
    p_hat = (
        p_hat_fallback
        if low_norm_mask
        else p_hat_candidate / np.linalg.norm(p_hat_candidate)
    )

    # Continue as in irot_full
    cosB = np.dot(p_hat, p)
    sinB = -np.dot(p_hat, v)

    v_hat_update = q * sinA + v * (cosA * cosB) + p * (cosA * sinB)

    beta = np.arctan2(sinB, cosB)
    alpha = np.arcsin(sinA)

    return alpha, beta, v_hat_update, p_hat


def irot_alpha(v: np.ndarray, p: np.ndarray, v_hat: np.ndarray):
    """
    Inverse rotation function.

    v, p, v_hat: (3,) arrays
    Returns: alpha, beta, v_hat, p_hat
    """
    v = v / np.linalg.norm(v)
    p = p / np.linalg.norm(p)
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
):
    """
    landmarks: List[np.ndarray] - known hand landmarks in 3D space
    morphology: (5, 7) - hand morphology (not batched)

    Returns:
    angles: (20,) - recovered joint angles
    """
    angles = np.zeros((5, 4), dtype=landmarks[0].dtype)

    # NOTE: assuming 0, 4, 8, 12, 16 landmarks are existing in the morphology
    # 20 landmarks -> 21 landmarks, restoring thumb base from the morphology
    # so that now assuming 0, 1, 5, 9, 13, 17 are existing in the morphology
    thumb_base = morphology[0][3:6]

    # Insert thumb_base at the correct position
    landmarks.insert(1, thumb_base)

    for i, morph in enumerate(morphology):
        base_idx = i * 4 + 1
        joint = landmarks[base_idx] - landmarks[0]
        gamma = morph[6]

        v = joint / np.linalg.norm(joint)

        # Get p perpendicular to v
        p_proj = v * np.dot(v, P)
        p = P - p_proj
        p = p / np.linalg.norm(p)

        # Rotate p around v according to parameter gamma
        sinG, cosG = np.sin(gamma), np.cos(gamma)
        p = p * cosG + np.cross(p, v) * sinG

        target = (landmarks[base_idx + 1] - landmarks[base_idx]) / np.linalg.norm(
            landmarks[base_idx + 1] - landmarks[base_idx]
        )
        target2 = (landmarks[base_idx + 2] - landmarks[base_idx + 1]) / np.linalg.norm(
            landmarks[base_idx + 2] - landmarks[base_idx + 1]
        )
        alpha, beta, v, p = irot_full_merged(v, p, target, target2)
        angles[i, 0] = alpha
        angles[i, 1] = beta

        for j in range(1, 3):
            target = (
                landmarks[base_idx + j + 1] - landmarks[base_idx + j]
            ) / np.linalg.norm(landmarks[base_idx + j + 1] - landmarks[base_idx + j])
            alpha, beta, v, p = irot_alpha(v, p, target)
            angles[i, j + 1] = alpha

    return angles.flatten()
