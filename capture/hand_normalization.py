import numpy as np
from .linal_utils import rotation_matrix_from_vectors
import numpy as np
import torch


def normalize_hand(
    hand_3d_points: torch.Tensor | np.ndarray,
    whrist_base: int = 0,
    middle_finger_inner_bone: int = 8,
    point_finger_inner_bone: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Normalize one hand (L,3) or a batch of hands (B, L, 3):
      1. translate so point `whrist_base` is at the origin
      2. rotate so `middle_finger_inner_bone` aligns with +Y
      3. scale so that vector to `middle_finger_inner_bone` has length 1
      4. rotate around Y so `point_finger_inner_bone` lies in the +Z half-plane
    Returns same shape as input.
    """
    if not isinstance(hand_3d_points, torch.Tensor):
        hand_3d_points = torch.tensor(hand_3d_points, dtype=torch.float32)

    # if batch of hands, just loop
    if hand_3d_points.dim() == 3:
        B, L, _ = hand_3d_points.shape
        normalized = torch.empty_like(hand_3d_points)
        for b in range(B):
            # recursive call on each (L,3)
            normalized[b] = normalize_hand(
                hand_3d_points[b],
                whrist_base,
                middle_finger_inner_bone,
                point_finger_inner_bone,
                eps,
            )
        return normalized

    # --- below is the original single‑hand logic for shape (L,3) ---
    device, dtype = hand_3d_points.device, hand_3d_points.dtype

    # 1) translate so wrist base → origin
    T = -hand_3d_points[whrist_base]
    hand = hand_3d_points + T

    # 2) rotate so that middle finger inner bone aligns with +Y
    target_y = torch.tensor([0, 1, 0], dtype=dtype, device=device)
    R1 = rotation_matrix_from_vectors(hand[middle_finger_inner_bone], target_y, eps=eps)
    hand = hand @ R1.T

    # 3) scale so middle‐finger inner bone length → 1
    hand = hand / (hand[middle_finger_inner_bone][1] + eps)

    # 4) rotate around Y so the point finger inner bone lies in +Z half-plane
    v = hand[point_finger_inner_bone]
    xz = torch.stack([v[0], v[2]])
    norm_xz = xz.norm()
    if norm_xz >= eps:
        sinA, cosA = xz[0] / norm_xz, xz[1] / norm_xz
        R2 = torch.tensor(
            [[cosA, 0.0, -sinA], [0.0, 1.0, 0.0], [sinA, 0.0, cosA]],
            dtype=dtype,
            device=device,
        )
        hand = hand @ R2.T

    return hand
