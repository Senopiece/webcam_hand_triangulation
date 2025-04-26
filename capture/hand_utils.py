import numpy as np
from typing import List

# convert mediapipe landmarks to reduced format
def rm_th_base(points_3d: List[np.ndarray]):
    points_3d_no_th_base = points_3d.copy()
    points_3d_no_th_base.pop(1)
    return points_3d_no_th_base