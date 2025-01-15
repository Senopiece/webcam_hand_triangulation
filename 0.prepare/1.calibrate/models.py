
from typing import Any, List
import cv2
import numpy as np
from pydantic import BaseModel

class PoV(BaseModel):
    cam_id: int
    frame: cv2.typing.MatLike
    cap: cv2.VideoCapture
    corners: Any | None
    shots: List[Any]
    mtx: np.ndarray
    dist_coeffs: np.ndarray

    class Config:
        arbitrary_types_allowed = True
