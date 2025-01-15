
from typing import Any, List
import cv2
import numpy as np
from pydantic import BaseModel, Field

from async_cb import CBProcessingPool

class PoV(BaseModel):
    cam_id: int
    cap: cv2.VideoCapture
    processor: CBProcessingPool

    frame: cv2.typing.MatLike | None = None
    corners: Any | None = None

    shots: List[Any] = []

    mtx: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None

    class Config:
        arbitrary_types_allowed = True
