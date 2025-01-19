import multiprocessing
import threading
from typing import List
import cv2
import numpy as np
from pydantic import BaseModel, Field

from workers.pool import AsyncWorkersPool

class IntrinsicCameraParams(BaseModel):
    mtx: np.ndarray
    dist_coeffs: np.ndarray

    class Config:
        arbitrary_types_allowed = True

class ExtrinsicCameraParams(BaseModel):
    rvec: np.ndarray
    T: np.ndarray
    R: np.ndarray

    class Config:
        arbitrary_types_allowed = True

class CameraParams(BaseModel):
    intrinsic: IntrinsicCameraParams
    extrinsic: ExtrinsicCameraParams
    P: np.ndarray
    focus: float
    fps: float
    size: List[float] = Field(..., min_items=2, max_items=2)

    class Config:
        arbitrary_types_allowed = True

class PoV(BaseModel):
    cam_idx: int
    cap_process: multiprocessing.Process
    parameters: CameraParams
    tracker: AsyncWorkersPool

    frame: cv2.typing.MatLike | None = None
    
    hand_landmarks: List[np.ndarray] | None = None

    class Config:
        arbitrary_types_allowed = True

class ContextedLandmark(BaseModel):
    cam_idx: int
    P: np.ndarray
    lm: np.ndarray # undistorted pixel coords

    class Config:
        arbitrary_types_allowed = True

class Landmark(BaseModel):
    x: float
    y: float