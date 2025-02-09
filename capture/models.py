from typing import Tuple
import numpy as np
from pydantic import BaseModel


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
    size: Tuple[int, int]
    track: str

    class Config:
        arbitrary_types_allowed = True


class ContextedLandmark(BaseModel):
    cam_idx: int
    P: np.ndarray
    lm: np.ndarray  # undistorted pixel coords

    class Config:
        arbitrary_types_allowed = True
