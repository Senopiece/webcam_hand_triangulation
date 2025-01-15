from typing import List
import cv2
import numpy as np
from pydantic import BaseModel, Field

from async_hands import HandTrackersPool

class IntrinsicConf(BaseModel):
    focal_length_pixels: List[float] = Field(..., min_items=2, max_items=2)
    skew_coefficient: float
    principal_point: List[float] = Field(..., min_items=2, max_items=2)
    dist_coeffs: List[float]

class ExtrinsicConf(BaseModel):
    translation_mm: List[float] = Field(..., min_items=3, max_items=3)
    rotation_rodrigues: List[float] = Field(..., min_items=3, max_items=3)

class CamConf(BaseModel):
    index: int
    intrinsic: IntrinsicConf
    extrinsic: ExtrinsicConf
    focus: float

class IntrinsicCameraParams(BaseModel):
    mtx: np.ndarray
    dist_coeffs: np.ndarray

    class Config:
        arbitrary_types_allowed = True

class ExtrinsicCameraParams(BaseModel):
    rvec: np.ndarray
    T: np.ndarray
    R: np.ndarray = Field(init=False)

    def __init__(self, **data):
        super().__init__(**data)
        self.R, _ = cv2.Rodrigues(self.rvec)

    class Config:
        arbitrary_types_allowed = True

class CameraParams(BaseModel):
    intrinsic: IntrinsicCameraParams
    extrinsic: ExtrinsicCameraParams
    P: np.ndarray = Field(init=False)
    focus: float

    def __init__(self, **data):
        super().__init__(**data)

        # Make Projection matrix
        RT = np.hstack((self.extrinsic.R, self.extrinsic.T))  # Rotation and translation
        self.P = self.intrinsic.mtx @ RT  # Projection matrix

    class Config:
        arbitrary_types_allowed = True

class PoV(BaseModel):
    cam_idx: int
    cap: cv2.VideoCapture
    parameters: CameraParams

    frame: cv2.typing.MatLike
    
    hand_landmarks: List[np.ndarray] | None

    tracker: HandTrackersPool

    class Config:
        arbitrary_types_allowed = True

class ContextedLandmark:
    cam_idx: int
    P: np.ndarray
    lm: np.ndarray # undistorted pixel coords
