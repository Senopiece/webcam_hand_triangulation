from typing import Dict
import cv2
from pydantic import ValidationError
from models import CameraParams, ExtrinsicCameraParams, IntrinsicCameraParams
import json5
import numpy as np


def load_cam_params(cam_decl, cameras_calibs):
    idx = cam_decl["index"]
    calib = next(calib for calib in cameras_calibs if calib["index"] == idx)

    # Intrinsic parameters
    intrinsic = calib["intrinsic"]
    intrinsic_mtx = np.array(
        [
            [
                intrinsic["focal_length_pixels"][0],
                intrinsic["skew_coefficient"],
                intrinsic["principal_point"][0],
            ],
            [
                0,
                intrinsic["focal_length_pixels"][1],
                intrinsic["principal_point"][1],
            ],
            [0, 0, 1],
        ]
    )
    dist_coeffs = np.array(intrinsic["dist_coeffs"])

    # Extrinsic parameters
    extrinsic = calib["extrinsic"]

    T = np.array([extrinsic["translation_mm"]], dtype=np.float64).T
    if T.shape != (3, 1):
        raise ValueError(f"Invalid translation_mm shape for camera {idx}, expected 3x1.")

    rvec = np.array([extrinsic["rotation_rodrigues"]], dtype=np.float64).T
    if rvec.shape != (3, 1):
        raise ValueError(f"Invalid rotation_rodrigues shape for camera {idx}, expected 1x3.")

    R, _ = cv2.Rodrigues(rvec)

    # Make Projection matrix
    RT = np.hstack((R, T))  # Rotation and translation
    P = intrinsic_mtx @ RT  # Projection matrix

    # Return parameters
    size = list(map(int, cam_decl["size"].split("x")))
    return idx, CameraParams(
        intrinsic=IntrinsicCameraParams(mtx=intrinsic_mtx, dist_coeffs=dist_coeffs),
        extrinsic=ExtrinsicCameraParams(rvec=rvec, T=T, R=R),
        focus=cam_decl["focus"],
        fps=cam_decl["fps"],
        size=size,
        P=P,
    )


def load_cameras_parameters(decls_file: str, calibrations_file: str) -> Dict[int, CameraParams]:
    with open(decls_file, "r") as f:
        cameras_decls = json5.load(f)
    
    with open(calibrations_file, "r") as f:
        cameras_calibs = json5.load(f)

    cameras: Dict[int, CameraParams] = {}
    for cam_decl in cameras_decls:
        idx, params = load_cam_params(cam_decl, cameras_calibs)
        cameras[idx] = params

    return cameras