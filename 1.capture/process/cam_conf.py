from typing import Dict
from pydantic import ValidationError
from models import CamConf, CameraParams
import json5
import numpy as np


def load_cam_conf(cam_conf: CamConf):
    idx = cam_conf.index

    # Intrinsic parameters
    intrinsic = cam_conf.intrinsic
    intrinsic_mtx = np.array(
        [
            [
                intrinsic.focal_length_pixels[0],
                intrinsic.skew_coefficient,
                intrinsic.principal_point[0],
            ],
            [
                0,
                intrinsic.focal_length_pixels[1],
                intrinsic.principal_point[1],
            ],
            [0, 0, 1],
        ]
    )
    dist_coeffs = np.array(intrinsic.dist_coeffs)

    # Extrinsic parameters
    extrinsic = cam_conf.extrinsic

    T = np.array([extrinsic.translation_mm], dtype=np.float64).T
    if T.shape != (3, 1):
        raise ValueError(f"Invalid translation_mm shape for camera {idx}, expected 3x1.")

    rvec = np.array([extrinsic.rotation_rodrigues], dtype=np.float64).T
    if rvec.shape != (3, 1):
        raise ValueError(f"Invalid rotation_rodrigues shape for camera {idx}, expected 1x3.")

    # Return parameters
    return idx, CameraParams(
        intrinsic_mtx=intrinsic_mtx,
        dist_coeffs=dist_coeffs,
        rvec=rvec,
        T=T,
        focus=cam_conf.focus,
    )


def load_cameras_parameters(cameras_file: str):
    with open(cameras_file, "r") as f:
        cameras_confs = json5.load(f)

    cameras: Dict[int, CamConf] = {}
    for cam_conf_dict in cameras_confs:
        try:
            cam_conf = CamConf(**cam_conf_dict)
            idx, params = load_cam_conf(cam_conf)
            cameras[idx] = params
        except ValidationError as e:
            print(f"Validation error for camera configuration: {e}")

    return cameras