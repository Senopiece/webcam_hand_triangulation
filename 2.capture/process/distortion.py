import cv2
import numpy as np


def undistort_pixel_coord(point, intrinsic_mtx, dist_coeffs):
    """
    Input point is in pixel coordinates
    Result is in pixel coordinates
    """
    x, y = point

    undistorted = cv2.undistortPoints(
        np.array([[[x, y]]], dtype=np.float32),
        intrinsic_mtx,
        dist_coeffs,
        P=intrinsic_mtx,
    )

    return undistorted[0][0]