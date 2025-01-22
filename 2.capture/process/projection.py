import cv2
import numpy as np


def distorted_project(point3d, rvec, T, intrinsic_mtx, dist_coeffs):
    """
    Project with distortion
    Result is in pixel coordinates
    """
    projected_point, _ = cv2.projectPoints(
        point3d,
        rvec,
        T,
        intrinsic_mtx,
        dist_coeffs,
    )
    projected_point = projected_point[0][0]
    return projected_point[0], projected_point[1]

def project(point3d, P):
    """
    Project without distortion
    Result is in pixel coordinates
    """
    # Convert the 3D point to homogeneous coordinates
    point_3d_homogeneous = np.append(point3d, 1)  # [X, Y, Z, 1]

    # Project the point using the projection matrix
    pixel_homogeneous = P @ point_3d_homogeneous  # [u, v, w]

    # Normalize to get pixel coordinates
    pixel_coordinates = pixel_homogeneous[:2] / pixel_homogeneous[2]  # [u/w, v/w]

    return pixel_coordinates