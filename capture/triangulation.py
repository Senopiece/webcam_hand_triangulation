from itertools import combinations
from typing import List
import cv2
import numpy as np

from models import ContextedLandmark
from projection import project


def stereo_triangulate_lmcs(P1, P2, lm1, lm2):
    # Requires undistorted pixel coords landmarks as input

    # points must be shaped as (2, N). Here N=1 since we have one point.
    pts1 = np.array([[lm1[0]], [lm1[1]]], dtype=np.float64)
    pts2 = np.array([[lm2[0]], [lm2[1]]], dtype=np.float64)

    # Triangulate
    pts4D = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # Convert from homogeneous to Cartesian coordinates
    X = pts4D[:3, 0] / pts4D[3, 0]

    return X


def best_stereo_triangulate_lmcs(lmcs: List[ContextedLandmark]):
    """
    Triangulate 3D point from multiple camera views.
    Makes all stereo projections and chooses the best
    """
    # Iterate all pairs of cameras to find best triangulation for each point
    best_point = None
    best_score = float("+inf")
    best_lmcs = []

    for stereo_lmcs in combinations(
        lmcs,
        2,
    ):
        point_3d = stereo_triangulate_lmcs(stereo_lmcs[0].P, stereo_lmcs[1].P, stereo_lmcs[0].lm, stereo_lmcs[1].lm)

        mean_reprojection_error = 0
        for lmc in stereo_lmcs:
            # Reproject onto the camera without distortion
            x1, y1, z1 = project(point_3d, lmc.P)
            x0, y0, z0 = lmc.lm

            # Compute the error
            reprojection_error = np.linalg.norm(np.array([x1, y1]) - np.array([x0, y0]))

            mean_reprojection_error += reprojection_error

        mean_reprojection_error /= len(stereo_lmcs)

        if mean_reprojection_error < best_score:
            best_point = point_3d
            best_score = mean_reprojection_error
            best_lmcs = stereo_lmcs

    return [lmc.cam_idx for lmc in best_lmcs], best_point

def triangulate_lmcs(lmcs: List[ContextedLandmark]):
    """
    Triangulate 3D point from multiple point of views.
    Alternative to best_stereo_triangulate_point using custom svg
    """
    if len(lmcs) >= 2:
        # Prepare matrices for triangulation
        A = []
        for lmc in lmcs:
            P = lmc.P
            x, y = lmc.lm
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])

        A = np.array(A)

        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]

        return [lmc.cam_idx for lmc in lmcs], X[:3]
    else:
        return [], None

