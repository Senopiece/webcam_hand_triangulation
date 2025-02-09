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
    Project a 3D point into 2D pixel coordinates and include the normalized z-value.

    Parameters:
        point3d (np.array): The 3D point [X, Y, Z].
        P (np.array): The 4x4 projection matrix.

    Returns:
        tuple: (pixel_coordinates, normalized_z)
            pixel_coordinates (np.array): The 2D pixel coordinates [u, v].
            normalized_z (float): The normalized depth value z/w.
    """
    # Convert the 3D point to homogeneous coordinates
    point_3d_homogeneous = np.append(point3d, 1)  # [X, Y, Z, 1]

    # Project the point using the projection matrix
    pixel_homogeneous = P @ point_3d_homogeneous  # [u, v, w]

    # Normalize to get pixel coordinates
    pixel_coordinates = pixel_homogeneous[:2] / pixel_homogeneous[2]  # [u/w, v/w]

    # Extract the normalized depth (z/w)
    normalized_z = pixel_homogeneous[2]  # The w-normalized z-value

    return pixel_coordinates, normalized_z


def compute_sphere_rotating_camera_projection_matrix(
    fov, near, far, spherical_pos, roll, distance, target, frame_width, frame_height
):
    """
    Computes the camera projection matrix for a spherical rotating camera.
    Fixes scaling when target is not (0, 0, 0) and applies roll while maintaining focus on the target.
    """
    # Aspect ratio and perspective projection matrix
    aspect_ratio = frame_width / frame_height
    f = 1 / np.tan(fov / 2)  # Focal length
    proj = np.array(
        [
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ]
    )

    # Spherical to Cartesian conversion
    azimuth, elevation = spherical_pos
    cx = distance * np.cos(elevation) * np.cos(azimuth)
    cy = distance * np.cos(elevation) * np.sin(azimuth)
    cz = distance * np.sin(elevation)
    camera_position = np.array([cx, cy, cz]) + target  # Offset by target

    # Camera forward, right, and up vectors
    forward = target - camera_position
    forward /= np.linalg.norm(forward)  # Normalize forward vector

    world_up = np.array([0, 0, 1])  # World up vector
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)  # Normalize right vector

    up = np.cross(forward, right)
    up /= np.linalg.norm(up)  # Normalize up vector

    # Apply roll transformation around the forward vector
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    forward_axis = forward  # Axis of rotation (forward vector)

    # Rodrigues' rotation formula to rotate `right` and `up` around `forward`
    def rotate_vector_around_axis(vector, axis, angle):
        return (
            vector * cos_roll
            + np.cross(axis, vector) * sin_roll
            + axis * np.dot(axis, vector) * (1 - cos_roll)
        )

    right = rotate_vector_around_axis(right, forward_axis, roll)
    up = rotate_vector_around_axis(up, forward_axis, roll)

    # View matrix construction
    view = np.eye(4)
    view[:3, :3] = np.vstack([right, up, -forward])  # Orientation matrix
    view[:3, 3] = -view[:3, :3] @ camera_position  # Translation

    # Combine projection and view matrices
    proj_view = proj @ view

    # Screen space transformation
    screen_transform = np.array(
        [
            [frame_width / 2, 0, 0, frame_width / 2],
            [0, -frame_height / 2, 0, frame_height / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Final matrix: screen_transform * proj_view
    return screen_transform @ proj_view
