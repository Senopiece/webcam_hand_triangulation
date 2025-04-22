import numpy as np
import torch


def rotation_matrix_from_vectors(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    a = vec1 / vec1.norm()
    b = vec2 / vec2.norm()
    v = torch.cross(a, b, dim=-1)
    if v.norm() < eps:
        return torch.eye(3, dtype=vec1.dtype, device=vec1.device)
    c = torch.dot(a, b)
    s = v.norm()
    K = torch.tensor(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        dtype=vec1.dtype,
        device=vec1.device,
    )
    R = torch.eye(3, dtype=vec1.dtype, device=vec1.device)
    return R + K + K @ K * ((1 - c) / (s * s))


def signed_distance_to_edge(p, a, b):
    """
    Calculate the signed distance from a point to a line segment.
    :param p: The (x, y) coordinates of the point.
    :param a: The (x, y) coordinates of the start of the line segment.
    :param b: The (x, y) coordinates of the end of the line segment.
    :return: The distance from the point to the line segment.
    """
    ba = b - a
    pa = p - a
    h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
    projection = a + h * ba
    return np.linalg.norm(p - projection)


def is_point_in_polygon(p, polygon):
    """
    Check if a point is inside a polygon using the winding number method.
    :param p: The (x, y) coordinates of the point to check.
    :param polygon: A NumPy array of (x, y) tuples defining the polygon vertices.
    :return: True if the point is inside the polygon, False otherwise.
    """
    angle_sum = 0.0
    num_points = len(polygon)
    for i in range(num_points):
        a = polygon[i]
        b = polygon[(i + 1) % num_points]
        va = a - p
        vb = b - p
        angle = np.arctan2(np.cross(va, vb), np.dot(va, vb))
        angle_sum += angle
    return np.abs(angle_sum) > 1e-5  # ~2*pi means the point is inside


def is_point_in_smoothed_polygon(p, polygon, smoothing_distance):
    """
    Check if a point is inside the smoothed polygon by measuring the signed distance to all edges.
    :param p: The (x, y) coordinates of the point to check.
    :param polygon: A NumPy array of (x, y) tuples defining the polygon vertices.
    :param smoothing_distance: The smoothing distance to consider as inside the polygon.
    :return: True if the point is inside the smoothed polygon, False otherwise.
    """
    num_points = len(polygon)
    min_distance = np.inf

    for i in range(num_points):
        a = polygon[i]
        b = polygon[(i + 1) % num_points]
        distance = signed_distance_to_edge(p, a, b)
        min_distance = min(min_distance, distance)

    # Check if the point is either inside the polygon or within the smoothing distance
    if is_point_in_polygon(p, polygon) or min_distance < smoothing_distance:
        return True
    return False
