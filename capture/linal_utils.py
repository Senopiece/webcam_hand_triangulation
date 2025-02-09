import numpy as np


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions


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
