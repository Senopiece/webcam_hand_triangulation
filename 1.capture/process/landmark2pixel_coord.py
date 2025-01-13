# Mediapipe Landmarks arrive in normalized coordinates, so we need to convert them to pixel coordinates
# to make trianulation and projections (since they are in pixel coordinates)


def landmark_to_pixel_coord(frame_shape, lm):
    """
    Result is in pixel coordinates
    """
    h, w, _ = frame_shape

    x = lm.x * w
    y = lm.y * h

    return x, y
