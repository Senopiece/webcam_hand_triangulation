import math
import numpy as np


def angle_between_vectors(v1, v2):
    # Compute the angle between two vectors
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Clip the value to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return angle_rad


def compute_finger_length(landmarks, joints):
    # Compute the length of a finger by summing the lengths of its bones
    length = 0
    for start_idx, end_idx in joints:
        p0 = np.array(landmarks[start_idx])
        p1 = np.array(landmarks[end_idx])
        segment_length = np.linalg.norm(p1 - p0)
        length += segment_length
    return length


# TODO: bad natureness score
def calculate_natureness(landmarks_3d):
    # Define joint connections and natural ranges (in degrees)
    joint_info = {
        "thumb": {
            "joints": [(1, 2), (2, 3), (3, 4)],
            "ranges": [(0, 50), (0, 80), (0, 80)],  # MCP, IP, tip
        },
        "index": {
            "joints": [(5, 6), (6, 7), (7, 8)],
            "ranges": [(0, 90), (0, 110), (0, 80)],  # MCP, PIP, DIP
        },
        "middle": {
            "joints": [(9, 10), (10, 11), (11, 12)],
            "ranges": [(0, 90), (0, 110), (0, 80)],
        },
        "ring": {
            "joints": [(13, 14), (14, 15), (15, 16)],
            "ranges": [(0, 90), (0, 110), (0, 80)],
        },
        "pinky": {
            "joints": [(17, 18), (18, 19), (19, 20)],
            "ranges": [(0, 90), (0, 110), (0, 80)],
        },
    }

    angle_total_deviation = 0
    angle_max_deviation = 0

    # Calculate deviations based on joint angles
    for finger in joint_info.values():
        joints = finger["joints"]
        ranges = finger["ranges"]

        for i, (start_idx, end_idx) in enumerate(joints):
            # Get the points
            p0 = np.array(landmarks_3d[start_idx])
            p1 = np.array(landmarks_3d[end_idx])

            if i == 0:
                # For the first joint, compute the vector from wrist to MCP
                wrist_idx = 0  # wrist
                p_prev = np.array(landmarks_3d[wrist_idx])
            else:
                # For other joints, use the previous joint's start point
                p_prev = np.array(landmarks_3d[joints[i - 1][0]])

            # Vectors representing bones
            v1 = p_prev - p0
            v2 = p1 - p0

            # Compute angle between v1 and v2
            angle_rad = angle_between_vectors(v1, v2)
            angle_deg = math.degrees(angle_rad)

            # Get natural range for the joint
            angle_min, angle_max = ranges[i]

            # Compute deviation from the natural range
            if angle_deg < angle_min:
                deviation = angle_min - angle_deg
            elif angle_deg > angle_max:
                deviation = angle_deg - angle_max
            else:
                deviation = 0

            angle_total_deviation += deviation
            angle_max_deviation += angle_max - angle_min

    # Calculate natureness score based on joint angles
    if angle_max_deviation > 0:
        angle_natureness_score = max(
            0, 100 - (angle_total_deviation / angle_max_deviation * 100)
        )
    else:
        angle_natureness_score = 100

    # Compute finger lengths
    finger_lengths = {}
    for finger_name, finger in joint_info.items():
        joints = finger["joints"]
        length = compute_finger_length(landmarks_3d, joints)
        finger_lengths[finger_name] = length

    # Normalize the lengths by the length of the middle finger
    middle_length = finger_lengths["middle"]
    normalized_lengths = {
        finger: length / middle_length for finger, length in finger_lengths.items()
    }

    # Define expected ratios (normalized to middle finger)
    expected_ratios = {
        "thumb": 0.75,  # Example value
        "index": 0.95,
        "middle": 1.0,  # Reference finger
        "ring": 0.95,
        "pinky": 0.75,
    }

    # Acceptable deviation for length ratios (e.g., +/-10%)
    acceptable_deviation = 0.1  # 10%

    length_total_deviation = 0
    length_max_deviation = 0

    # Calculate deviations based on finger length ratios
    for finger in expected_ratios:
        expected_ratio = expected_ratios[finger]
        actual_ratio = normalized_lengths[finger]
        lower_bound = expected_ratio * (1 - acceptable_deviation)
        upper_bound = expected_ratio * (1 + acceptable_deviation)

        if actual_ratio < lower_bound:
            deviation = lower_bound - actual_ratio
        elif actual_ratio > upper_bound:
            deviation = actual_ratio - upper_bound
        else:
            deviation = 0

        length_total_deviation += deviation
        # Max possible deviation is acceptable_deviation * expected_ratio
        length_max_deviation += acceptable_deviation * expected_ratio

    # Calculate natureness score based on finger lengths
    if length_max_deviation > 0:
        length_natureness_score = max(
            0, 100 - (length_total_deviation / length_max_deviation * 100)
        )
    else:
        length_natureness_score = 100

    # Combine the two natureness scores
    natureness_score = (angle_natureness_score + length_natureness_score) / 2

    return natureness_score
