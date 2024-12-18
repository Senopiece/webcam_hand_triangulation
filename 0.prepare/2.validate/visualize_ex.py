import json5
import numpy as np
import matplotlib.pyplot as plt
import argparse


def euler_angles_to_rotation_matrix(yaw, pitch, roll):
    """
    Converts Euler angles (in radians) to a rotation matrix.
    """
    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    R = R_z @ R_y @ R_x
    return R


def draw_camera(ax, R, T, color="b", label=""):
    """
    Draws a simple representation of a camera in 3D space.

    Parameters:
    - ax: The matplotlib 3D axis to draw on.
    - R: Rotation matrix (3x3).
    - T: Translation vector (3x1).
    - color: Color of the camera representation.
    - label: Label for the camera.
    """
    # Define camera coordinate frame in its local space
    camera_size = 2  # Adjust the size of the camera representation
    corners = (
        np.array(
            [
                [0, 0, 0],
                [1, 0.5, 1.5],
                [1, -0.5, 1.5],
                [-1, -0.5, 1.5],
                [-1, 0.5, 1.5],
            ]
        )
        * camera_size
    )

    # Transform corners to world coordinates
    corners_transformed = (R @ corners.T).T + T.reshape(1, 3)

    # Plot the camera body (pyramid)
    ax.plot(
        [corners_transformed[0, 0], corners_transformed[1, 0]],
        [corners_transformed[0, 1], corners_transformed[1, 1]],
        [corners_transformed[0, 2], corners_transformed[1, 2]],
        color=color,
    )
    ax.plot(
        [corners_transformed[0, 0], corners_transformed[2, 0]],
        [corners_transformed[0, 1], corners_transformed[2, 1]],
        [corners_transformed[0, 2], corners_transformed[2, 2]],
        color=color,
    )
    ax.plot(
        [corners_transformed[0, 0], corners_transformed[3, 0]],
        [corners_transformed[0, 1], corners_transformed[3, 1]],
        [corners_transformed[0, 2], corners_transformed[3, 2]],
        color=color,
    )
    ax.plot(
        [corners_transformed[0, 0], corners_transformed[4, 0]],
        [corners_transformed[0, 1], corners_transformed[4, 1]],
        [corners_transformed[0, 2], corners_transformed[4, 2]],
        color=color,
    )

    # Plot the base of the camera (rectangle)
    ax.plot(
        [corners_transformed[1, 0], corners_transformed[2, 0]],
        [corners_transformed[1, 1], corners_transformed[2, 1]],
        [corners_transformed[1, 2], corners_transformed[2, 2]],
        color=color,
    )
    ax.plot(
        [corners_transformed[2, 0], corners_transformed[3, 0]],
        [corners_transformed[2, 1], corners_transformed[3, 1]],
        [corners_transformed[2, 2], corners_transformed[3, 2]],
        color=color,
    )
    ax.plot(
        [corners_transformed[3, 0], corners_transformed[4, 0]],
        [corners_transformed[3, 1], corners_transformed[4, 1]],
        [corners_transformed[3, 2], corners_transformed[4, 2]],
        color=color,
    )
    ax.plot(
        [corners_transformed[4, 0], corners_transformed[1, 0]],
        [corners_transformed[4, 1], corners_transformed[1, 1]],
        [corners_transformed[4, 2], corners_transformed[1, 2]],
        color=color,
    )

    # Add label
    ax.text(
        T[0],
        T[1],
        T[2],
        f"Camera {label}",
        color=color,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize Camera Extrinsics")
    parser.add_argument(
        "--file",
        type=str,
        default="setup.json5",
        help="Path to the cameras declarations file",
    )
    args = parser.parse_args()
    cameras_path = args.file

    # Load camera configurations from the JSON file
    with open(cameras_path, "r") as f:
        cameras_confs = json5.load(f)

    # Extract extrinsic parameters
    cameras = []
    for camera_conf in cameras_confs:
        idx = camera_conf["index"]
        if "extrinsic" not in camera_conf:
            print(f"Warning: Camera {idx} does not have extrinsic parameters.")
            continue

        extrinsic = camera_conf["extrinsic"]
        translation_cm = extrinsic["translation_centimeters"]
        rotation_rad = extrinsic["rotation_radians"]

        T = np.array(
            [
                translation_cm["x"],
                translation_cm["y"],
                translation_cm["z"],
            ]
        )

        yaw = rotation_rad["yaw"]
        pitch = rotation_rad["pitch"]
        roll = rotation_rad["roll"]

        R = euler_angles_to_rotation_matrix(yaw, pitch, roll)

        cameras.append(
            {
                "index": idx,
                "R": R,
                "T": T,
            }
        )

    # Plot cameras
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Define colors for cameras
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    color_map = {}
    for i, cam in enumerate(cameras):
        color_map[cam["index"]] = colors[i % len(colors)]

    # Draw cameras
    for cam in cameras:
        draw_camera(
            ax,
            cam["R"],
            cam["T"],
            color=color_map[cam["index"]],
            label=str(cam["index"]),
        )

    # Set labels
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.set_title("Camera Extrinsics Visualization")

    # Adjust the view angle for better visualization
    ax.view_init(elev=30, azim=-60)

    # Show grid
    ax.grid(True)

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
