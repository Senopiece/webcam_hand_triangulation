from typing import List
import numpy as np
import torch
import numpy as np
import plotly.graph_objs as go


def read_hands(file: str):
    poses: List[np.ndarray] = []
    pose_size = 20 * 3  # number of floats
    with open(file, "rb") as f:
        data = f.read()
        total_floats = len(data) // 4  # float32 = 4 bytes
        if total_floats % pose_size != 0:
            raise ValueError("File size is not a multiple of single pose size")
        array = np.frombuffer(data, dtype=np.float32)
        for i in range(0, total_floats, pose_size):
            pose = array[i : i + pose_size].reshape((20, 3))
            poses.append(pose)
    return np.stack(poses, axis=0)


def plot_3d_hands(
    landmarks_sequence: torch.Tensor | np.ndarray,
    ps: torch.Tensor | np.ndarray | None = None,
    ps2: torch.Tensor | np.ndarray | None = None,
):
    if isinstance(landmarks_sequence, torch.Tensor):
        landmarks_sequence = landmarks_sequence.detach().cpu().numpy()

    if ps is not None and isinstance(ps, torch.Tensor):
        ps = ps.detach().cpu().numpy()
    if ps2 is not None and isinstance(ps2, torch.Tensor):
        ps2 = ps2.detach().cpu().numpy()

    n_frames = landmarks_sequence.shape[0]
    n_pts = landmarks_sequence.shape[1]
    labels = [str(i) for i in range(n_pts)]

    fig = go.Figure()

    # Add traces for the first frame
    fig.add_trace(
        go.Scatter3d(
            x=landmarks_sequence[0, :, 0],
            y=landmarks_sequence[0, :, 1],
            z=landmarks_sequence[0, :, 2],
            mode="markers+text",
            marker=dict(size=4, color="blue"),
            text=labels,
            textposition="top center",
            textfont=dict(size=8, color="black"),
            name="Landmarks",
        )
    )

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),  # Thumb
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 7),  # Index
        (0, 8),
        (8, 9),
        (9, 10),
        (10, 11),  # Middle
        (0, 12),
        (12, 13),
        (13, 14),
        (14, 15),  # Ring
        (0, 16),
        (16, 17),
        (17, 18),
        (18, 19),  # Pinky
        (1, 4),
        (4, 8),
        (8, 12),
        (12, 16),  # Palm
    ]

    # Add traces for connections for the first frame
    for i, j in connections:
        fig.add_trace(
            go.Scatter3d(
                x=[landmarks_sequence[0, i, 0], landmarks_sequence[0, j, 0]],
                y=[landmarks_sequence[0, i, 1], landmarks_sequence[0, j, 1]],
                z=[landmarks_sequence[0, i, 2], landmarks_sequence[0, j, 2]],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )

    def add_beams(landmarks, vectors, color):
        for i in range(n_pts):
            start = landmarks[i]
            end = start + vectors[i] * 0.1
            fig.add_trace(
                go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                )
            )

    # Add ps (red) and ps2 (green) beams if provided for first frame
    if ps is not None:
        add_beams(landmarks_sequence[0], ps[0], "red")
    if ps2 is not None:
        add_beams(landmarks_sequence[0], ps2[0], "green")

    # Create frames for the animation
    frames = []
    for k in range(n_frames):
        frame_data = [
            go.Scatter3d(
                x=landmarks_sequence[k, :, 0],
                y=landmarks_sequence[k, :, 1],
                z=landmarks_sequence[k, :, 2],
                mode="markers+text",
                marker=dict(size=4, color="blue"),
                text=labels,
                textposition="top center",
                textfont=dict(size=8, color="black"),
                name="Landmarks",
            )
        ] + [
            go.Scatter3d(
                x=[landmarks_sequence[k, i, 0], landmarks_sequence[k, j, 0]],
                y=[landmarks_sequence[k, i, 1], landmarks_sequence[k, j, 1]],
                z=[landmarks_sequence[k, i, 2], landmarks_sequence[k, j, 2]],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
            for i, j in connections
        ]

        if ps is not None:
            for i in range(n_pts):
                start = landmarks_sequence[k, i]
                end = start + ps[k, i] * 0.1
                frame_data.append(
                    go.Scatter3d(
                        x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        z=[start[2], end[2]],
                        mode="lines",
                        line=dict(color="red", width=2),
                        showlegend=False,
                    )
                )

        if ps2 is not None:
            for i in range(n_pts):
                start = landmarks_sequence[k, i]
                end = start + ps2[k, i] * 0.1
                frame_data.append(
                    go.Scatter3d(
                        x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        z=[start[2], end[2]],
                        mode="lines",
                        line=dict(color="green", width=2),
                        showlegend=False,
                    )
                )

        frames.append(go.Frame(data=frame_data, name=str(k)))

    fig.frames = frames

    # Add slider and play/pause button
    sliders = [
        {
            "pad": {"b": 10, "t": 10},
            "len": 0.9,
            "x": 0.1,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top",
            "steps": [
                {
                    "args": [
                        [f.name],
                        {
                            "frame": {"duration": 33.33, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(frames)
            ],
        }
    ]

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 33.33, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 30},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=sliders,
        scene=dict(
            xaxis=dict(
                range=[2, -2],
                title="X",
            ),
            yaxis=dict(
                range=[-1, 3],
                title="Y",
            ),
            zaxis=dict(
                range=[-2, 2],
                title="Z",
            ),
            camera=dict(eye=dict(x=0.2, y=0.2, z=0.2)),
        ),
        scene_aspectmode="cube",
        title="3D Hand Pose with Point Labels",
        margin=dict(l=0, r=0, b=0, t=30),
        height=400,
    )

    fig.show()
