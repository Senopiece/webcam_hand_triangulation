import numpy as np


class HandWriter:
    def __init__(self, file: str):
        self.file = file
        self._f = open(file, "ab")  # append binary

    def add(self, pose: np.ndarray):
        if pose.shape != (20, 3) or pose.dtype != np.float32:
            raise ValueError(
                f"Pose must be of shape (20, 3) and dtype float32, but got {pose.shape} and dtype {pose.dtype}"
            )
        self._f.write(pose.tobytes())

    def close(self):
        self._f.close()
