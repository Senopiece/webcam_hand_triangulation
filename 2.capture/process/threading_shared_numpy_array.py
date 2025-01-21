import numpy as np


class SharedNumpyArray:
    def __init__(self):
        self.data = None

    def set(self, array: np.ndarray | None):
        self.data = array

    def get(self) -> np.ndarray | None:
        return self.data