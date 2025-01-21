import multiprocessing
import multiprocessing.managers
import numpy as np

class SharedNumpyArray:
    def __init__(self, manager: multiprocessing.managers.SyncManager):
        self.meta = manager.dict()
        self.meta["obj"] = None

    def set(self, array: np.ndarray | None):
        self.meta["obj"] = array

    def get(self):
        return self.meta["obj"]
