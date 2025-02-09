import time


class FPSCounter:
    def __init__(self):
        self._count = 0
        self._fps = 0
        self._update_time = time.time()

    def count(self):
        self._count += 1
        current_time = time.time()
        if current_time - self._update_time >= 1.0:
            self._fps = self._count
            self._update_time = current_time
            self._count = 0

    def get_fps(self):
        return self._fps
