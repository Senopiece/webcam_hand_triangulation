import time


class FPSCounter:
    def __init__(self, label: str | None = None):
        self.label = label + " " if label else ""

        self.frame_count = 0
        self.start_time = time.time()

        self.total_frame_count = 0
        self.very_start_time = self.start_time

    def count(self, count=1):
        self.total_frame_count += count
        self.frame_count += count
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            print(f"{self.label}FPS: {fps:.2f}")
            self.frame_count = 0
            self.start_time = time.time()

    def mean(self):
        print(
            f"Mean {self.label}FPS: {self.total_frame_count / (time.time() - self.very_start_time):.2f}"
        )
