from multiprocessing import Queue, Condition, Lock
from finalizable_queue import EmptyFinalized, FinalizableQueue

class MultiprocessingFinalizableQueue(FinalizableQueue):
    def __init__(self):
        self._queue = Queue()
        self._finalized = False
        self._lock = Lock()
        self._non_empty_or_finalized = Condition()

    def put(self, item):
        with self._non_empty_or_finalized:
            with self._lock:
                if self._finalized:
                    raise RuntimeError("Cannot put items into a finalized queue.")
                self._queue.put(item)
                self._non_empty_or_finalized.notify()

    def get(self):
        with self._non_empty_or_finalized:
            while True:
                with self._lock:
                    if not self.empty():
                        return self._queue.get_nowait()
                    if self._finalized:
                        raise EmptyFinalized()
                self._non_empty_or_finalized.wait()
    
    def task_done(self):
        pass
    
    def qsize(self):
        return self._queue.qsize()
    
    def empty(self):
        return self._queue.empty()

    def finalize(self):
        with self._non_empty_or_finalized:
            with self._lock:
                self._finalized = True
                self._non_empty_or_finalized.notify_all()
