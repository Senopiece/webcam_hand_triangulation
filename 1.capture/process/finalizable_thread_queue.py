import queue
import threading

class EmptyFinalized(Exception):
    pass

class FinalizableQueue:
    def __init__(self, maxsize=0):
        self._queue = queue.Queue(maxsize)
        self._finalized = False
        self._non_empty_or_finalized = threading.Condition()

    def put(self, item):
        with self._non_empty_or_finalized:
            if self._finalized:
                raise RuntimeError("Cannot put items into a finalized queue.")
            self._queue.put(item)
            self._non_empty_or_finalized.notify()

    def get(self):
        with self._non_empty_or_finalized:
            if self.empty():
                if self._finalized:
                    raise EmptyFinalized()
                self._non_empty_or_finalized.wait()
                if self.empty() and self._finalized:
                    raise EmptyFinalized()
            return self._queue.get_nowait()
    
    def task_done(self):
        return self._queue.task_done()
    
    def qsize(self):
        return self._queue.qsize()
    
    def empty(self):
        return self._queue.empty()

    def finalize(self):
        with self._non_empty_or_finalized:
            self._finalized = True
            self._non_empty_or_finalized.notify_all()

    def is_finalized(self):
        return self._finalized