import multiprocessing

class FinalizableQueue:
    def __init__(self, maxsize=0):
        self._queue = multiprocessing.Queue(maxsize)
        self._finalized_event = multiprocessing.Event()

    def put(self, item, block=True, timeout: float|None = None):
        """
        Adds an item to the queue if it's not finalized.
        """
        if self._finalized_event.is_set():
            raise RuntimeError("Cannot put items into a finalized queue.")
        self._queue.put(item, block, timeout)

    def get(self, block=True, timeout: float|None = None):
        """
        Retrieves an item from the queue.
        """
        return self._queue.get(block, timeout)
    
    def qsize(self):
        return self._queue.qsize()
    
    def empty(self):
        return self._queue.empty()

    def finalize(self):
        """
        Finalizes the queue, preventing any further put operations.
        """
        self._finalized_event.set()

    def is_finalized(self):
        """
        Returns True if the queue is finalized, otherwise False.
        This value is sharable among processes.
        """
        return self._finalized_event.is_set()
