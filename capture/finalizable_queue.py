from abc import ABC, abstractmethod
import multiprocessing
import queue
import threading
from typing import TypeVar, Generic

T = TypeVar("T")  # Generic type for the items in the queue


class EmptyFinalized(Exception):
    pass


class FinalizableQueue(ABC, Generic[T]):
    @abstractmethod
    def put(self, item: T) -> None: ...

    @abstractmethod
    def get(self) -> T: ...

    @abstractmethod
    def task_done(self) -> None: ...

    @abstractmethod
    def qsize(self) -> int: ...

    @abstractmethod
    def empty(self) -> bool: ...

    @abstractmethod
    def finalize(self) -> None: ...


class ThreadFinalizableQueue(FinalizableQueue[T]):
    def __init__(self) -> None:
        self._queue: queue.Queue[T] = queue.Queue()
        self._finalized: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._non_empty_or_finalized: threading.Condition = threading.Condition()

    def put(self, item: T) -> None:
        with self._non_empty_or_finalized:
            with self._lock:
                if self._finalized:
                    raise RuntimeError("Cannot put items into a finalized queue.")
                self._queue.put(item)
                self._non_empty_or_finalized.notify()

    def get(self) -> T:
        with self._non_empty_or_finalized:
            while True:
                with self._lock:
                    if not self.empty():
                        return self._queue.get_nowait()
                    if self._finalized:
                        raise EmptyFinalized()
                self._non_empty_or_finalized.wait()

    def task_done(self) -> None:
        with self._lock:
            return self._queue.task_done()

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def finalize(self) -> None:
        with self._non_empty_or_finalized:
            with self._lock:
                self._finalized = True
                self._non_empty_or_finalized.notify_all()


class ProcessFinalizableQueue(FinalizableQueue[T]):
    def __init__(self) -> None:
        self._queue: multiprocessing.Queue[T] = multiprocessing.Queue()
        self._finalized = multiprocessing.Value("b", False)
        self._lock = multiprocessing.Lock()
        self._non_empty_or_finalized = multiprocessing.Condition()

    def put(self, item: T) -> None:
        with self._non_empty_or_finalized:
            with self._lock:
                if self._finalized.value:
                    raise RuntimeError("Cannot put items into a finalized queue.")
                self._queue.put(item)
                self._non_empty_or_finalized.notify()

    def get(self) -> T:
        with self._non_empty_or_finalized:
            while True:
                with self._lock:
                    if not self.empty():
                        return self._queue.get_nowait()
                    if self._finalized.value:
                        raise EmptyFinalized()
                self._non_empty_or_finalized.wait()

    def task_done(self) -> None:
        pass

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def finalize(self) -> None:
        with self._non_empty_or_finalized:
            with self._lock:
                self._finalized.value = True
                self._non_empty_or_finalized.notify_all()
