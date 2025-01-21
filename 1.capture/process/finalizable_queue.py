from abc import ABC, abstractmethod


class EmptyFinalized(Exception):
    pass


class FinalizableQueue(ABC):
    @abstractmethod
    def put(self, item):
        ...

    @abstractmethod
    def get(self):
        ...
    
    @abstractmethod
    def task_done(self):
        ...
    
    @abstractmethod
    def qsize(self):
        ...
    
    @abstractmethod
    def empty(self):
        ...

    @abstractmethod
    def finalize(self):
        ...
