from typing import Generic, TypeVar

T = TypeVar("T")

class Wrapped(Generic[T]):
    def __init__(self):
        self.data = None

    def set(self, data: T | None):
        self.data = data

    def get(self) -> T | None:
        return self.data