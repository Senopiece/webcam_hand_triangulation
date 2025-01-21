from abc import abstractmethod
from typing import Any, Callable

class SyncWorker:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()

    @abstractmethod
    def dispose(self):
        pass

    @abstractmethod
    def send(self, data: Any) -> Any:
        pass

class SyncWorkerDelegate(SyncWorker):
    def __init__(self,  _send: Callable[[Any], Any], _dispose: Callable[..., None]):
        self._send = _send
        self._dispose = _dispose
    
    def dispose(self):
        return self._dispose()

    def send(self, data: Any) -> Any:
        return self._send(data)

class AsyncWorker:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.dispose()

    @abstractmethod
    async def dispose(self):
        pass

    @abstractmethod
    async def send(self, data: Any) -> Any:
        pass