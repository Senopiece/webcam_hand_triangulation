import asyncio
import threading
from typing import Callable
from .abstract import AsyncWorker, SyncWorker

class ThreadedAsyncWorker(AsyncWorker):
    def __init__(self, spawn: Callable[..., SyncWorker]):
        # NOTE: Assumed for .send to be called in the same loop the object is created

        self._spawn = spawn
        self.loop = asyncio.get_running_loop()
        self.in_data_event = threading.Event()
        self.out_data_event = asyncio.Event()
        self.out_data_event.set()
        self.disposed = False
        self.data = None
        self.res = None

        # Spawn a worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    async def dispose(self):
        await self.out_data_event.wait()
        self.disposed = True
        self.in_data_event.set()

    def _worker(self):
        subworker = self._spawn()

        while True:
            # Wait for input data
            self.in_data_event.wait()

            if self.disposed:
                subworker.dispose()
                break

            self.res = subworker.send(self.data)

            self.in_data_event.clear()  # drop anything that came while we were processing

            self.loop.call_soon_threadsafe(self.out_data_event.set)

    async def send(self, data):
        # NOTE: Unstable behavior if calling next send without avaiting the previous one
        # Meaning the next data will surely be throwed, but the result of the next data send
        # may be from previous one as well from the next sucesefully processed data

        if self.out_data_event.is_set():
            self.out_data_event.clear()
            self.data = data
            self.in_data_event.set()

        await self.out_data_event.wait()
        return self.res