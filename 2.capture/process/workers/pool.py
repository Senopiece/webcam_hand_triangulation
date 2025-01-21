import asyncio
from typing import List
from .abstract import AsyncWorker

class AsyncWorkersPool(AsyncWorker):
    def __init__(self, pool: List[AsyncWorker]):
        self.pool = pool
        self.results = asyncio.Queue()  # Filled in order of .send calls
        self.idle_workers = asyncio.Queue(len(pool))
        self.last_task = None

        # Add all workers to the queue initially
        for worker in self.pool:
            self.idle_workers.put_nowait(worker)

    async def dispose(self):
        await asyncio.gather(*[worker.dispose() for worker in self.pool])

    async def _send(self, prev_task, worker: AsyncWorker, data):
        # NOTE: will hang if exception rises somewhere there

        res = await worker.send(data)

        self.idle_workers.put_nowait(worker)

        # Ensure queue is filled in order
        if prev_task is not None:
            await prev_task

        self.results.put_nowait(res)

    async def send(self, data):
        """
        Waits for an available worker and sends the data to it.
        NOTE: will return immediately if a worker is available, get the result from the results queue
              otherwise will block for the fist available worker, still get the result from the results queue
        """
        # Wait for a free worker from the queue
        worker = await self.idle_workers.get()
        self.last_task = asyncio.create_task(self._send(self.last_task, worker, data))