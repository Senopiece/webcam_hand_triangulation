import asyncio
import multiprocessing
import threading
from typing import Callable, Optional

from .abstract import AsyncWorker, SyncWorker

# TODO: review the code and remove unnecessary components, add more typings
class MultiprocessedAsyncWorker(AsyncWorker):
    def __init__(self, spawn: Callable[..., SyncWorker]):
        self.loop = asyncio.get_running_loop()
        
        # Two queues:
        #   - input_queue: main process => child process
        #   - output_queue: child process => main process
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        # Start the dedicated worker process
        self.worker_process = multiprocessing.Process(
            target=self._worker,
            args=(self.input_queue, self.output_queue, spawn),
            daemon=True,
        )
        self.worker_process.start()

        # Concurrency-related flags
        self._disposed = False
        self._busy_lock = asyncio.Lock()  # used to manage single concurrency
        self._current_result_future: Optional[asyncio.Future] = None

        # Unique ID for each request so we know which result corresponds
        self._next_request_id = 0
        self._futures_by_id = {}

        # Start a background thread to read results from output_queue
        self._result_thread = threading.Thread(
            target=self._result_collector, daemon=True
        )
        self._result_thread.start()
    
    @staticmethod
    def _worker(input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue, spawn: Callable[..., SyncWorker]):
        subworker = spawn()

        while True:
            item = input_queue.get()  # blocking read
            if item is None:
                break

            (request_id, data) = item
            if request_id is None or data is None:
                break

            res = subworker.send(data)

            output_queue.put((request_id, res))

        subworker.dispose()

    def _result_collector(self):
        """
        Runs in a background thread:
          - continuously reads results from `output_queue` (blocking)
          - for each result, finds the correct future in the main loop and sets it
        """
        while True:
            try:
                result = self.output_queue.get()
            except (EOFError, OSError):
                # If the queue is somehow closed, just exit
                break

            if result is None:
                # Sentinel => child is done
                break

            request_id, res = result

            # Switch back to the main thread (event loop) to set the future
            def _set_result_in_main():
                future = self._futures_by_id.pop(request_id, None)
                if future and not future.done():
                    future.set_result(res)

            self.loop.call_soon_threadsafe(_set_result_in_main)

    async def dispose(self):
        """
        Dispose the child process. Must not be reused after disposal.
        """
        if self._disposed:
            return

        # Wait for current operation to finish
        if self._current_result_future is not None:
            try:
                await self._current_result_future
            except:
                pass

        self._disposed = True

        # Send sentinel to worker => it will exit its loop
        self.input_queue.put(None)

        # Join the worker process
        self.worker_process.join(timeout=2.0)

        # Finally, also put sentinel into result_queue (in case the result thread is blocking)
        self.output_queue.put(None)

        # Attempt to close the queues
        self.input_queue.close()
        self.output_queue.close()

    async def send(self, data):
        if self._disposed:
            raise RuntimeError("Cannot send data after dispose() was called.")

        # Wait until we're free to send the next data
        await self._busy_lock.acquire()

        # Create a future that will eventually hold result
        future = self.loop.create_future()

        # Register the future using a new request_id
        request_id = self._next_request_id
        self._next_request_id += 1
        self._futures_by_id[request_id] = future

        # Put (request_id, data) onto the input queue for the worker process
        self.input_queue.put((request_id, data))

        # Keep track so we can await in dispose if we want
        self._current_result_future = future

        def _release_lock(_):
            # Once the future completes (success or exception),
            # release the concurrency lock
            if self._busy_lock.locked():
                self._busy_lock.release()

        future.add_done_callback(_release_lock)

        # Return the result
        return await future


class MultiprocessedHands:
    def __init__(self, spawn: Callable[..., SyncWorker]):        
        # Two queues:
        #   - input_queue: main process => child process
        #   - output_queue: child process => main process
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        # Start the dedicated worker process
        self.worker_process = multiprocessing.Process(
            target=self._worker,
            args=(self.input_queue, self.output_queue, spawn),
            daemon=True,
        )
        self.worker_process.start()

        # Concurrency-related flags
        self._disposed = False
    
    @staticmethod
    def _worker(input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue, spawn: Callable[..., SyncWorker]):
        subworker = spawn()

        while True:
            data = input_queue.get()  # blocking read
            if data is None:
                break

            res = subworker.send(data)

            output_queue.put(res)

        subworker.dispose()

    def dispose(self):
        """
        Dispose the child process. Must not be reused after disposal.
        """
        if self._disposed:
            return

        self._disposed = True

        # Send sentinel to worker => it will exit its loop
        self.input_queue.put(None)

        # Join the worker process
        self.worker_process.join(timeout=2.0)

        # Attempt to close the queues
        self.input_queue.close()
        self.output_queue.close()

    def send(self, data):
        # Concurrent sends are not supported, may arrive out of order
        if self._disposed:
            raise RuntimeError("Cannot send data after dispose() was called.")

        self.input_queue.put(data)
    
    def wait_result(self):
        return self.output_queue.get()