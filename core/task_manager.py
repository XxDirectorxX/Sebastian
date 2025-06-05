# core/task_manager.py

import threading
from queue import Queue, Empty
from typing import Callable, Any, Optional

class TaskManager:
    """
    Manages task queue and execution for asynchronous or deferred operations.
    """

    def __init__(self):
        self.task_queue = Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def add_task(self, func: Callable, *args, **kwargs):
        """
        Add a callable task to the queue.

        Args:
            func (Callable): Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        """
        self.task_queue.put((func, args, kwargs))

    def _worker(self):
        """
        Worker thread to execute tasks serially.
        """
        while True:
            try:
                func, args, kwargs = self.task_queue.get(timeout=1)
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    # Log exception appropriately here
                    print(f"[TaskManager] Error executing task: {e}")
                finally:
                    self.task_queue.task_done()
            except Empty:
                continue

    def wait_for_all(self, timeout: Optional[float] = None):
        """
        Block until all tasks are completed or timeout reached.

        Args:
            timeout (float or None): Timeout in seconds or None for indefinite.
        """
        self.task_queue.join()
