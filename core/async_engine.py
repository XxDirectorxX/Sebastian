"""
Asynchronous processing engine for Sebastian assistant.

Implements concurrent task processing, non-blocking I/O operations,
and resource management for optimal performance under load.
"""

import asyncio
import logging
import time
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Coroutine, Set, Union
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import functools
import traceback

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Priority levels for async tasks."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class TaskStatus(Enum):
    """Status states for async tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class AsyncTask:
    """
    Represents an asynchronous task in the system.
    
    Tracks execution state, dependencies, and results.
    """
    
    def __init__(self, 
                coro: Coroutine,
                task_id: str = None,
                priority: TaskPriority = TaskPriority.NORMAL,
                timeout: Optional[float] = None,
                dependencies: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize async task.
        
        Args:
            coro: Coroutine to execute
            task_id: Unique task identifier (generated if None)
            priority: Task priority level
            timeout: Maximum execution time in seconds
            dependencies: List of task IDs that must complete first
            metadata: Additional task metadata
        """
        self.coro = coro
        self.task_id = task_id or str(uuid.uuid4())
        self.priority = priority
        self.timeout = timeout
        self.dependencies = dependencies or []
        self.metadata = metadata or {}
        
        # Execution state
        self.status = TaskStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
        self.task_object = None  # asyncio.Task object
        
    def __lt__(self, other):
        """Compare tasks by priority for sorting."""
        if not isinstance(other, AsyncTask):
            return NotImplemented
        return self.priority.value > other.priority.value  # Higher value = higher priority

class AsyncEngine:
    """
    Core asynchronous processing engine.
    
    Manages concurrent execution of tasks with priority queuing,
    dependency resolution, and resource management.
    """
    
    def __init__(self, 
                max_concurrency: int = 10,
                thread_pool_size: int = 20,
                default_timeout: float = 30.0):
        """
        Initialize async engine.
        
        Args:
            max_concurrency: Maximum concurrent async tasks
            thread_pool_size: Size of thread pool for blocking operations
            default_timeout: Default task timeout in seconds
        """
        self.max_concurrency = max_concurrency
        self.thread_pool_size = thread_pool_size
        self.default_timeout = default_timeout
        
        # Task tracking
        self.tasks: Dict[str, AsyncTask] = {}
        self.active_tasks: Set[str] = set()
        self.completed_tasks: Dict[str, Any] = {}
        self.task_queue = asyncio.PriorityQueue()
        
        # Execution resources
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self.loop = None
        self.running = False
        self._run_thread = None
        
        # Resource limiting
        self.semaphore = None  # Will be initialized when loop starts
        
        # Status
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "tasks_timeout": 0
        }
        
        logger.info(f"Async engine initialized with max concurrency {max_concurrency}")
        
    def start(self):
        """Start the async engine in a background thread."""
        if self.running:
            logger.warning("Async engine is already running")
            return
            
        self.running = True
        self._run_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._run_thread.start()
        
        logger.info("Async engine started")
        
    def _run_loop(self):
        """Run the asyncio event loop in the background thread."""
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Initialize concurrency limiter
            self.semaphore = asyncio.Semaphore(self.max_concurrency)
            
            # Start task processor
            self.loop.run_until_complete(self._process_tasks())
        except Exception as e:
            logger.error(f"Error in async engine event loop: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.loop.close()
            self.loop = None
            self.running = False
            logger.info("Async engine stopped")
            
    async def _process_tasks(self):
        """Process tasks from the queue until shutdown."""
        while self.running:
            try:
                # Get next task with highest priority
                _, task = await self.task_queue.get()
                
                # Check if task dependencies are met
                dependencies_met = True
                for dep_id in task.dependencies:
                    if dep_id not in self.completed_tasks:
                        # Put back in queue with slight delay
                        dependencies_met = False
                        self.loop.create_task(self._requeue_task(task, 0.5))
                        break
                        
                if not dependencies_met:
                    self.task_queue.task_done()
                    continue
                    
                # Limit concurrency
                async with self.semaphore:
                    # Run the task
                    self.active_tasks.add(task.task_id)
                    task.status = TaskStatus.RUNNING
                    task.start_time = time.time()
                    
                    # Create timeout wrapper if needed
                    timeout = task.timeout or self.default_timeout
                    task_coro = asyncio.wait_for(task.coro, timeout=timeout)
                    
                    # Execute task
                    task.task_object = self.loop.create_task(self._execute_task(task, task_coro))
                    
                self.task_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # Prevent tight loop on error
                
    async def _execute_task(self, task: AsyncTask, coro: Coroutine):
        """Execute a task with error handling and completion tracking."""
        try:
            # Execute the coroutine
            result = await coro
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.end_time = time.time()
            
            # Track completion
            self.completed_tasks[task.task_id] = {
                "result": result,
                "completed_at": task.end_time
            }
            self.stats["tasks_completed"] += 1
            
            logger.debug(f"Task {task.task_id} completed in {task.end_time - task.start_time:.2f}s")
            
            # Execute any completion callbacks
            if "on_complete" in task.metadata and callable(task.metadata["on_complete"]):
                try:
                    await self._maybe_await(task.metadata["on_complete"](task.result))
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")
                    
        except asyncio.TimeoutError:
            # Task timed out
            task.status = TaskStatus.TIMEOUT
            task.error = "Task execution timed out"
            task.end_time = time.time()
            self.stats["tasks_timeout"] += 1
            
            logger.warning(f"Task {task.task_id} timed out after {task.timeout}s")
            
            # Execute timeout callback if provided
            if "on_timeout" in task.metadata and callable(task.metadata["on_timeout"]):
                try:
                    await self._maybe_await(task.metadata["on_timeout"](task))
                except Exception as e:
                    logger.error(f"Error in timeout callback: {e}")
                    
        except asyncio.CancelledError:
            # Task was cancelled
            task.status = TaskStatus.CANCELLED
            task.error = "Task was cancelled"
            task.end_time = time.time()
            self.stats["tasks_cancelled"] += 1
            
            logger.info(f"Task {task.task_id} was cancelled")
            
            # Execute cancellation callback if provided
            if "on_cancel" in task.metadata and callable(task.metadata["on_cancel"]):
                try:
                    await self._maybe_await(task.metadata["on_cancel"](task))
                except Exception as e:
                    logger.error(f"Error in cancellation callback: {e}")
                    
        except Exception as e:
            # Task failed with exception
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.end_time = time.time()
            self.stats["tasks_failed"] += 1
            
            logger.error(f"Task {task.task_id} failed: {e}")
            logger.error(traceback.format_exc())
            
            # Execute error callback if provided
            if "on_error" in task.metadata and callable(task.metadata["on_error"]):
                try:
                    await self._maybe_await(task.metadata["on_error"](e, task))
                except Exception as callback_error:
                    logger.error(f"Error in error callback: {callback_error}")
                    
        finally:
            # Remove from active tasks
            self.active_tasks.discard(task.task_id)
            
    async def _maybe_await(self, result):
        """Helper to await result if it's awaitable."""
        if asyncio.iscoroutine(result):
            return await result
        return result
            
    async def _requeue_task(self, task: AsyncTask, delay: float):
        """Requeue a task after a delay."""
        await asyncio.sleep(delay)
        await self.task_queue.put((task.priority.value, task))
        
    def submit_task(self, 
                  coro_or_func,
                  *args,
                  task_id: str = None,
                  priority: TaskPriority = TaskPriority.NORMAL,
                  timeout: Optional[float] = None,
                  dependencies: Optional[List[str]] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  **kwargs) -> str:
        """
        Submit a task for asynchronous execution.
        
        Args:
            coro_or_func: Coroutine or function to execute
            *args: Arguments to pass to the coroutine/function
            task_id: Unique task identifier (generated if None)
            priority: Task priority level
            timeout: Maximum execution time in seconds
            dependencies: List of task IDs that must complete first
            metadata: Additional task metadata
            **kwargs: Keyword arguments to pass to the coroutine/function
            
        Returns:
            Task ID
        """
        if not self.running:
            raise RuntimeError("Async engine is not running")
            
        # Generate task ID if not provided
        task_id = task_id or str(uuid.uuid4())
        
        # Handle both coroutines and regular functions
        if asyncio.iscoroutine(coro_or_func) or asyncio.iscoroutinefunction(coro_or_func):
            # It's already a coroutine or coroutine function
            if asyncio.iscoroutinefunction(coro_or_func):
                coro = coro_or_func(*args, **kwargs)
            else:
                coro = coro_or_func
        else:
            # Regular function, wrap it in a coroutine
            async def run_in_executor():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.thread_pool,
                    lambda: coro_or_func(*args, **kwargs)
                )
            coro = run_in_executor()
            
        # Create task object
        task = AsyncTask(
            coro=coro,
            task_id=task_id,
            priority=priority,
            timeout=timeout,
            dependencies=dependencies,
            metadata=metadata
        )
        
        # Register task
        self.tasks[task_id] = task
        self.stats["tasks_submitted"] += 1
        
        # Submit to queue
        if self.loop and self.running:
            asyncio.run_coroutine_threadsafe(
                self.task_queue.put((task.priority.value, task)), 
                self.loop
            )
        
        logger.debug(f"Submitted task {task_id} with priority {priority.name}")
        return task_id
        
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dict with task status and details
        """
        if task_id not in self.tasks:
            return {"status": "unknown", "error": "Task not found"}
            
        task = self.tasks[task_id]
        result = {
            "id": task.task_id,
            "status": task.status.value,
            "priority": task.priority.name,
            "started": task.start_time is not None
        }
        
        if task.start_time:
            result["start_time"] = task.start_time
            result["elapsed"] = (task.end_time or time.time()) - task.start_time
            
        if task.end_time:
            result["end_time"] = task.end_time
            result["duration"] = task.end_time - task.start_time
            
        if task.status == TaskStatus.COMPLETED:
            result["result"] = task.result
        elif task.status in (TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED):
            result["error"] = task.error
            
        return result
        
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if it's still running.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled, False otherwise
        """
        if task_id not in self.tasks or task_id not in self.active_tasks:
            return False
            
        task = self.tasks[task_id]
        if task.task_object and not task.task_object.done():
            task.task_object.cancel()
            return True
            
        return False
        
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for task completion and get result.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dict with task result or status
        """
        if not self.running:
            raise RuntimeError("Async engine is not running")
            
        if task_id not in self.tasks:
            return {"status": "unknown", "error": "Task not found"}
            
        task = self.tasks[task_id]
        
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return self.get_task_status(task_id)
                
            time.sleep(0.1)
            
        # Timed out waiting
        return {
            "id": task_id,
            "status": "waiting_timeout",
            "task_status": task.status.value
        }
        
    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """
        Shut down the async engine.
        
        Args:
            wait: Whether to wait for tasks to complete
            timeout: Maximum time to wait in seconds
        """
        if not self.running:
            logger.warning("Async engine is already stopped")
            return
            
        logger.info("Shutting down async engine...")
        
        # Stop accepting new tasks
        self.running = False
        
        if wait:
            # Wait for active tasks to complete
            start_time = time.time()
            while self.active_tasks and time.time() - start_time < timeout:
                time.sleep(0.1)
                
        # Cancel any remaining tasks
        if self.loop and self.active_tasks:
            for task_id in list(self.active_tasks):
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    if task.task_object and not task.task_object.done():
                        task.task_object.cancel()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=wait)
        
        logger.info(f"Async engine shutdown complete. Stats: {self.stats}")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.
        
        Returns:
            Dict with execution statistics
        """
        stats = self.stats.copy()
        stats.update({
            "active_tasks": len(self.active_tasks),
            "pending_tasks": self.task_queue.qsize() if self.task_queue else 0,
            "running": self.running
        })
        return stats

# Create singleton instance
_async_engine = None

def get_async_engine():
    """Get singleton instance of AsyncEngine."""
    global _async_engine
    if _async_engine is None:
        _async_engine = AsyncEngine()
    return _async_engine

def init_async_engine():
    """Initialize and start async engine if not already running."""
    engine = get_async_engine()
    if not engine.running:
        engine.start()
    return engine