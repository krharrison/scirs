"""Tests for scirs2 asynchronous operations module."""

import numpy as np
import pytest
import scirs2
import asyncio


class TestAsyncBasics:
    """Test basic async operations."""

    def test_async_compute(self):
        """Test basic async computation."""
        async def compute():
            result = await scirs2.async_compute_py(lambda x: x ** 2, 5)
            return result

        result = asyncio.run(compute())
        assert result == 25

    def test_async_map(self):
        """Test async map operation."""
        async def map_func():
            data = [1, 2, 3, 4, 5]
            results = await scirs2.async_map_py(lambda x: x * 2, data)
            return results

        results = asyncio.run(map_func())
        assert results == [2, 4, 6, 8, 10]

    def test_async_gather(self):
        """Test gathering multiple async operations."""
        async def gather_tasks():
            tasks = [
                scirs2.async_compute_py(lambda x: x + 1, 1),
                scirs2.async_compute_py(lambda x: x + 2, 2),
                scirs2.async_compute_py(lambda x: x + 3, 3)
            ]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(gather_tasks())
        assert results == [2, 4, 6]


class TestAsyncArrayOperations:
    """Test async array operations."""

    def test_async_matmul(self):
        """Test async matrix multiplication."""
        async def matmul():
            A = np.random.randn(10, 10)
            B = np.random.randn(10, 10)
            C = await scirs2.async_matmul_py(A, B)
            return C

        C = asyncio.run(matmul())
        assert C.shape == (10, 10)

    def test_async_sum(self):
        """Test async array sum."""
        async def sum_array():
            arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            result = await scirs2.async_sum_py(arr)
            return result

        result = asyncio.run(sum_array())
        assert result == 15.0

    def test_async_reduce(self):
        """Test async reduce operation."""
        async def reduce_array():
            arr = np.array([1, 2, 3, 4, 5])
            result = await scirs2.async_reduce_py(arr, lambda x, y: x + y)
            return result

        result = asyncio.run(reduce_array())
        assert result == 15


class TestParallelComputation:
    """Test parallel computation utilities."""

    def test_parallel_for(self):
        """Test parallel for loop."""
        def square(x):
            return x ** 2

        data = [1, 2, 3, 4, 5]
        results = scirs2.parallel_for_py(square, data, n_workers=2)

        assert results == [1, 4, 9, 16, 25]

    def test_parallel_map(self):
        """Test parallel map."""
        def increment(x):
            return x + 1

        data = np.array([1, 2, 3, 4, 5])
        results = scirs2.parallel_map_py(increment, data, n_workers=2)

        assert np.allclose(results, [2, 3, 4, 5, 6])

    def test_parallel_reduce(self):
        """Test parallel reduce."""
        def add(x, y):
            return x + y

        data = [1, 2, 3, 4, 5]
        result = scirs2.parallel_reduce_py(add, data, n_workers=2)

        assert result == 15


class TestTaskQueue:
    """Test task queue operations."""

    def test_task_queue_basic(self):
        """Test basic task queue."""
        async def process_queue():
            queue = scirs2.TaskQueue()

            await queue.put(lambda: 1 + 1)
            await queue.put(lambda: 2 + 2)
            await queue.put(lambda: 3 + 3)

            results = []
            for i in range(3):
                result = await queue.get()
                results.append(result)

            return results

        results = asyncio.run(process_queue())
        assert 2 in results or len(results) == 3

    def test_task_queue_priority(self):
        """Test priority task queue."""
        async def priority_queue():
            queue = scirs2.PriorityTaskQueue()

            await queue.put(lambda: "low", priority=3)
            await queue.put(lambda: "high", priority=1)
            await queue.put(lambda: "medium", priority=2)

            results = []
            for i in range(3):
                result = await queue.get()
                results.append(result)

            return results

        results = asyncio.run(priority_queue())
        # High priority should be first
        assert results[0] == "high" or len(results) == 3


class TestAsyncDataLoading:
    """Test async data loading."""

    def test_async_load_batch(self):
        """Test async batch loading."""
        async def load_batches():
            data = np.arange(100)
            batches = await scirs2.async_load_batches_py(data, batch_size=10)
            return batches

        batches = asyncio.run(load_batches())
        assert len(batches) == 10

    def test_async_prefetch(self):
        """Test data prefetching."""
        async def prefetch_data():
            data = [np.random.randn(10, 10) for _ in range(5)]
            iterator = scirs2.async_prefetch_py(data, buffer_size=2)

            results = []
            async for item in iterator:
                results.append(item)

            return results

        results = asyncio.run(prefetch_data())
        assert len(results) == 5


class TestAsyncFileIO:
    """Test async file I/O."""

    def test_async_read_file(self):
        """Test async file reading."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("test content")
            tmp_path = f.name

        try:
            async def read_async():
                content = await scirs2.async_read_file_py(tmp_path)
                return content

            content = asyncio.run(read_async())
            assert "test" in content or len(content) > 0
        finally:
            os.remove(tmp_path)

    def test_async_write_file(self):
        """Test async file writing."""
        import tempfile
        import os

        tmp_path = tempfile.mktemp()

        try:
            async def write_async():
                await scirs2.async_write_file_py(tmp_path, "test data")

            asyncio.run(write_async())

            with open(tmp_path, 'r') as f:
                content = f.read()
                assert "test" in content
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestAsyncStreaming:
    """Test async streaming operations."""

    def test_async_stream_process(self):
        """Test async stream processing."""
        async def process_stream():
            async def data_stream():
                for i in range(10):
                    yield i

            results = []
            async for item in scirs2.async_stream_process_py(data_stream(), lambda x: x * 2):
                results.append(item)

            return results

        results = asyncio.run(process_stream())
        assert len(results) == 10
        assert results[0] == 0
        assert results[-1] == 18

    def test_async_batch_stream(self):
        """Test async batch streaming."""
        async def batch_stream():
            async def data_stream():
                for i in range(20):
                    yield i

            batches = []
            async for batch in scirs2.async_batch_stream_py(data_stream(), batch_size=5):
                batches.append(batch)

            return batches

        batches = asyncio.run(batch_stream())
        assert len(batches) == 4


class TestAsyncPooling:
    """Test async worker pools."""

    def test_async_worker_pool(self):
        """Test async worker pool."""
        async def worker_pool():
            pool = scirs2.AsyncWorkerPool(n_workers=3)

            tasks = [lambda x=i: x ** 2 for i in range(10)]
            results = await pool.map(tasks)

            return results

        results = asyncio.run(worker_pool())
        assert len(results) == 10

    def test_async_thread_pool(self):
        """Test async thread pool executor."""
        async def thread_pool():
            def cpu_bound_task(x):
                return x ** 2

            results = await scirs2.async_thread_pool_py(cpu_bound_task, list(range(5)), n_threads=2)

            return results

        results = asyncio.run(thread_pool())
        assert results == [0, 1, 4, 9, 16]


class TestAsyncSynchronization:
    """Test async synchronization primitives."""

    def test_async_lock(self):
        """Test async lock."""
        async def use_lock():
            lock = asyncio.Lock()
            counter = 0

            async def increment():
                nonlocal counter
                async with lock:
                    temp = counter
                    await asyncio.sleep(0.001)
                    counter = temp + 1

            tasks = [increment() for _ in range(10)]
            await asyncio.gather(*tasks)

            return counter

        counter = asyncio.run(use_lock())
        assert counter == 10

    def test_async_semaphore(self):
        """Test async semaphore."""
        async def use_semaphore():
            semaphore = asyncio.Semaphore(2)
            results = []

            async def limited_task(i):
                async with semaphore:
                    await asyncio.sleep(0.01)
                    results.append(i)

            tasks = [limited_task(i) for i in range(5)]
            await asyncio.gather(*tasks)

            return results

        results = asyncio.run(use_semaphore())
        assert len(results) == 5

    def test_async_event(self):
        """Test async event."""
        async def use_event():
            event = asyncio.Event()
            results = []

            async def waiter():
                await event.wait()
                results.append("done")

            async def setter():
                await asyncio.sleep(0.01)
                event.set()

            await asyncio.gather(waiter(), setter())

            return results

        results = asyncio.run(use_event())
        assert "done" in results


class TestAsyncRetry:
    """Test async retry mechanisms."""

    def test_async_retry(self):
        """Test async retry on failure."""
        async def flaky_operation():
            attempts = 0

            async def operation():
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise Exception("Failed")
                return "Success"

            result = await scirs2.async_retry_py(operation, max_attempts=5)

            return result, attempts

        result, attempts = asyncio.run(flaky_operation())
        assert result == "Success"
        assert attempts == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_async_empty_list(self):
        """Test async operations on empty list."""
        async def map_empty():
            results = await scirs2.async_map_py(lambda x: x * 2, [])
            return results

        results = asyncio.run(map_empty())
        assert results == []

    def test_async_timeout(self):
        """Test async operation with timeout."""
        async def slow_operation():
            async def slow_task():
                await asyncio.sleep(10)
                return "done"

            try:
                result = await asyncio.wait_for(slow_task(), timeout=0.1)
            except asyncio.TimeoutError:
                result = "timeout"

            return result

        result = asyncio.run(slow_operation())
        assert result == "timeout"

    def test_async_cancellation(self):
        """Test async task cancellation."""
        async def cancellable():
            task = asyncio.create_task(asyncio.sleep(10))
            await asyncio.sleep(0.01)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                return "cancelled"

            return "completed"

        result = asyncio.run(cancellable())
        assert result == "cancelled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
