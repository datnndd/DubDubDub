import multiprocessing, os
import time

from videotrans.configure.config import app_cfg, settings, logger


def _task_worker_wrapper(func, kwargs):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # Execute actual task here
    return func(**kwargs)


class AsyncResultFutureWrapper:
    def __init__(self, async_result, pool_executor):
        self.async_result = async_result
        self._pool = pool_executor  # Holds reference to process pool to monitor health status

    def result(self, timeout=None):
        # Default total timeout is 1 hour, or custom
        start_time = time.time()
        actual_timeout = timeout if timeout is not None else 3600

        while True:
            # 1. Check whether finished (normal completion or caught Python exception)
            if self.async_result.ready():
                try:
                    return self.async_result.get(timeout=1)
                except Exception as e:
                    return None, f"Subprocess Error: {str(e)}"

            # 2. Check whether process pool is healthy
            # If all worker processes in pool disappeared or pool was terminated
            if not self._is_pool_healthy():
                return None, "Subprocess crashed hard (Segmentation Fault/OOM)"

            # 3. Check for timeout
            if (time.time() - start_time) > actual_timeout:
                return None, "Task timeout (Possible deadlock in C++ layer)"

            # 4. Check for external cancellation signal
            if app_cfg.exit_soft:
                return None, "Task interrupted by user"

            # 5. Event-driven wait to avoid polling lag without losing health/cancel checks
            wait_fn = getattr(self.async_result, "wait", None)
            if callable(wait_fn):
                wait_fn(timeout=0.2)
            else:
                time.sleep(0.2)

    def wait(self, timeout=None):
        wait_fn = getattr(self.async_result, "wait", None)
        if callable(wait_fn):
            return wait_fn(timeout=timeout)
        return self.done()

    def _is_pool_healthy(self):
        """Check whether worker processes in pool are still alive."""
        try:
            # Private attribute _pool contains all worker Process objects
            # While accessing private attributes has risk, in Python 3.10+ this is the most direct health check
            workers = getattr(self._pool, '_pool', [])
            if not workers:
                return False
            # As long as one worker process is alive, pool is considered functional
            return any(w.is_alive() for w in workers)
        except:
            return False

    def done(self):
        return self.async_result.ready()


# ==========================================
# Global singleton manager
# ==========================================

class GlobalProcessManager:
    _executor_cpu = None
    _executor_gpu = None

    @classmethod
    def get_cpu_process_nums(cls):
        cpu_count = int(os.cpu_count() or 1)
        try:
            man_set = int(float(settings.get('process_max', 0)))
        except (ValueError, TypeError):
            man_set = 0
        if man_set > 0:
            # Minimum 1
            return int(max(min(man_set, 8, cpu_count), 1))

        import psutil
        mem = psutil.virtual_memory()
        # Maximum 8 processes, minimum 1
        return int(max(min((int(mem.available / (1024 ** 3)) // 4), 8, cpu_count), 1))

    @classmethod
    def get_gpu_process_nums(cls):
        cpu_count = int(os.cpu_count() or 1)
        try:
            process_max_gpu = int(float(settings.get('process_max_gpu', 0)))
        except (TypeError, ValueError):
            process_max_gpu = 0

        # Manually specified GPU process count takes highest precedence (e.g. single card with large VRAM running multiple tasks)
        if process_max_gpu > 0:
            # Minimum 1
            return int(max(min(process_max_gpu, 8, cpu_count), 1))

        # No GPU or multi-GPU disabled: start only 1 GPU process
        if app_cfg.NVIDIA_GPU_NUMS < 1 or not bool(settings.get('multi_gpus', False)):
            return 1
        # Minimum 1
        return int(max(min(app_cfg.NVIDIA_GPU_NUMS, 8, cpu_count), 1))

    @classmethod
    def get_executor_cpu(cls):
        if cls._executor_cpu is None:
            ctx = multiprocessing.get_context('spawn')
            max_workers = cls.get_cpu_process_nums()
            logger.debug(f'CPU process pool: {max_workers=}')
            cls._executor_cpu = ctx.Pool(
                processes=int(max_workers),
                maxtasksperchild=1  # <--- Terminate CPU worker after each task to thoroughly release physical memory
            )
        return cls._executor_cpu

    @classmethod
    def get_executor_gpu(cls):
        """
        max_workers set to 1, meaning only one AI task runs at a time.
        """
        if cls._executor_gpu is None:
            ctx = multiprocessing.get_context('spawn')
            max_workers = cls.get_gpu_process_nums()
            logger.debug(f'GPU process pool: {max_workers=}')
            cls._executor_gpu = ctx.Pool(
                processes=int(max_workers),
                maxtasksperchild=1
            )

        return cls._executor_gpu

    @classmethod
    def submit_task_cpu(cls, func, **kwargs):
        _executor = cls.get_executor_cpu()
        # Record error log using error_callback
        async_result = _executor.apply_async(
            _task_worker_wrapper,
            args=(func, kwargs),
            error_callback=lambda e: logger.error(f"CPU process pool callback exception: {e}")
        )
        return AsyncResultFutureWrapper(async_result, _executor)

    @classmethod
    def submit_task_gpu(cls, func, **kwargs):
        _executor = cls.get_executor_gpu()
        async_result = _executor.apply_async(
            _task_worker_wrapper,
            args=(func, kwargs),
            error_callback=lambda e: logger.error(f"GPU进程池回调异常: {e}")
        )

        return AsyncResultFutureWrapper(async_result, _executor)

    @classmethod
    def shutdown(cls):
        if cls._executor_cpu:
            cls._executor_cpu.close()
            cls._executor_cpu.join()
            cls._executor_cpu = None
        if cls._executor_gpu:
            cls._executor_gpu.close()
            cls._executor_gpu.join()
            cls._executor_gpu = None
