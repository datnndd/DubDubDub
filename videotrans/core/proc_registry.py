# -*- coding: utf-8 -*-
"""In-flight subprocess tracking and cross-platform process tree cancellation."""
from __future__ import annotations

import logging
import subprocess
import threading
from contextvars import ContextVar
from typing import Optional

import psutil

logger = logging.getLogger("videotrans.proc_registry")

_active_procs: dict[str, list[subprocess.Popen]] = {}
_active_procs_lock = threading.Lock()

_current_job_id: ContextVar[Optional[str]] = ContextVar("current_job_id", default=None)


def get_current_job_id() -> Optional[str]:
    """Retrieve the active job ID for the current async/thread execution context."""
    return _current_job_id.get()


def set_current_job_id(job_id: Optional[str]) -> None:
    """Set the active job ID for the current async/thread execution context."""
    _current_job_id.set(job_id)


def register_proc(job_id: str, proc: subprocess.Popen) -> None:
    """Track an in-flight subprocess under a job id."""
    if not job_id:
        return
    with _active_procs_lock:
        _active_procs.setdefault(job_id, []).append(proc)


def unregister_proc(job_id: str, proc: subprocess.Popen) -> None:
    """Unregister a finished subprocess."""
    if not job_id:
        return
    with _active_procs_lock:
        lst = _active_procs.get(job_id)
        if lst and proc in lst:
            lst.remove(proc)
        if lst is not None and not lst:
            _active_procs.pop(job_id, None)


def has_active_procs(job_id: str) -> bool:
    """Return True if there are active subprocesses registered for this job."""
    with _active_procs_lock:
        return bool(_active_procs.get(job_id))


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a process and all its spawned child processes cross-platform."""
    if proc.poll() is not None:
        return
    try:
        for child in psutil.Process(proc.pid).children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    finally:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def kill_job_procs(job_id: str) -> None:
    """Kill every subprocess still running under a given job id. Idempotent."""
    with _active_procs_lock:
        procs = list(_active_procs.get(job_id, []))

    for proc in procs:
        try:
            kill_process_tree(proc)
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning("Failed to kill subprocess for job %s: %s", job_id, e)

    with _active_procs_lock:
        _active_procs.pop(job_id, None)
