"""Execute one background job and persist its lifecycle."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from videotrans.configure import config as runtime_config
from videotrans.core.job_store import append_event, mark_cancelled, mark_done, mark_failed, mark_running
from videotrans.core.proc_registry import set_current_job_id
from videotrans.core.project_store import update_project
from videotrans.task.orchestrator import EventKind, TaskEvent, TaskRequest, TaskStatus

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass(frozen=True)
class JobRunners:
    full: Callable
    asr: Callable
    translation: Callable

    def for_type(self, job_type: str) -> Callable:
        if job_type == "asr":
            return self.asr
        if job_type == "translation":
            return self.translation
        return self.full


def _update_project(project_id: str | None, **fields: Any) -> None:
    if not project_id:
        return
    try:
        update_project(project_id, **fields)
    except Exception:
        pass


def sync_project_for_terminal_status(
    project_id: str | None,
    status: str,
    job_type: str = "full",
) -> None:
    if not project_id:
        return
    try:
        if status == "cancelled":
            update_project(project_id, status="paused")
        elif status == "failed":
            update_project(project_id, status="failed")
        elif status == "succeeded":
            target_stage = 2 if job_type == "asr" else (3 if job_type == "translation" else 4)
            update_project(project_id, stage=target_stage, status="completed")
    except Exception:
        pass


def _notify_subscribers(subs: list[tuple[Any, Any]], seq: int, payload: dict[str, Any]) -> None:
    for q, loop in subs:
        try:
            loop.call_soon_threadsafe(q.put_nowait, (seq, payload))
        except Exception:
            pass


def execute_job(job: Any, params: dict[str, Any], runners: JobRunners) -> None:
    """Run and persist a job; registry cleanup remains the manager's responsibility."""
    set_current_job_id(job.id)
    deferred_succeeded: list[TaskEvent] = []
    terminal_sent = False
    tx_lock = getattr(job, "_tx_lock", None) or contextlib.nullcontext()

    def _event_sink(event: TaskEvent) -> None:
        nonlocal terminal_sent
        if event.kind == EventKind.SUCCEEDED:
            deferred_succeeded.append(event)
            return
        accepted = job.accept(event)
        if event.kind in {EventKind.FAILED, EventKind.CANCELLED} and accepted is not False:
            terminal_sent = True

    try:
        with tx_lock:
            with job._lock:
                if job.status in TERMINAL_STATUSES:
                    return
            try:
                mark_running(job.id)
            except Exception:
                pass
            _update_project(job.project_id, status="processing")

        result = runners.for_type(job.job_type)(TaskRequest(params), _event_sink, job.token)
        subs: list[tuple[Any, Any]] = []
        seq = 0
        payload: dict[str, Any] = {}

        with tx_lock:
            with job._lock:
                already_terminal = job.status in TERMINAL_STATUSES
                cur_status = job.status
                cur_stage = job.stage
                cur_progress = job.progress
                cur_message = job.message
                cur_error = job.error
                cur_outputs = job.outputs
                cur_segments = job.segments
                cur_transcript_opts = getattr(job, "transcript_options", None)
                cur_asr_dur = getattr(job, "asr_duration", None)
                fallback_seq = min(len(job.events) + 1, 200)

            if result is None or not hasattr(result, "status"):
                if already_terminal:
                    if terminal_sent and cur_status == TaskStatus.FAILED.value:
                        try:
                            mark_failed(job.id, error=cur_error or "Job runner returned an invalid result")
                        except Exception:
                            pass
                    sync_project_for_terminal_status(job.project_id, cur_status, job.job_type)
                    subs = []
                else:
                    err_msg = "Job runner returned an invalid result"
                    try:
                        mark_failed(job.id, error=err_msg)
                    except Exception:
                        pass
                    sync_project_for_terminal_status(job.project_id, "failed", job.job_type)
                    if not terminal_sent:
                        payload = {
                            "kind": EventKind.FAILED.value,
                            "stage": cur_stage,
                            "message": err_msg,
                            "progress": cur_progress,
                            "status": TaskStatus.FAILED.value,
                            "error": err_msg,
                            "details": {},
                        }
                        try:
                            seq = append_event(job.id, payload)
                        except Exception:
                            seq = fallback_seq
                    with job._lock:
                        job.status = TaskStatus.FAILED.value
                        job.error = err_msg
                        job.message = err_msg
                        if not terminal_sent:
                            job.events.append({
                                "kind": EventKind.FAILED.value,
                                "stage": cur_stage,
                                "message": err_msg,
                                "progress": cur_progress,
                                "details": {},
                            })
                            job.events[:] = job.events[-200:]
                            subs = list(job._subscribers)
                        else:
                            subs = []
            elif already_terminal and (not terminal_sent or result.status == TaskStatus.SUCCEEDED or cur_status != result.status.value):
                # Preserve existing terminal state (e.g. external cancel() won the race)
                sync_project_for_terminal_status(job.project_id, cur_status, job.job_type)
                subs = []
            else:
                deferred_event = deferred_succeeded[-1] if deferred_succeeded else None
                resolved_outputs = cur_outputs
                resolved_segments = cur_segments
                resolved_transcript_opts = cur_transcript_opts
                resolved_asr_dur = cur_asr_dur

                if not already_terminal:
                    result_outputs = tuple(getattr(result, "outputs", ()) or ())
                    if not result_outputs and deferred_event and deferred_event.details:
                        raw_outputs = deferred_event.details.get("outputs")
                        if isinstance(raw_outputs, (list, tuple)):
                            result_outputs = tuple(Path(p) for p in raw_outputs if p)
                    if result_outputs:
                        resolved_outputs = result_outputs

                    result_segments = getattr(result, "segments", ()) or ()
                    if result_segments:
                        resolved_segments = tuple(result_segments)
                    elif (
                        deferred_event
                        and deferred_event.details
                        and isinstance(deferred_event.details.get("segments"), (list, tuple))
                    ):
                        resolved_segments = tuple(deferred_event.details["segments"])

                    result_transcript_opts = getattr(result, "transcript_options", None)
                    if result_transcript_opts is not None:
                        resolved_transcript_opts = result_transcript_opts
                    elif (
                        deferred_event
                        and deferred_event.details
                        and deferred_event.details.get("transcript_options") is not None
                    ):
                        resolved_transcript_opts = deferred_event.details["transcript_options"]

                    result_asr_dur = getattr(result, "asr_duration", None)
                    if result_asr_dur is not None:
                        resolved_asr_dur = result_asr_dur
                    elif (
                        deferred_event
                        and deferred_event.details
                        and deferred_event.details.get("asr_duration") is not None
                    ):
                        try:
                            resolved_asr_dur = float(deferred_event.details["asr_duration"])
                        except (ValueError, TypeError):
                            pass

                if result.status == TaskStatus.SUCCEEDED:
                    next_stage = (
                        deferred_event.stage
                        if deferred_event is not None and deferred_event.stage is not None
                        else cur_stage
                    )
                    if deferred_event is not None:
                        msg = deferred_event.message or "Processing complete"
                        event_details = dict(deferred_event.details)
                    else:
                        msg = "Processing complete"
                        event_details = {}

                    if resolved_outputs and not event_details.get("outputs"):
                        event_details["outputs"] = [str(p) for p in resolved_outputs]
                    if resolved_segments and not event_details.get("segments"):
                        event_details["segments"] = list(resolved_segments)
                    if resolved_transcript_opts is not None and event_details.get("transcript_options") is None:
                        event_details["transcript_options"] = resolved_transcript_opts
                    if resolved_asr_dur is not None and event_details.get("asr_duration") is None:
                        event_details["asr_duration"] = resolved_asr_dur

                    payload = {
                        "kind": EventKind.SUCCEEDED.value,
                        "stage": next_stage,
                        "message": msg,
                        "progress": 100.0,
                        "status": "succeeded",
                        "error": cur_error,
                        "details": event_details,
                    }
                    try:
                        seq = append_event(job.id, payload)
                    except Exception:
                        seq = fallback_seq
                    try:
                        mark_done(job.id, message=msg)
                    except Exception:
                        pass
                    sync_project_for_terminal_status(job.project_id, "succeeded", job.job_type)

                    with job._lock:
                        job.outputs = resolved_outputs
                        job.segments = resolved_segments
                        job.transcript_options = resolved_transcript_opts
                        job.asr_duration = resolved_asr_dur
                        job.stage = next_stage
                        job.message = msg
                        job.progress = 100.0
                        job.status = "succeeded"
                        job.events.append({
                            "kind": EventKind.SUCCEEDED.value,
                            "stage": next_stage,
                            "message": msg,
                            "progress": 100.0,
                            "details": event_details,
                        })
                        job.events[:] = job.events[-200:]
                        subs = list(job._subscribers)
                elif result.status == TaskStatus.CANCELLED:
                    if already_terminal:
                        if terminal_sent and cur_status == TaskStatus.CANCELLED.value:
                            try:
                                mark_cancelled(job.id, message=cur_message or "Processing cancelled")
                            except Exception:
                                pass
                        sync_project_for_terminal_status(job.project_id, cur_status, job.job_type)
                        subs = []
                    else:
                        cancel_msg = "Processing cancelled"
                        try:
                            mark_cancelled(job.id, message=cancel_msg)
                        except Exception:
                            pass
                        sync_project_for_terminal_status(job.project_id, "cancelled", job.job_type)
                        if not terminal_sent:
                            payload = {
                                "kind": EventKind.CANCELLED.value,
                                "stage": cur_stage,
                                "message": cancel_msg,
                                "progress": cur_progress,
                                "status": "cancelled",
                                "error": cur_error,
                                "details": {},
                            }
                            try:
                                seq = append_event(job.id, payload)
                            except Exception:
                                seq = fallback_seq
                        with job._lock:
                            job.status = "cancelled"
                            job.message = cancel_msg
                            if not terminal_sent:
                                job.events.append({
                                    "kind": EventKind.CANCELLED.value,
                                    "stage": cur_stage,
                                    "message": cancel_msg,
                                    "progress": cur_progress,
                                    "details": {},
                                })
                                job.events[:] = job.events[-200:]
                                subs = list(job._subscribers)
                else:
                    if already_terminal:
                        if terminal_sent and cur_status == TaskStatus.FAILED.value:
                            try:
                                mark_failed(job.id, error=cur_error or "Processing failed")
                            except Exception:
                                pass
                        sync_project_for_terminal_status(job.project_id, cur_status, job.job_type)
                        subs = []
                    else:
                        err_msg = result.failure.message if getattr(result, "failure", None) else "Processing failed"
                        try:
                            mark_failed(job.id, error=err_msg)
                        except Exception:
                            pass
                        sync_project_for_terminal_status(job.project_id, "failed", job.job_type)
                        if not terminal_sent:
                            payload = {
                                "kind": EventKind.FAILED.value,
                                "stage": cur_stage,
                                "message": err_msg,
                                "progress": cur_progress,
                                "status": "failed",
                                "error": err_msg,
                                "details": {},
                            }
                            try:
                                seq = append_event(job.id, payload)
                            except Exception:
                                seq = fallback_seq
                        with job._lock:
                            job.status = "failed"
                            job.error = err_msg
                            job.message = err_msg
                            if not terminal_sent:
                                job.events.append({
                                    "kind": EventKind.FAILED.value,
                                    "stage": cur_stage,
                                    "message": err_msg,
                                    "progress": cur_progress,
                                    "details": {},
                                })
                                job.events[:] = job.events[-200:]
                                subs = list(job._subscribers)

        if subs:
            _notify_subscribers(subs, seq, payload)
    except Exception as exc:
        runtime_config.logger.exception("Unhandled exception in job execution %s: %s", job.id, exc, exc_info=True)
        subs = []
        seq = 0
        payload = {}
        with tx_lock:
            with job._lock:
                already_terminal = job.status in TERMINAL_STATUSES
                cur_status = job.status
                cur_stage = job.stage
                cur_progress = job.progress
                cur_error = job.error
                fallback_seq = min(len(job.events) + 1, 200)
            if already_terminal:
                if terminal_sent and cur_status == TaskStatus.FAILED.value:
                    try:
                        mark_failed(job.id, error=cur_error or "Internal job execution error")
                    except Exception:
                        pass
                sync_project_for_terminal_status(job.project_id, cur_status, job.job_type)
                subs = []
            else:
                failure_error = str(exc) or "Internal job execution error"
                failure_message = f"Processing failed: {failure_error}"
                if not terminal_sent:
                    payload = {
                        "kind": EventKind.FAILED.value,
                        "stage": cur_stage,
                        "message": failure_message,
                        "progress": cur_progress,
                        "status": TaskStatus.FAILED.value,
                        "error": failure_error,
                        "details": {},
                    }
                    try:
                        seq = append_event(job.id, payload)
                    except Exception:
                        seq = fallback_seq
                try:
                    mark_failed(job.id, error=failure_error)
                except Exception:
                    pass
                sync_project_for_terminal_status(job.project_id, "failed", job.job_type)
                with job._lock:
                    job.status = TaskStatus.FAILED.value
                    job.error = failure_error
                    job.message = failure_message
                    if not terminal_sent:
                        job.events.append({
                            "kind": EventKind.FAILED.value,
                            "stage": cur_stage,
                            "message": failure_message,
                            "progress": cur_progress,
                            "details": {},
                        })
                        job.events[:] = job.events[-200:]
                        subs = list(job._subscribers)
        if subs:
            _notify_subscribers(subs, seq, payload)
    finally:
        set_current_job_id(None)
