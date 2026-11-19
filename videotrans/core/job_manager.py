# -*- coding: utf-8 -*-
"""Background job lifecycle, execution management, and status subscribers."""

from __future__ import annotations

import contextlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from videotrans.configure import config as runtime_config
from videotrans.task.orchestrator import (
    CancellationToken,
    EventKind,
    TaskEvent,
    TaskRequest,
    run,
    run_staged_asr,
    run_staged_translation,
)
from videotrans.core.job_store import (
    create_job as db_create_job,
    get_job as db_get_job,
    update_job as db_update_job,
    mark_cancelled as db_mark_cancelled,
    append_event,
    events_since,
)
from videotrans.core.project_store import update_project
from videotrans.core.proc_registry import (
    kill_job_procs,
)
from videotrans.core.job_executor import (
    JobRunners,
    TERMINAL_STATUSES,
    execute_job,
    sync_project_for_terminal_status,
)


class ActiveJobError(RuntimeError):
    """Raised when the same ingested media already has an active job."""


@dataclass
class JobRecord:
    id: str
    token: CancellationToken
    status: str = "queued"
    stage: str | None = None
    progress: float | None = None
    message: str = "Queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    outputs: tuple[Path, ...] = ()
    segments: tuple[dict[str, Any], ...] = ()
    asr_duration: float | None = None
    transcript_options: dict[str, Any] | None = None
    error: str | None = None
    media_id: str | None = None
    job_type: str = "full"
    project_id: str | None = None
    _subscribers: list[tuple[Any, Any]] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _tx_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def subscribe(self, queue: Any, loop: Any) -> None:
        with self._lock:
            self._subscribers.append((queue, loop))

    def unsubscribe(self, queue: Any) -> None:
        with self._lock:
            self._subscribers = [sub for sub in self._subscribers if sub[0] is not queue]

    def accept(self, event: TaskEvent) -> bool:
        item = {
            "kind": event.kind.value,
            "stage": event.stage,
            "message": event.message,
            "progress": event.progress,
            "details": dict(event.details),
        }
        subs: list[tuple[Any, Any]] = []
        seq = 0
        payload: dict[str, Any] = {}
        with self._tx_lock:
            with self._lock:
                if self.status in TERMINAL_STATUSES:
                    return False

                next_status = self.status
                next_stage = event.stage
                next_progress = event.progress if event.progress is not None else self.progress
                next_message = event.message if event.message else self.message
                next_error = self.error

                if event.kind in {EventKind.RUNNING, EventKind.STAGE_STARTED}:
                    next_status = "running"
                elif event.kind == EventKind.FAILED:
                    next_status = "failed"
                    next_error = event.message or "Processing failed"
                elif event.kind == EventKind.CANCELLED:
                    next_status = "cancelled"
                    next_message = event.message or "Processing cancelled"
                elif event.kind == EventKind.SUCCEEDED:
                    next_status = "succeeded"
                    next_progress = 100.0

                is_terminal = next_status in TERMINAL_STATUSES

                new_outputs = None
                new_asr_duration = None
                new_segments = None
                new_transcript_options = None
                if event.details:
                    if "outputs" in event.details:
                        raw_outputs = event.details["outputs"]
                        if isinstance(raw_outputs, (list, tuple)):
                            new_outputs = tuple(Path(p) for p in raw_outputs if p)
                    if event.details.get("source_type") == "asr_timing" and "duration" in event.details:
                        try:
                            new_asr_duration = float(event.details["duration"])
                        except (ValueError, TypeError):
                            pass
                    elif "asr_duration" in event.details and event.details.get("asr_duration") is not None:
                        try:
                            new_asr_duration = float(event.details["asr_duration"])
                        except (ValueError, TypeError):
                            pass
                    if "segments" in event.details:
                        raw_segs = event.details["segments"]
                        if isinstance(raw_segs, (list, tuple)):
                            new_segments = tuple(raw_segs)
                    if "transcript_options" in event.details and event.details.get("transcript_options") is not None:
                        new_transcript_options = event.details["transcript_options"]

                payload = {
                    "kind": event.kind.value,
                    "stage": next_stage,
                    "message": next_message,
                    "progress": next_progress,
                    "status": next_status,
                    "error": next_error,
                    "details": dict(event.details),
                }
                fallback_seq = min(len(self.events) + 1, 200)

            try:
                seq = append_event(self.id, payload)
                db_update_job(
                    self.id,
                    status=(next_status if next_status != "queued" else None),
                    stage=next_stage,
                    progress=next_progress,
                    message=next_message,
                    error=next_error,
                    finished=is_terminal,
                )
            except Exception:
                seq = fallback_seq

            if is_terminal:
                sync_project_for_terminal_status(self.project_id, next_status, self.job_type)

            with self._lock:
                self.events.append(item)
                self.events[:] = self.events[-200:]
                self.stage = next_stage
                self.progress = next_progress
                self.message = next_message
                self.error = next_error
                if new_outputs is not None:
                    self.outputs = new_outputs
                if new_asr_duration is not None:
                    self.asr_duration = new_asr_duration
                if new_segments is not None:
                    self.segments = new_segments
                if new_transcript_options is not None:
                    self.transcript_options = new_transcript_options
                self.status = next_status
                subs = list(self._subscribers)

        for q, loop in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, (seq, payload))
            except Exception:
                pass
        return True

    def update(
        self,
        status: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
        stage: str | None = None,
    ) -> None:
        subs: list[tuple[Any, Any]] = []
        seq = 0
        payload: dict[str, Any] = {}
        with self._tx_lock:
            with self._lock:
                if self.status in TERMINAL_STATUSES:
                    return
                next_status = status if status is not None else self.status
                next_progress = progress if progress is not None else self.progress
                next_message = message if message is not None else self.message
                next_error = error if error is not None else self.error
                next_stage = stage if stage is not None else self.stage
                is_terminal = next_status in TERMINAL_STATUSES

                payload = {
                    "kind": "update",
                    "stage": next_stage,
                    "message": next_message,
                    "progress": next_progress,
                    "status": next_status,
                    "error": next_error,
                    "details": {},
                }
                fallback_seq = min(len(self.events) + 1, 200)

            try:
                seq = append_event(self.id, payload)
                db_update_job(
                    self.id,
                    status=(next_status if next_status != "queued" else None),
                    stage=next_stage,
                    progress=next_progress,
                    message=next_message,
                    error=next_error,
                    finished=is_terminal,
                )
            except Exception:
                seq = fallback_seq

            if is_terminal:
                sync_project_for_terminal_status(self.project_id, next_status, self.job_type)

            with self._lock:
                self.status = next_status
                self.progress = next_progress
                self.message = next_message
                self.error = next_error
                self.stage = next_stage
                self.events.append(payload)
                self.events[:] = self.events[-200:]
                subs = list(self._subscribers)

        for q, loop in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, (seq, payload))
            except Exception:
                pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "jobType": self.job_type,
                "projectId": self.project_id,
                "stage": self.stage,
                "progress": self.progress,
                "message": self.message,
                "events": list(self.events),
                "error": self.error,
                "outputs": [
                    {"name": path.name, "url": f"/api/jobs/{self.id}/outputs/{index}"}
                    for index, path in enumerate(self.outputs)
                ],
                "segments": list(self.segments),
                "transcriptOptions": self.transcript_options,
                "asrDuration": self.asr_duration,
            }


class JobManager:
    def __init__(
        self,
        runner: Callable = run,
        asr_runner: Callable | None = None,
        translation_runner: Callable | None = None,
    ) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._active_by_media: dict[str, str] = {}
        self._lock = threading.Lock()
        self._runner = runner
        self._asr_runner = asr_runner or run_staged_asr
        self._translation_runner = translation_runner or run_staged_translation
        self._runners = JobRunners(self._runner, self._asr_runner, self._translation_runner)

    def submit(
        self,
        params: dict[str, Any],
        *,
        media_id: str,
        job_type: str = "full",
        project_id: str | None = None,
    ) -> JobRecord:
        with self._lock:
            active_id = self._active_by_media.get(media_id)
            if active_id:
                raise ActiveJobError(f"Media already running in job {active_id}")
            job = JobRecord(
                uuid.uuid4().hex,
                CancellationToken(),
                media_id=media_id,
                job_type=job_type,
                project_id=project_id,
            )
            self._jobs[job.id] = job
            self._active_by_media[media_id] = job.id
            try:
                db_create_job(
                    job.id,
                    type=job_type,
                    project_id=project_id,
                    meta={"media_id": media_id},
                )
            except Exception as e:
                runtime_config.logger.warning("Failed to create job in db: %s", e)
        if project_id:
            with job._tx_lock:
                with job._lock:
                    is_terminal = job.status in TERMINAL_STATUSES
                if not is_terminal:
                    try:
                        update_project(project_id, status="processing")
                    except Exception:
                        pass
        threading.Thread(target=self._execute, args=(job, params), daemon=True).start()
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
        if record is not None:
            return record
        try:
            db_job = db_get_job(job_id)
            if db_job:
                meta = db_job.get("meta") or {}
                media_id = meta.get("media_id") if isinstance(meta, dict) else None
                rec = JobRecord(
                    id=db_job["id"],
                    token=CancellationToken(),
                    status=db_job.get("status", "failed"),
                    stage=db_job.get("stage"),
                    progress=db_job.get("progress"),
                    message=db_job.get("message", ""),
                    error=db_job.get("error"),
                    media_id=media_id,
                    project_id=db_job.get("project_id"),
                    job_type=db_job.get("type", "full"),
                )
                try:
                    raw_events = events_since(job_id, after_seq=0, limit=1000)
                    parsed_events: list[dict[str, Any]] = []
                    for ev in raw_events:
                        raw_payload = ev.get("payload")
                        if isinstance(raw_payload, str):
                            try:
                                payload = json.loads(raw_payload)
                            except Exception:
                                continue
                        elif isinstance(raw_payload, dict):
                            payload = raw_payload
                        else:
                            continue
                        parsed_events.append(payload)
                        details = payload.get("details")
                        if isinstance(details, dict):
                            if "outputs" in details:
                                raw_outputs = details["outputs"]
                                if isinstance(raw_outputs, (list, tuple)):
                                    rec.outputs = tuple(Path(p) for p in raw_outputs if p)
                            if details.get("source_type") == "asr_timing" and "duration" in details:
                                try:
                                    rec.asr_duration = float(details["duration"])
                                except (ValueError, TypeError):
                                    pass
                            elif "asr_duration" in details and details.get("asr_duration") is not None:
                                try:
                                    rec.asr_duration = float(details["asr_duration"])
                                except (ValueError, TypeError):
                                    pass
                            if "segments" in details:
                                raw_segs = details["segments"]
                                if isinstance(raw_segs, (list, tuple)):
                                    rec.segments = tuple(raw_segs)
                            if "transcript_options" in details and details.get("transcript_options") is not None:
                                rec.transcript_options = details["transcript_options"]
                    rec.events = parsed_events[-200:]
                except Exception:
                    pass
                return rec
        except Exception:
            pass
        return None

    def cancel(self, job_id: str) -> JobRecord | None:
        job = self.get(job_id)
        if not job:
            return None
        subs: list[tuple[Any, Any]] = []
        seq = 0
        payload: dict[str, Any] = {}
        did_cancel = False
        tx_lock = getattr(job, "_tx_lock", None) or contextlib.nullcontext()
        with tx_lock:
            with job._lock:
                if job.status in TERMINAL_STATUSES:
                    return job
                cur_stage = job.stage
                cur_progress = job.progress
            job.token.cancel()
            cancel_msg = "Processing cancelled"
            payload = {
                "kind": "cancelled",
                "status": "cancelled",
                "message": cancel_msg,
                "progress": cur_progress,
            }
            try:
                db_mark_cancelled(job.id)
            except Exception:
                pass
            try:
                seq = append_event(job.id, payload)
            except Exception:
                seq = 0
            sync_project_for_terminal_status(job.project_id, "cancelled", job.job_type)
            with job._lock:
                job.status = "cancelled"
                job.message = cancel_msg
                job.events.append({
                    "kind": "cancelled",
                    "stage": cur_stage,
                    "message": cancel_msg,
                    "progress": cur_progress,
                    "details": {},
                })
                job.events[:] = job.events[-200:]
                subs = list(job._subscribers)
                did_cancel = True
        if did_cancel:
            kill_job_procs(job_id)
            for q, loop in subs:
                try:
                    loop.call_soon_threadsafe(q.put_nowait, (seq, payload))
                except Exception:
                    pass
        return job

    def _execute(self, job: JobRecord, params: dict[str, Any]) -> None:
        try:
            execute_job(job, params, self._runners)
        finally:
            with self._lock:
                if self._active_by_media.get(job.media_id) == job.id:
                    self._active_by_media.pop(job.media_id, None)


def run_prepare_review(
    request: TaskRequest,
    event_sink: Callable[[TaskEvent], None] | None = None,
    cancellation_token: CancellationToken | None = None,
    *,
    runner: Callable = run,
):
    """Run only the stages needed to produce a transcript for Review Transcript."""
    return runner(
        request,
        event_sink,
        cancellation_token,
        stop_after_stage="diariz",
    )



JOBS = JobManager(runner=run_prepare_review, translation_runner=run_staged_translation)
