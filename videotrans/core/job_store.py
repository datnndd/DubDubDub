# -*- coding: utf-8 -*-
"""Job-metadata and SSE event persistence across server restarts."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from videotrans.core.db import db_conn

logger = logging.getLogger("videotrans.job_store")

_EVENT_CAP_PER_JOB = 500


def _row_to_job(row: Any) -> dict[str, Any]:
    d = dict(row)
    if "meta_json" in d:
        try:
            d["meta"] = json.loads(d.pop("meta_json") or "{}")
        except Exception:
            d["meta"] = {}
    return d


def create_job(
    job_id: str,
    *,
    type: str,
    project_id: Optional[str] = None,
    stage: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Insert a new job in 'pending' status."""
    now = time.time()
    with db_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs
            (id, type, project_id, status, stage, progress, message, created_at, updated_at, meta_json)
            VALUES (?, ?, ?, 'pending', ?, 0.0, 'Queued', ?, ?, ?)
            """,
            (job_id, type, project_id, stage, now, now, json.dumps(meta or {})),
        )
    return get_job(job_id)  # type: ignore[return-value]


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
    finished: bool = False,
) -> Optional[dict[str, Any]]:
    """Update fields on an existing job."""
    now = time.time()
    updates: dict[str, Any] = {"updated_at": now}
    if status is not None:
        updates["status"] = status
    if stage is not None:
        updates["stage"] = stage
    if progress is not None:
        updates["progress"] = progress
    if message is not None:
        updates["message"] = message
    if error is not None:
        updates["error"] = error
    if finished:
        updates["finished_at"] = now
    if meta is not None:
        updates["meta_json"] = json.dumps(meta)

    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
    values = list(updates.values()) + [job_id]

    with db_conn() as conn:
        conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)

    return get_job(job_id)


def mark_running(job_id: str, stage: Optional[str] = None, message: str = "Processing") -> None:
    update_job(job_id, status="running", stage=stage, message=message)


def mark_done(
    job_id: str,
    *,
    message: str = "Processing complete",
    meta: Optional[dict[str, Any]] = None,
) -> None:
    update_job(job_id, status="succeeded", progress=100.0, message=message, meta=meta, finished=True)


def mark_failed(job_id: str, error: str) -> None:
    update_job(job_id, status="failed", error=error, message=f"Failed: {error}", finished=True)


def mark_cancelled(job_id: str, message: str = "Processing cancelled") -> None:
    update_job(job_id, status="cancelled", message=message, finished=True)


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """Get job record by ID."""
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def list_jobs(
    *,
    status: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Query jobs with optional status or project_id filters."""
    where = []
    params: list[Any] = []
    if status == "active":
        where.append("status IN ('pending', 'running')")
    elif status:
        where.append("status = ?")
        params.append(status)
    if project_id:
        where.append("project_id = ?")
        params.append(project_id)

    sql = "SELECT * FROM jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with db_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_job(r) for r in rows]


def append_event(job_id: str, payload: str | dict[str, Any]) -> int:
    """Persist an SSE event to job_events and return its sequence number.
    
    Caps stored events per job to _EVENT_CAP_PER_JOB rows.
    """
    raw_payload = payload if isinstance(payload, str) else json.dumps(payload)
    now = time.time()
    with db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS s FROM job_events WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        next_seq = int(row["s"]) + 1

        conn.execute(
            "INSERT INTO job_events (job_id, seq, created_at, payload) VALUES (?, ?, ?, ?)",
            (job_id, next_seq, now, raw_payload),
        )

        cnt = conn.execute(
            "SELECT COUNT(*) AS n FROM job_events WHERE job_id = ?",
            (job_id,),
        ).fetchone()["n"]
        if cnt > _EVENT_CAP_PER_JOB:
            conn.execute(
                """
                DELETE FROM job_events WHERE job_id = ? AND seq IN
                (SELECT seq FROM job_events WHERE job_id = ? ORDER BY seq ASC LIMIT ?)
                """,
                (job_id, job_id, cnt - _EVENT_CAP_PER_JOB),
            )
    return next_seq


def events_since(job_id: str, after_seq: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
    """Retrieve sequenced events after after_seq for SSE reconnection."""
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT seq, created_at, payload FROM job_events
            WHERE job_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?
            """,
            (job_id, after_seq, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def sweep_orphans_on_startup() -> int:
    """Mark any 'pending' or 'running' jobs as failed on server startup."""
    msg = "Job was interrupted by a server restart. Re-run from the task's project to continue."
    now = time.time()
    with db_conn() as conn:
        cur = conn.execute(
            """
            UPDATE jobs SET status='failed', updated_at=?, finished_at=?, error=?
            WHERE status IN ('pending', 'running')
            """,
            (now, now, msg),
        )
        n = cur.rowcount
    if n > 0:
        logger.info("Swept %d orphaned in-flight job(s) after restart", n)
    return n
