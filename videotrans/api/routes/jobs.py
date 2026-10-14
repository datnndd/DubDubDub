# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from videotrans.configure import config as runtime_config
from videotrans.configure.config import TEMP_DIR
from videotrans.core.edit_asset_store import EDIT_ASSET_STORE
from videotrans.core.job_manager import ActiveJobError, JOBS
from videotrans.core.media_store import MEDIA, MediaRecord
from videotrans.core.job_store import (
    get_job as db_get_job,
    list_jobs as db_list_jobs,
    events_since,
)
from videotrans.core.project_store import update_project

router = APIRouter()


class JobCreateRequest(BaseModel):
    mediaId: Optional[str] = Field(None, alias="media_id")
    options: Optional[dict[str, Any]] = None
    jobType: Optional[str] = Field(None, alias="job_type")
    projectId: Optional[str] = Field(None, alias="project_id")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


@router.get("/api/jobs")
async def list_jobs_handler(
    request: Request,
    status: Optional[str] = None,
    project_id: Optional[str] = None,
    projectId: Optional[str] = None,
    limit: int = 100,
) -> JSONResponse:
    pid = project_id or projectId or request.query_params.get("project_id") or request.query_params.get("projectId")
    jobs = db_list_jobs(status=status, project_id=pid, limit=limit)
    return JSONResponse({"jobs": jobs})


@router.post("/api/jobs", status_code=202)
@router.post("/api/render", status_code=202)
@router.post("/api/export", status_code=202)
async def create_job_handler(
    payload: JobCreateRequest,
    request: Request,
) -> JSONResponse:
    media_id = str(payload.mediaId or "")
    options = payload.options
    if not isinstance(options, dict):
        raise HTTPException(status_code=400, detail="Job options are required")

    media_store = getattr(request.app.state, "media_store", None) or MEDIA
    media = media_store.get(media_id)
    default_job_type = "render" if request.url.path in {"/api/render", "/api/export"} else "full"
    job_type = str(payload.jobType or options.get("jobType") or default_job_type).lower()

    project_id = str(payload.projectId or options.get("projectId") or options.get("project_id") or "") or None
    if project_id:
        options["projectId"] = project_id

    if (media is None or not media.path.is_file()) and project_id:
        try:
            from videotrans.core.project_store import get_project
            proj = get_project(project_id)
            if proj:
                p_path = proj.get("media_path")
                if p_path and Path(p_path).is_file():
                    p_file = Path(p_path)
                    media = media_store.inspect(
                        p_file,
                        filename=proj.get("name") or p_file.name,
                        media_id=media_id or proj.get("media_id"),
                    )
        except Exception:
            pass

    if job_type == "translation" and (media is None or not media.path.is_file()):
        fallback_path = Path(TEMP_DIR) / f"webui-trans-{uuid.uuid4().hex}.mp4"
        fallback_path.touch(exist_ok=True)
        media_id = media_id or f"trans-mock-{uuid.uuid4().hex}"
        media = MediaRecord(media_id, fallback_path, fallback_path.name, 0, {"time": 1000})
    elif media is None or not media.path.is_file():
        raise HTTPException(status_code=400, detail="Select and inspect a media file before starting")

    edit_asset_store = getattr(request.app.state, "edit_asset_store", None) or EDIT_ASSET_STORE
    for option_key, path_key in (("backgroundAudioId", "backgroundMusicPath"), ("thumbnailId", "thumbnailPath")):
        asset_id = str(options.get(option_key) or "")
        if not asset_id:
            continue
        asset_path = edit_asset_store.get(asset_id)
        if asset_path is None or not asset_path.is_file():
            raise HTTPException(status_code=400, detail=f"Unknown or expired {option_key}")
        options[path_key] = asset_path.resolve().as_posix()

    try:
        task_params_builder = getattr(request.app.state, "task_params_builder", None)
        params = task_params_builder(media.path, options, job_type=job_type, project_id=project_id)

        if job_type not in {"render", "translation"}:
            asr_validator = getattr(request.app.state, "asr_validator", None)
            if asr_validator:
                asr_validator(params["recogn_type"], getattr(request.app.state, "settings_store", None))

        if job_type not in {"asr", "render"}:
            translation_validator = getattr(request.app.state, "translation_validator", None)
            if translation_validator:
                translation_validator(params["translate_type"], getattr(request.app.state, "settings_store", None))

        gpu_initializer = getattr(request.app.state, "gpu_initializer", None)
        if gpu_initializer:
            gpu_initializer()

        manager = getattr(request.app.state, "job_manager", None) or JOBS
        job = manager.submit(params, media_id=media.id, job_type=job_type, project_id=project_id)
        if project_id:
            try:
                update_project(project_id, status="processing")
            except Exception:
                pass
    except ActiveJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(job.snapshot(), status_code=202)


@router.get("/api/jobs/{job_id}")
async def job_handler(job_id: str, request: Request) -> JSONResponse:
    manager = getattr(request.app.state, "job_manager", None) or JOBS
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job.snapshot())


@router.get("/api/jobs/{job_id}/transcript")
@router.get("/api/jobs/{job_id}/segments")
async def job_transcript_handler(job_id: str, request: Request) -> JSONResponse:
    manager = getattr(request.app.state, "job_manager", None) or JOBS
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse({
        "segments": list(job.segments),
        "transcriptOptions": getattr(job, "transcript_options", None),
    })


@router.post("/api/jobs/{job_id}/cancel")
async def cancel_job_handler(job_id: str, request: Request) -> JSONResponse:
    manager = getattr(request.app.state, "job_manager", None) or JOBS
    job = manager.cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job.snapshot())


@router.get("/api/jobs/{job_id}/outputs/{index}")
async def output_handler(job_id: str, index: int, request: Request) -> FileResponse:
    manager = getattr(request.app.state, "job_manager", None) or JOBS
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        output = job.outputs[index]
    except (ValueError, IndexError):
        raise HTTPException(status_code=404, detail="Output not found")
    if not output.is_file():
        raise HTTPException(status_code=404, detail="Output no longer exists")
    return FileResponse(output, filename=output.name)


@router.get("/api/jobs/{job_id}/stream")
async def job_stream_handler(
    job_id: str,
    request: Request,
    after_seq: int = Query(0),
) -> StreamingResponse:
    manager = getattr(request.app.state, "job_manager", None) or JOBS
    job = manager.get(job_id)
    db_job = db_get_job(job_id)
    if not job and not db_job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        past_events = events_since(job_id, after_seq=after_seq)
        last_seq = after_seq
        for ev in past_events:
            seq = ev["seq"]
            payload = ev["payload"]
            raw = payload if isinstance(payload, str) else json.dumps(payload)
            yield f"id: {seq}\nevent: message\ndata: {raw}\n\n"
            last_seq = max(last_seq, seq)

        cur_job_mem = manager.get(job_id)
        cur_job_db = db_get_job(job_id)
        status = (cur_job_mem.status if cur_job_mem else (cur_job_db.get("status") if cur_job_db else None)) or "unknown"

        if status in {"succeeded", "failed", "cancelled"}:
            close_payload = json.dumps({"status": status, "terminal": True, "seq": last_seq})
            yield f"id: {last_seq + 1}\nevent: done\ndata: {close_payload}\n\n"
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        if cur_job_mem:
            cur_job_mem.subscribe(queue, loop)

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    seq, payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    is_terminal = isinstance(payload, dict) and payload.get("status") in {"succeeded", "failed", "cancelled"}
                    if seq > last_seq or (last_seq == 0 and seq >= 0):
                        raw_data = json.dumps(payload) if isinstance(payload, dict) else str(payload)
                        yield f"id: {seq}\nevent: message\ndata: {raw_data}\n\n"
                        last_seq = max(last_seq, seq)
                    if is_terminal:
                        break
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    check_db = db_get_job(job_id)
                    check_mem = manager.get(job_id)
                    check_status = (check_mem.status if check_mem else (check_db.get("status") if check_db else None))
                    if check_status in {"succeeded", "failed", "cancelled"}:
                        break
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            if cur_job_mem:
                cur_job_mem.unsubscribe(queue)

        check_done_db = db_get_job(job_id)
        final_mem = manager.get(job_id)
        final_status = (final_mem.status if final_mem else (check_done_db.get("status") if check_done_db else None)) or status
        if final_status in {"succeeded", "failed", "cancelled"}:
            close_payload = json.dumps({"status": final_status, "terminal": True, "seq": last_seq})
            yield f"id: {last_seq + 1}\nevent: done\ndata: {close_payload}\n\n"

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


def register_routes(app) -> None:
    app.include_router(router)
