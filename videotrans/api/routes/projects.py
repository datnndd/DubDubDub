import asyncio
import os
import math
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
import wave
import zipfile
from pathlib import Path
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from videotrans.core.project_store import (
    _segment_srt_timestamp,
    create_project,
    get_project,
    get_project_dir,
    get_parent_project_dir,
    get_projects_root,
    get_or_create_date_parent_project,
    list_subprojects,
    list_projects,
    update_project,
    update_project_state,
    delete_project,
    find_project_by_audio_hash,
)
from videotrans.util.segment_ops import parse_timestamp
from videotrans.core.job_store import list_jobs as list_project_jobs
from videotrans.core.stage_reset import BaselineMissingError, reset_project_stage
from videotrans.core.content_hash import compute_content_hash
from videotrans.core.media_store import MEDIA
from videotrans.core import voice_store

router = APIRouter()


class ProjectCreateRequest(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = "Untitled Project"
    parentId: Optional[str] = Field(None, alias="parent_id")
    mediaId: Optional[str] = Field(None, alias="media_id")
    mediaPath: Optional[str] = Field(None, alias="media_path")
    duration: Optional[float] = 0.0
    stage: Optional[int] = 1
    status: Optional[str] = "pending"
    state: Optional[dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = None
    parentId: Optional[str] = Field(None, alias="parent_id")
    mediaId: Optional[str] = Field(None, alias="media_id")
    mediaPath: Optional[str] = Field(None, alias="media_path")
    duration: Optional[float] = None
    stage: Optional[int] = None
    status: Optional[str] = None
    state: Optional[dict[str, Any]] = None
    merge: Optional[bool] = True

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class BatchCreateItem(BaseModel):
    name: Optional[str] = "Untitled Video"
    mediaId: Optional[str] = Field(None, alias="media_id")
    mediaPath: Optional[str] = Field(None, alias="media_path")
    duration: Optional[float] = 0.0

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class BatchCreateProjectsRequest(BaseModel):
    items: list[BatchCreateItem]
    date: Optional[str] = None
    parentId: Optional[str] = Field(None, alias="parent_id")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


@router.get("/api/projects")
async def list_projects_handler(
    limit: int = Query(100),
    parentId: Optional[str] = Query(None, alias="parent_id"),
) -> JSONResponse:
    # Ensure today's date parent project exists so it is always available for grouping
    get_or_create_date_parent_project()
    projects = list_projects(limit=limit, parent_id=parentId)
    return JSONResponse({"projects": projects})


@router.get("/api/projects/today-parent")
async def get_today_parent_project_handler() -> JSONResponse:
    parent = get_or_create_date_parent_project()
    subprojects = list_subprojects(parent["id"])
    return JSONResponse({"parent": parent, "subprojects": subprojects})


@router.post("/api/projects/batch-create", status_code=201)
async def batch_create_projects_handler(
    payload: BatchCreateProjectsRequest,
    request: Request,
) -> JSONResponse:
    parent_id = payload.parentId
    if parent_id:
        parent = get_project(parent_id)
        if not parent:
            parent = get_or_create_date_parent_project(payload.date)
    else:
        parent = get_or_create_date_parent_project(payload.date)

    media_store = getattr(request.app.state, "media_store", None) or MEDIA
    created_subprojects = []

    for item in payload.items:
        media_id = str(item.mediaId or "") or None
        media = media_store.get(media_id) if media_id else None
        media_path = str(media.path) if media else (str(item.mediaPath or "") or None)
        name = str(item.name or "Untitled Video").strip() or "Untitled Video"

        duration = 0.0
        if media and media.info.get("time"):
            try:
                duration = float(media.info["time"]) / 1000.0
            except (ValueError, TypeError):
                pass
        elif item.duration is not None:
            try:
                duration = float(item.duration)
            except (ValueError, TypeError):
                pass

        audio_hash = ""
        if media and media.path.is_file():
            try:
                audio_hash = compute_content_hash(media.path)
            except Exception:
                pass

        proj = create_project(
            name=name,
            parent_id=parent["id"],
            media_id=media_id,
            media_path=media_path,
            duration=duration,
            stage=1,
            status="pending",
            audio_hash=audio_hash,
            state={"parent_id": parent["id"]},
        )
        created_subprojects.append(proj)

    return JSONResponse({
        "ok": True,
        "parent": parent,
        "projects": created_subprojects,
    }, status_code=201)


@router.get("/api/projects/{parent_id}/subprojects")
async def list_subprojects_handler(parent_id: str) -> JSONResponse:
    subprojects = list_subprojects(parent_id)
    return JSONResponse({"subprojects": subprojects})


@router.post("/api/projects", status_code=201)
async def create_project_handler(
    payload: ProjectCreateRequest,
    request: Request,
) -> JSONResponse:
    media_id = str(payload.mediaId or "") or None
    name = str(payload.name or "Untitled Project").strip() or "Untitled Project"
    media_store = getattr(request.app.state, "media_store", None) or MEDIA
    media = media_store.get(media_id) if media_id else None
    media_path = str(media.path) if media else (str(payload.mediaPath or "") or None)

    duration = 0.0
    if media and media.info.get("time"):
        try:
            duration = float(media.info["time"]) / 1000.0
        except (ValueError, TypeError):
            pass
    elif payload.duration is not None:
        try:
            duration = float(payload.duration)
        except (ValueError, TypeError):
            pass

    audio_hash = ""
    if media and media.path.is_file():
        try:
            audio_hash = compute_content_hash(media.path)
        except Exception:
            pass

    stage = 1
    if payload.stage is not None:
        try:
            stage = int(payload.stage)
        except (ValueError, TypeError):
            pass

    status = str(payload.status or "pending")
    state = payload.state if isinstance(payload.state, dict) else {}

    # Auto-assign to today's date parent project if parentId not provided and not a parent project itself
    parent_id = payload.parentId
    if not parent_id and not state.get("is_parent"):
        parent = get_or_create_date_parent_project()
        parent_id = parent["id"]
        state["parent_id"] = parent_id
    elif parent_id:
        state["parent_id"] = parent_id

    if audio_hash and not state.get("segments"):
        cached_proj = find_project_by_audio_hash(audio_hash)
        if cached_proj and cached_proj.get("state", {}).get("segments"):
            state["segments"] = cached_proj["state"]["segments"]
            if cached_proj.get("stage", 1) > stage:
                stage = cached_proj["stage"]

    project = create_project(
        project_id=str(payload.id or "") or None,
        name=name,
        parent_id=parent_id,
        media_id=media_id,
        media_path=media_path,
        duration=duration,
        stage=stage,
        status=status,
        audio_hash=audio_hash,
        state=state,
    )

    return JSONResponse(project, status_code=201)


@router.get("/api/projects/{project_id}")
async def get_project_handler(project_id: str) -> JSONResponse:
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return JSONResponse(project)


@router.post("/api/projects/{project_id}/stages/{stage}/reset")
async def reset_stage_handler(project_id: str, stage: int) -> JSONResponse:
    if stage not in (1, 2, 3, 4):
        raise HTTPException(status_code=400, detail="Stage must be 1, 2, 3, or 4")
    if list_project_jobs(project_id=project_id, status="active", limit=1):
        raise HTTPException(status_code=409, detail="Wait for the active project job to finish before resetting a stage")
    try:
        return JSONResponse(reset_project_stage(project_id, stage))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BaselineMissingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/api/projects/{project_id}")
@router.put("/api/projects/{project_id}/state")
@router.post("/api/projects/{project_id}/state")
async def update_project_handler(
    project_id: str,
    payload: ProjectUpdateRequest,
    request: Request,
) -> JSONResponse:
    kwargs: dict[str, Any] = {}
    if payload.state is not None and isinstance(payload.state, dict):
        kwargs["state"] = payload.state
    if payload.stage is not None:
        try:
            kwargs["stage"] = int(payload.stage)
        except (ValueError, TypeError):
            pass
    if payload.status is not None:
        kwargs["status"] = str(payload.status)
    if payload.name is not None:
        kwargs["name"] = str(payload.name)
    if payload.duration is not None:
        try:
            kwargs["duration"] = float(payload.duration)
        except (ValueError, TypeError):
            pass

    media_store = getattr(request.app.state, "media_store", None) or MEDIA
    media_id = str(payload.mediaId or "") or None
    if not media_id and isinstance(payload.state, dict):
        backend_state = payload.state.get("backend")
        if isinstance(backend_state, dict):
            media_id = str(backend_state.get("mediaId") or "") or None

    if media_id:
        kwargs["media_id"] = media_id
        media = media_store.get(media_id)
        if media:
            kwargs["media_path"] = str(media.path)
            if payload.duration is None and media.info.get("time"):
                try:
                    kwargs["duration"] = float(media.info["time"]) / 1000.0
                except (ValueError, TypeError):
                    pass
    elif payload.mediaPath:
        kwargs["media_path"] = str(payload.mediaPath)

    if payload.state is not None and isinstance(payload.state, dict):
        try:
            project = update_project_state(
                project_id,
                state_dict=payload.state,
                stage=kwargs.get("stage"),
                status=kwargs.get("status"),
                merge=True if payload.merge is None else bool(payload.merge),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid project state: {exc}") from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not save project subtitles: {exc}") from exc
        extra_keys = ("name", "duration", "media_id", "media_path")
        if project and any(k in kwargs for k in extra_keys):
            extra = {k: kwargs[k] for k in extra_keys if k in kwargs}
            project = update_project(project_id, **extra)
    else:
        project = update_project(project_id, **kwargs)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return JSONResponse(project)


class BulkDeleteProjectsRequest(BaseModel):
    ids: list[str]

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class DubbingAssemblyRequest(BaseModel):
    segments: list[dict[str, Any]]


def _resample_mono_float32(data: Any, orig_sr: int, target_sr: int) -> Any:
    import numpy as np

    if orig_sr == target_sr or data.size == 0:
        return np.asarray(data, dtype=np.float32)
    try:
        import soxr
        resampled = soxr.resample(data, orig_sr, target_sr)
        return np.asarray(resampled, dtype=np.float32)
    except Exception:
        from scipy.signal import resample_poly
        g = math.gcd(target_sr, orig_sr)
        resampled = resample_poly(data, target_sr // g, orig_sr // g)
        return np.asarray(resampled, dtype=np.float32)


_assemble_locks_guard = threading.Lock()
_assemble_project_locks: dict[str, threading.Lock] = {}


def _get_assemble_lock(project_id: str) -> threading.Lock:
    with _assemble_locks_guard:
        lock = _assemble_project_locks.get(project_id)
        if lock is None:
            lock = threading.Lock()
            _assemble_project_locks[project_id] = lock
        return lock


_MAX_DUBBING_DURATION_S = 86_400.0


def _parse_dubbing_timestamp(segment: dict[str, Any], sec_key: str, time_key: str) -> float:
    raw_sec = segment.get(sec_key)
    if raw_sec is not None:
        if isinstance(raw_sec, bool):
            raise ValueError(f"invalid {sec_key}")
        try:
            val = float(raw_sec)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid {sec_key}") from exc
    else:
        raw_time = segment.get(time_key)
        if raw_time is None or isinstance(raw_time, bool):
            raise ValueError(f"missing {sec_key}/{time_key}")
        if isinstance(raw_time, (int, float)):
            try:
                val = float(raw_time)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid {time_key}") from exc
        elif isinstance(raw_time, str):
            cleaned = raw_time.strip().replace(",", ".")
            if not cleaned:
                raise ValueError(f"empty {time_key}")
            if ":" in cleaned:
                parts = cleaned.split(":")
                if len(parts) not in (2, 3):
                    raise ValueError(f"invalid {time_key}")
                try:
                    if len(parts) == 3:
                        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                        if h < 0 or m < 0 or s < 0 or not math.isfinite(s):
                            raise ValueError(f"invalid {time_key}")
                    else:
                        m, s = int(parts[0]), float(parts[1])
                        if m < 0 or s < 0 or not math.isfinite(s):
                            raise ValueError(f"invalid {time_key}")
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"invalid {time_key}") from exc
                val = parse_timestamp(raw_time)
            else:
                try:
                    val = float(cleaned)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"invalid {time_key}") from exc
        else:
            raise ValueError(f"invalid {time_key}")

    if not math.isfinite(val) or val < 0 or val > _MAX_DUBBING_DURATION_S:
        raise ValueError(f"{sec_key} out of supported range (0..{_MAX_DUBBING_DURATION_S}s)")
    return val


def _assemble_dubbing_sync(
    project_id: str,
    project_duration_s: Any,
    segments: list[dict[str, Any]],
) -> None:
    import numpy as np
    import soundfile as sf

    TARGET_SR = 48000
    SAMPLES_PER_MS = TARGET_SR // 1000  # 48
    MAX_TOTAL_SAMPLES = int(_MAX_DUBBING_DURATION_S * TARGET_SR)

    if isinstance(project_duration_s, bool):
        raise HTTPException(status_code=400, detail="Invalid project duration")
    try:
        proj_dur_val = float(project_duration_s or 0.0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid project duration: {exc}") from exc
    if not math.isfinite(proj_dur_val) or proj_dur_val < 0 or proj_dur_val > _MAX_DUBBING_DURATION_S:
        raise HTTPException(status_code=400, detail=f"Project duration exceeds supported bounds (0..{_MAX_DUBBING_DURATION_S}s)")

    parsed_segments: list[dict[str, Any]] = []
    duration_ms = max(0, int(round(proj_dur_val * 1000)))
    for segment in segments:
        if not str(segment.get("targetText") or segment.get("sourceText") or "").strip():
            continue
        segment_id = segment.get("id", "unknown")
        preview_id = str(segment.get("previewAudioId") or "")
        if not re.fullmatch(r"prev_[0-9a-f]{12}", preview_id):
            raise HTTPException(status_code=400, detail=f"Generate a voice preview for segment {segment_id} before opening Stage 4")
        try:
            start = _parse_dubbing_timestamp(segment, "startSec", "startTime")
            end = _parse_dubbing_timestamp(segment, "endSec", "endTime")
            if end <= start:
                raise ValueError("invalid timing")
            audio_path = voice_store.get_preview_audio_path(f"{preview_id}.wav")
            if not audio_path.is_file() or audio_path.stat().st_size == 0:
                raise ValueError("preview file missing")
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if data.size == 0 or sr <= 0:
                raise ValueError("empty preview audio")
            if data.ndim > 1:
                data = np.mean(data, axis=1, dtype=np.float32)
            else:
                data = np.asarray(data, dtype=np.float32)
            if not np.all(np.isfinite(data)):
                data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
            if int(sr) != TARGET_SR:
                data = _resample_mono_float32(data, int(sr), TARGET_SR)
        except (TypeError, ValueError, OverflowError, OSError, sf.SoundFileError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=f"Preview audio or timing is invalid for segment {segment_id}: {exc}") from exc

        start_ms = int(round(start * 1000))
        end_ms = int(round(end * 1000))
        parsed_segments.append({
            "segment_id": segment_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "clip": data,
        })

    if not parsed_segments:
        raise HTTPException(status_code=400, detail="Generate dubbing previews before opening Stage 4")

    parsed_segments.sort(key=lambda s: s["start_ms"])

    clips: list[tuple[int, np.ndarray]] = []
    total_samples = duration_ms * SAMPLES_PER_MS
    for idx, seg in enumerate(parsed_segments):
        start_ms = seg["start_ms"]
        end_ms = seg["end_ms"]
        clip: np.ndarray = seg["clip"]
        clip_ms = max(1, int(round(len(clip) * 1000 / TARGET_SR)))
        next_start_ms = parsed_segments[idx + 1]["start_ms"] if idx + 1 < len(parsed_segments) else None

        if next_start_ms is not None:
            max_allowed_end_ms = max(start_ms + 1, next_start_ms - 30)
        else:
            max_allowed_end_ms = max(duration_ms, end_ms, start_ms + clip_ms)

        available_samples = max(SAMPLES_PER_MS, (max_allowed_end_ms - start_ms) * SAMPLES_PER_MS)

        if len(clip) > available_samples:
            clip = clip[:available_samples].copy()
            fade_in_samples = min(20 * SAMPLES_PER_MS, max(1, len(clip) // 2))
            fade_out_samples = min(30 * SAMPLES_PER_MS, max(1, len(clip) // 2))
        else:
            clip = clip.copy()
            fade_in_samples = min(20 * SAMPLES_PER_MS, max(1, len(clip) // 2))
            fade_out_samples = fade_in_samples

        if fade_in_samples > 1:
            clip[:fade_in_samples] *= np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
        if fade_out_samples > 1:
            clip[-fade_out_samples:] *= np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)

        start_sample = start_ms * SAMPLES_PER_MS
        clips.append((start_sample, clip))
        total_samples = max(total_samples, start_sample + len(clip))

    total_samples = max(1, total_samples)
    if total_samples > MAX_TOTAL_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Dubbing duration exceeds supported bounds (0..{_MAX_DUBBING_DURATION_S}s)",
        )
    try:
        merged = np.zeros(total_samples, dtype=np.float32)
        for start_sample, clip in clips:
            merged[start_sample : start_sample + len(clip)] += clip
        np.clip(merged, -1.0, 1.0, out=merged)
    except (OverflowError, MemoryError) as exc:
        raise HTTPException(status_code=400, detail=f"Dubbing duration is too large to assemble: {exc}") from exc

    output = get_project_dir(project_id) / "dubbing" / "voiceover_merged.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(f"voiceover_merged.{uuid.uuid4().hex[:8]}.tmp.wav")
    with _get_assemble_lock(project_id):
        try:
            sf.write(str(pending), merged, TARGET_SR, subtype="PCM_16", format="WAV")
            for attempt in range(6):
                try:
                    os.replace(pending, output)
                    break
                except PermissionError:
                    if attempt >= 5:
                        raise
                    time.sleep(0.01 * (attempt + 1))
        finally:
            pending.unlink(missing_ok=True)


@router.post("/api/projects/{project_id}/dubbing/assemble")
async def assemble_project_dubbing_handler(project_id: str, payload: DubbingAssemblyRequest) -> JSONResponse:
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not payload.segments:
        raise HTTPException(status_code=400, detail="Generate dubbing previews before opening Stage 4")

    await asyncio.to_thread(_assemble_dubbing_sync, project_id, project.get("duration"), payload.segments)
    return JSONResponse({"ok": True, "audio_url": f"/api/projects/{project_id}/dubbing/audio"})


@router.get("/api/projects/{project_id}/dubbing/audio")
async def get_project_dubbing_audio_handler(project_id: str) -> FileResponse:
    if not get_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    audio_path = get_project_dir(project_id) / "dubbing" / "voiceover_merged.wav"
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="Assembled dubbing audio is not available")
    return FileResponse(audio_path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.post("/api/projects/bulk-delete")
@router.delete("/api/projects/bulk-delete")
async def bulk_delete_projects_handler(payload: BulkDeleteProjectsRequest) -> JSONResponse:
    raw_ids = payload.ids or []
    deleted_ids = []
    for pid in raw_ids:
        if delete_project(str(pid), delete_files=True):
            deleted_ids.append(str(pid))
    return JSONResponse({"ok": True, "deleted": deleted_ids, "count": len(deleted_ids)})


@router.delete("/api/projects/{project_id}")
async def delete_project_handler(project_id: str) -> JSONResponse:
    deleted = delete_project(project_id, delete_files=True)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return JSONResponse({"ok": True, "id": project_id})


@router.post("/api/projects/{project_id}/resume")
async def resume_project_handler(project_id: str) -> JSONResponse:
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return JSONResponse({"ok": True, "project": project})


def _format_srt_time(sec: float) -> str:
    total_ms = max(0, int(round(float(sec) * 1000)))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt(segments: list[dict], field: str = "targetText") -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        if not isinstance(seg, dict):
            continue
        text = str(seg.get(field) or (seg.get("sourceText") if field == "targetText" else seg.get("targetText")) or "").strip()
        start = _segment_srt_timestamp(seg, "startSec", "startTime")
        if "endSec" in seg or seg.get("endTime") is not None:
            end = _segment_srt_timestamp(seg, "endSec", "endTime")
        else:
            end = _segment_srt_timestamp({"endSec": parse_timestamp(start) + 1.0}, "endSec", "endTime")
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip() + "\n"


def _create_fallback_wav(path: Path, duration_sec: float = 1.0) -> None:
    if isinstance(duration_sec, bool):
        raise ValueError("Invalid project duration")
    dur = float(duration_sec)
    if not math.isfinite(dur) or dur < 0 or dur > _MAX_DUBBING_DURATION_S:
        raise ValueError(f"Fallback WAV duration out of supported bounds (0..{_MAX_DUBBING_DURATION_S}s)")
    sample_rate = 44100
    num_frames = int(max(0.5, dur) * sample_rate)
    chunk_frames = min(num_frames, sample_rate * 10)
    zero_chunk = b"\x00" * (chunk_frames * 2)
    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        remaining = num_frames
        while remaining > 0:
            take = min(remaining, chunk_frames)
            wav_file.writeframes(zero_chunk if take == chunk_frames else b"\x00" * (take * 2))
            remaining -= take


def _prepare_capcut_assets(project_id: str) -> dict[str, Path]:
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    parent_id = project.get("parent_id")
    if parent_id:
        parent = get_project(parent_id)
        parent_name = (parent.get("name") if parent else None) or parent_id
        exports_dir = get_parent_project_dir(parent_name)
        safe_stem = re.sub(r"[^\w.-]+", "-", Path(project.get("name") or project_id).stem, flags=re.UNICODE).strip("-")
        prefix = f"{safe_stem}_" if safe_stem else ""
    else:
        proj_dir = get_project_dir(project_id)
        exports_dir = proj_dir / "exports"
        prefix = ""
    exports_dir.mkdir(parents=True, exist_ok=True)

    state = project.get("state") or {}
    segments = state.get("segments") or []
    if not isinstance(segments, list):
        raise HTTPException(status_code=400, detail="Invalid project segments")

    try:
        # 1. subtitles_edited.srt
        edited_srt_path = exports_dir / f"{prefix}subtitles_edited.srt"
        edited_content = _build_srt(segments, field="targetText")
        edited_srt_path.write_text(edited_content, encoding="utf-8")

        # 2. subtitles_target.srt
        target_srt_path = exports_dir / f"{prefix}subtitles_target.srt"
        target_content = _build_srt(segments, field="rawTranslation" if any(isinstance(s, dict) and "rawTranslation" in s for s in segments) else "targetText")
        target_srt_path.write_text(target_content, encoding="utf-8")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid segment timestamp: {exc}") from exc

    # 3. voiceover_merged.wav
    voiceover_path = exports_dir / f"{prefix}voiceover_merged.wav"
    found_audio = None
    proj_dir = get_project_dir(project_id)
    dubbing_dir = proj_dir / "dubbing"
    for cand_name in ["voiceover_merged.wav", "target.wav", "lastend.wav", "dubbed.wav"]:
        cand_dub = dubbing_dir / cand_name
        if cand_dub.is_file() and cand_dub.stat().st_size > 0:
            found_audio = cand_dub
            break
        cand = exports_dir / cand_name
        if cand.is_file() and cand.stat().st_size > 0:
            found_audio = cand
            break

    if found_audio and found_audio.is_file() and found_audio.resolve() != voiceover_path.resolve():
        try:
            shutil.copy2(found_audio, voiceover_path)
        except Exception:
            voiceover_path = found_audio
    elif not voiceover_path.is_file() or voiceover_path.stat().st_size == 0:
        try:
            raw_dur = project.get("duration")
            if isinstance(raw_dur, bool):
                raise ValueError("Invalid project duration")
            dur_val = float(raw_dur if raw_dur not in (None, 0, 0.0) else 1.0)
            _create_fallback_wav(voiceover_path, duration_sec=dur_val)
        except (TypeError, ValueError, OverflowError, MemoryError) as exc:
            voiceover_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Invalid project duration for audio export: {exc}") from exc

    # 4. bundle.zip
    zip_path = exports_dir / f"{prefix}capcut_export_{project_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(edited_srt_path, f"{prefix}subtitles_edited.srt")
        zf.write(target_srt_path, f"{prefix}subtitles_target.srt")
        zf.write(voiceover_path, f"{prefix}voiceover_merged.wav")

    return {
        "subtitles_edited.srt": edited_srt_path,
        "subtitles_target.srt": target_srt_path,
        "voiceover_merged.wav": voiceover_path,
        "bundle.zip": zip_path,
    }


@router.get("/api/projects/{project_id}/export-capcut/{asset_name}")
async def get_capcut_asset_handler(project_id: str, asset_name: str):
    clean_name = asset_name.lower().strip()
    if clean_name in {"video.mp4", "video"}:
        raise HTTPException(status_code=404, detail=f"Unknown asset {asset_name}")
    assets = _prepare_capcut_assets(project_id)
    if clean_name in {"subtitles_edited.srt", "edited.srt"}:
        target = assets["subtitles_edited.srt"]
        media_type = "text/plain; charset=utf-8"
        download_name = target.name
    elif clean_name in {"subtitles_target.srt", "target.srt"}:
        target = assets["subtitles_target.srt"]
        media_type = "text/plain; charset=utf-8"
        download_name = target.name
    elif clean_name in {"voiceover_merged.wav", "audio.wav", "audio"}:
        target = assets["voiceover_merged.wav"]
        media_type = "audio/wav"
        download_name = target.name
    elif clean_name in {"bundle.zip", "capcut_bundle.zip", "all.zip", "zip"}:
        target = assets["bundle.zip"]
        media_type = "application/zip"
        download_name = target.name
    else:
        raise HTTPException(status_code=404, detail=f"Unknown asset {asset_name}")

    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Asset {asset_name} could not be generated")
    return FileResponse(target, filename=download_name, media_type=media_type)


@router.post("/api/projects/{project_id}/open-folder")
async def open_project_folder_handler(project_id: str) -> JSONResponse:
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    parent_id = project.get("parent_id")
    if parent_id:
        parent = get_project(parent_id)
        parent_name = (parent.get("name") if parent else None) or parent_id
        target_dir = get_parent_project_dir(parent_name)
    else:
        proj_dir = get_project_dir(project_id)
        exports_dir = proj_dir / "exports"
        target_dir = exports_dir if exports_dir.is_dir() else proj_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(os, "startfile"):
            os.startfile(str(target_dir))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target_dir)])
        else:
            subprocess.Popen(["xdg-open", str(target_dir)])
    except Exception:
        pass
    return JSONResponse({"ok": True, "path": str(target_dir.resolve().as_posix())})


def register_routes(app) -> None:
    app.include_router(router)
