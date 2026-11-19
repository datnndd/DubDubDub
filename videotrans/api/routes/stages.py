# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import math
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from videotrans.configure import config as runtime_config
from videotrans.configure.excepts import get_msg_from_except
from videotrans.core.media_store import MEDIA
from videotrans.task.orchestrator import (
    CancellationToken,
    TaskRequest,
    TaskStatus,
    run_staged_translation,
)
from videotrans.util.segment_ops import split_segment
from videotrans.api.task_params import _translation_mode
from videotrans.api.ocr_helpers import NormalizedRoi, extract_ocr_segment_text

router = APIRouter()


class TranslationRequest(BaseModel):
    segments: list[Any]
    sourceLanguage: Optional[str] = "zh-cn"
    targetLanguage: Optional[str] = "en"
    translateType: Optional[int] = 0
    translationMode: Optional[Any] = None
    projectId: Optional[str] = Field(None, alias="project_id")
    mediaId: Optional[str] = Field(None, alias="media_id")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class OcrExtractRequest(BaseModel):
    mediaId: str
    startSec: float = 0.0
    endSec: float = 0.0
    roi: Any
    language: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class SegmentSplitRequest(BaseModel):
    segment: dict[str, Any]
    splitTime: float
    cursorPosition: Optional[int] = None
    nextId: Optional[Any] = Field(None, alias="next_id")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


@router.post("/api/translate")
async def translate_handler(
    payload: TranslationRequest,
    request: Request,
) -> JSONResponse:
    raw_segments = payload.segments
    if not isinstance(raw_segments, (list, tuple)):
        raise HTTPException(status_code=400, detail="segments list is required")

    source_lang = str(payload.sourceLanguage or "zh-cn")
    target_lang = str(payload.targetLanguage or "en")
    try:
        translate_type = int(payload.translateType if payload.translateType is not None else 0)
        _, aisendsrt = _translation_mode(payload.translationMode)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    segments = []
    has_non_empty = False
    for s in raw_segments:
        if isinstance(s, dict):
            raw_src = s.get("sourceText") if s.get("sourceText") is not None else s.get("text")
            text = str(raw_src or "").strip()
            item = dict(s)
            item["sourceText"] = text
            if not text:
                item["text"] = ""
            else:
                has_non_empty = True
            segments.append(item)

    if not segments or not has_non_empty:
        return JSONResponse({"ok": True, "segments": segments})

    import shutil
    import uuid
    from pathlib import Path
    from videotrans.configure.config import TEMP_DIR

    media_store = getattr(request.app.state, "media_store", None) or MEDIA
    media_id = getattr(payload, "mediaId", None) or (payload.model_dump().get("mediaId") if hasattr(payload, "model_dump") else None)
    media = media_store.get(str(media_id)) if media_id else None
    fallback_path: Optional[Path] = None
    if media is None or not media.path.is_file():
        fallback_path = Path(TEMP_DIR) / f"webui-trans-{uuid.uuid4().hex}.mp4"
        fallback_path.touch(exist_ok=True)
        media_file = fallback_path
    else:
        media_file = media.path

    job_uuid = uuid.uuid4().hex
    cache_dir = Path(TEMP_DIR) / f"cache-{job_uuid}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_dir = Path(TEMP_DIR) / f"target-{job_uuid}"
    target_dir.mkdir(parents=True, exist_ok=True)

    task_params = {
        "name": media_file.resolve().as_posix(),
        "target_dir": target_dir.resolve().as_posix(),
        "cache_folder": cache_dir.resolve().as_posix(),
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "source_language": source_lang,
        "target_language": target_lang,
        "translate_type": translate_type,
        "aisendsrt": aisendsrt,
        "segments": segments,
        "uuid": job_uuid,
    }
    if payload.projectId:
        task_params["project_id"] = str(payload.projectId)
    job_manager = getattr(request.app.state, "job_manager", None)
    runner = getattr(job_manager, "_translation_runner", None) or run_staged_translation
    token = CancellationToken()
    try:
        result = await asyncio.to_thread(runner, TaskRequest(task_params), None, token)
        if result.status == TaskStatus.FAILED:
            err_msg = result.failure.message if result.failure else "Translation failed"
            raise HTTPException(status_code=400, detail=err_msg)
        return JSONResponse({"ok": True, "segments": list(result.segments)})
    finally:
        if fallback_path is not None:
            fallback_path.unlink(missing_ok=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(target_dir, ignore_errors=True)


@router.post("/api/ocr/extract")
async def ocr_extract_handler(
    payload: OcrExtractRequest,
    request: Request,
) -> JSONResponse:
    media_id = str(payload.mediaId or "")
    media_store = getattr(request.app.state, "media_store", None) or MEDIA
    media = media_store.get(media_id)
    if media is None or not media.path.is_file():
        raise HTTPException(status_code=404, detail="Select and inspect a media file before OCR extraction")

    try:
        start_sec = float(payload.startSec)
        end_sec = float(payload.endSec)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(status_code=400, detail="startSec and endSec must be numeric")

    if not (math.isfinite(start_sec) and math.isfinite(end_sec)) or end_sec <= start_sec or start_sec < 0:
        raise HTTPException(status_code=400, detail="endSec must be finite and strictly greater than startSec >= 0")

    roi = payload.roi
    if isinstance(roi, dict):
        try:
            rx_val = roi.get("x")
            ry_val = roi.get("y")
            rw_val = roi.get("width") if roi.get("width") is not None else roi.get("w")
            rh_val = roi.get("height") if roi.get("height") is not None else roi.get("h")
            rx = float(0 if rx_val is None else rx_val)
            ry = float(0 if ry_val is None else ry_val)
            rw = float(0 if rw_val is None else rw_val)
            rh = float(0 if rh_val is None else rh_val)
            roi = [rx, ry, rw, rh]
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(status_code=400, detail="roi elements must be numeric")
    elif not (isinstance(roi, (list, tuple)) and len(roi) == 4):
        raise HTTPException(status_code=400, detail="roi must be a 4-element list [x, y, w, h] or a dict with x, y, width, height")

    try:
        rx, ry, rw, rh = [float(v) for v in roi]
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(status_code=400, detail="roi elements must be numeric")

    if not all(math.isfinite(v) for v in (rx, ry, rw, rh)) or rw <= 0 or rh <= 0 or rx < 0 or ry < 0 or rx + rw > 1.05 or ry + rh > 1.05:
        raise HTTPException(status_code=400, detail="roi must be finite and normalized (0..1) with positive width and height")

    normalized_roi = NormalizedRoi(rx, ry, rw, rh)
    language = payload.language

    ocr_extractor = getattr(request.app.state, "ocr_extractor", None) or extract_ocr_segment_text
    try:
        try:
            result = await asyncio.to_thread(
                ocr_extractor,
                media.path,
                start_sec,
                end_sec,
                normalized_roi,
                language=language,
            )
        except TypeError:
            try:
                result = await asyncio.to_thread(
                    ocr_extractor,
                    media.path,
                    start_sec,
                    end_sec,
                    normalized_roi,
                    language,
                )
            except TypeError:
                result = await asyncio.to_thread(
                    ocr_extractor,
                    media.path,
                    start_sec,
                    end_sec,
                    normalized_roi,
                )
    except Exception as exc:
        runtime_config.logger.exception("OCR extraction failed", exc_info=True)
        raise HTTPException(status_code=400, detail=f"OCR extraction failed: {get_msg_from_except(exc)}") from exc

    return JSONResponse(result)


@router.post("/api/segments/split")
async def split_segment_handler(
    payload: SegmentSplitRequest,
    request: Request,
) -> JSONResponse:
    segment = payload.segment
    if not isinstance(segment, dict):
        raise HTTPException(status_code=400, detail="segment object is required")

    try:
        split_time = float(payload.splitTime)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="splitTime must be numeric")

    cursor_pos = payload.cursorPosition
    if cursor_pos is not None:
        try:
            cursor_pos = int(cursor_pos)
        except (TypeError, ValueError):
            cursor_pos = None

    next_id = payload.nextId

    try:
        seg1, seg2 = split_segment(segment, split_time, cursor_pos, next_id=next_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse({"ok": True, "segments": [seg1, seg2]})


def register_routes(app) -> None:
    app.include_router(router)
