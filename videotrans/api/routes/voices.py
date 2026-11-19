# -*- coding: utf-8 -*-
"""RESTful routes for custom voice cloning, audio management, and audition preview."""
from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
import tempfile
import uuid
from typing import Any, Optional
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.datastructures import UploadFile

from videotrans import tts
from videotrans.api.catalog import TTS_PROVIDER_ALIASES
from videotrans.core import voice_store
from videotrans.services.audio_normalizer import (
    normalize_reference_audio,
    detect_optimal_speech_segment,
    probe_audio_duration,
)
from videotrans.services.voice_preview import (
    generate_fallback_preview_wav,
    get_path_lock,
    synthesize_voice_preview,
    synthesize_unified_tts_preview,
    synthesize_clone_preview,
    synthesize_tts_batch,
    register_batch_cancel,
    unregister_batch_cancel,
    cancel_batch,
)

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("videotrans.api.voices")
router = APIRouter()


def _reject_nonfinite_constant(val: str) -> Any:
    raise ValueError(f"Non-finite JSON constant not allowed: {val}")


def _validate_finite_json_value(val: Any) -> None:
    if isinstance(val, float):
        if not math.isfinite(val):
            raise ValueError("tuning_params cannot contain NaN or Infinity")
    elif isinstance(val, dict):
        for v in val.values():
            _validate_finite_json_value(v)
    elif isinstance(val, (list, tuple)):
        for item in val:
            _validate_finite_json_value(item)


async def _serve_locked_preview_response(
    path: Path,
    media_type: str,
    headers: Optional[dict[str, str]] = None,
    request: Optional[Request] = None,
) -> Optional[Response]:
    data: Optional[bytes] = None
    async with get_path_lock(path):
        if not path.is_file():
            return None
        st = path.stat()
        if int(st.st_size) <= 0:
            return None
        if int(st.st_size) <= 16 * 1024 * 1024:
            data = await asyncio.to_thread(path.read_bytes)

    if data is None:
        return FileResponse(path, media_type=media_type, headers=headers)

    resp_headers = dict(headers or {})
    resp_headers.setdefault("Accept-Ranges", "bytes")
    range_header = request.headers.get("range") if request is not None else None
    if range_header and range_header.lower().startswith("bytes="):
        spec = range_header[6:].strip()
        if "," not in spec and "-" in spec:
            start_s, end_s = spec.split("-", 1)
            try:
                if start_s == "" and end_s != "":
                    suffix_len = int(end_s)
                    if suffix_len > 0:
                        start = max(0, len(data) - suffix_len)
                        end = len(data) - 1
                    else:
                        start, end = 0, len(data) - 1
                else:
                    start = int(start_s) if start_s else 0
                    end = int(end_s) if end_s else len(data) - 1
                if 0 <= start <= end < len(data):
                    chunk = data[start : end + 1]
                    resp_headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
                    resp_headers["Content-Length"] = str(len(chunk))
                    return Response(content=chunk, status_code=206, media_type=media_type, headers=resp_headers)
                return Response(
                    status_code=416,
                    headers={"Content-Range": f"bytes */{len(data)}", **resp_headers},
                )
            except ValueError:
                pass
    return Response(content=data, media_type=media_type, headers=resp_headers)


class VoiceUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    instruct: Optional[str] = None
    ref_text: Optional[str] = None
    language: Optional[str] = None
    tuning_params: Optional[dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class VoicePreviewRequest(BaseModel):
    text: Optional[str] = None
    language: Optional[str] = None
    force_refresh: Optional[bool] = False

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class UnifiedTTSPreviewRequest(BaseModel):
    text: Optional[str] = None
    voice: Optional[str] = None
    voice_id: Optional[str] = Field(None, alias="voiceId")
    provider: Optional[Any] = None
    ttsType: Optional[Any] = Field(None, alias="tts_type")
    language: Optional[str] = None
    speed: Optional[float] = None
    rate: Optional[str] = None
    pitch: Optional[str] = None
    style: Optional[str] = None
    tuning_params: Optional[dict[str, Any]] = Field(None, alias="tuningParams")
    segment_id: Optional[Any] = Field(None, alias="segmentId")
    force_refresh: Optional[bool] = Field(False, alias="forceRefresh")
    auto_speed: Optional[bool] = Field(False, alias="autoSpeed")
    slot_duration_s: Optional[float] = Field(None, alias="slotDuration")
    max_speed_rate: Optional[float] = Field(1.35, alias="maxSpeedRate")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class BatchTTSSegmentItem(BaseModel):
    segment_id: Optional[Any] = Field(None, alias="segmentId")
    id: Optional[Any] = None
    text: Optional[str] = None
    voice: Optional[str] = None
    voice_id: Optional[str] = Field(None, alias="voiceId")
    provider: Optional[Any] = None
    ttsType: Optional[Any] = Field(None, alias="tts_type")
    language: Optional[str] = None
    speed: Optional[float] = None
    rate: Optional[str] = None
    pitch: Optional[str] = None
    style: Optional[str] = None
    tuning_params: Optional[dict[str, Any]] = Field(None, alias="tuningParams")
    force_refresh: Optional[bool] = Field(False, alias="forceRefresh")
    auto_speed: Optional[bool] = Field(None, alias="autoSpeed")
    slot_duration_s: Optional[float] = Field(None, alias="slotDuration")
    max_speed_rate: Optional[float] = Field(None, alias="maxSpeedRate")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class BatchTTSRequest(BaseModel):
    batch_id: Optional[str] = Field(None, alias="batchId")
    segments: Optional[list[BatchTTSSegmentItem]] = None
    items: Optional[list[BatchTTSSegmentItem]] = None
    provider: Optional[Any] = None
    ttsType: Optional[Any] = Field(None, alias="tts_type")
    voice: Optional[str] = None
    language: Optional[str] = None
    speed: Optional[float] = None
    style: Optional[str] = None
    tuning_params: Optional[dict[str, Any]] = Field(None, alias="tuningParams")
    force_refresh: Optional[bool] = Field(False, alias="forceRefresh")
    auto_speed: Optional[bool] = Field(False, alias="autoSpeed")
    max_speed_rate: Optional[float] = Field(1.35, alias="maxSpeedRate")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class VoiceDesignAssistRequest(BaseModel):
    prompt: str
    base_voice: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


def _parse_provider(raw: Any) -> int:
    """Parse provider integer or alias string; defaults to VieNeu (2) only when omitted."""
    if raw is None:
        return tts.DEFAULT_TTS
    if isinstance(raw, bool):
        raise ValueError(f"Unsupported TTS provider: {raw!r}")
    s = str(raw).strip().lower()
    if not s:
        return tts.DEFAULT_TTS
    if s in TTS_PROVIDER_ALIASES:
        return TTS_PROVIDER_ALIASES[s]
    try:
        val = int(s)
        if 0 <= val < len(tts.TTS_NAME_LIST):
            return val
    except ValueError:
        pass
    raise ValueError(f"Unsupported TTS provider: {raw!r}")


def _parse_provider_or_400(raw: Any) -> int:
    try:
        return _parse_provider(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/custom-voices")
async def list_custom_voices_handler(
    provider: Optional[str] = None,
    active: str = "true",
) -> JSONResponse:
    parsed_provider = _parse_provider_or_400(provider) if provider is not None else None
    active_only = active.strip().lower() not in {"false", "0", "no"}
    voices = voice_store.list_voices(provider=parsed_provider, active_only=active_only)
    return JSONResponse({"voices": voices})


@router.get("/api/custom-voices/{id}")
async def get_custom_voice_handler(id: str) -> JSONResponse:
    voice = voice_store.get_voice(id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice {id} not found")
    return JSONResponse(voice)


@router.post("/api/custom-voices", status_code=201)
async def create_custom_voice_handler(request: Request) -> JSONResponse:
    voice_store.init_voice_dirs()
    fields: dict[str, Any] = {}
    audio_file_obj: Optional[UploadFile] = None
    audio_bytes: Optional[bytes] = None
    audio_filename: str = ""

    content_type = (request.headers.get("content-type") or "").lower()

    if "multipart" in content_type:
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid multipart form: {exc}") from exc

        for key, val in form.items():
            if key in {"audio", "file"}:
                if hasattr(val, "filename") and val.filename:
                    audio_file_obj = val
                    audio_filename = Path(unquote(val.filename)).name
                elif isinstance(val, (bytes, bytearray)):
                    audio_bytes = bytes(val)
                    audio_filename = "sample.wav"
            else:
                fields[key] = str(val)
    elif "json" in content_type:
        try:
            fields = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
        if not isinstance(fields, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object")
    else:
        raise HTTPException(status_code=400, detail="Expected multipart/form-data or application/json")

    raw_name = fields.get("name", "")
    clean_name = " ".join(str(raw_name).split())
    if not clean_name:
        raise HTTPException(status_code=400, detail="Voice name is required")

    provider = _parse_provider_or_400(fields.get("provider"))
    language = str(fields.get("language") or "Auto").strip()
    description = str(fields.get("description") or "").strip()
    kind = str(fields.get("kind") or "clone").strip()
    ref_text = str(fields.get("ref_text") or "").strip()
    instruct = str(fields.get("instruct") or "").strip()
    external_voice_id = str(fields.get("external_voice_id") or "").strip()

    raw_tuning = fields.get("tuning_params") or fields.get("tuning") or {}
    if isinstance(raw_tuning, str):
        raw_tuning_stripped = raw_tuning.strip()
        if not raw_tuning_stripped:
            tuning_params = {}
        else:
            try:
                tuning_params = json.loads(raw_tuning_stripped, parse_constant=_reject_nonfinite_constant)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid tuning_params JSON: {exc}") from exc
            if not isinstance(tuning_params, dict):
                raise HTTPException(status_code=400, detail="tuning_params must be a JSON object")
    elif isinstance(raw_tuning, dict):
        tuning_params = raw_tuning
    else:
        raise HTTPException(status_code=400, detail="tuning_params must be a JSON object")
    try:
        _validate_finite_json_value(tuning_params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    voice_id = f"voice_{uuid.uuid4().hex[:8]}"
    ref_audio_rel = ""

    denoise_flag = str(fields.get("denoise", "true")).strip().lower() not in {"false", "0", "no"}
    tuning_params["denoise"] = denoise_flag

    cut_start_raw = fields.get("cut_start") or fields.get("start_time")
    cut_end_raw = fields.get("cut_end") or fields.get("end_time")
    try:
        start_time = float(cut_start_raw) if cut_start_raw is not None and str(cut_start_raw).strip() != "" else None
    except (ValueError, TypeError):
        start_time = None
    try:
        end_time = float(cut_end_raw) if cut_end_raw is not None and str(cut_end_raw).strip() != "" else None
    except (ValueError, TypeError):
        end_time = None

    if not audio_file_obj and not audio_bytes and fields.get("audio"):
        raw_audio = fields.get("audio")
        if isinstance(raw_audio, str):
            import base64
            if raw_audio.startswith("data:audio"):
                try:
                    _, encoded = raw_audio.split(",", 1)
                    audio_bytes = base64.b64decode(encoded)
                except Exception:
                    pass
            elif len(raw_audio) > 100:
                try:
                    audio_bytes = base64.b64decode(raw_audio)
                except Exception:
                    pass

    if audio_file_obj:
        ext = Path(audio_filename).suffix
        if not ext and getattr(audio_file_obj, "content_type", None):
            ct = str(audio_file_obj.content_type).lower()
            if "mpeg" in ct or "mp3" in ct:
                ext = ".mp3"
            elif "ogg" in ct:
                ext = ".ogg"
            elif "flac" in ct:
                ext = ".flac"
            elif "webm" in ct:
                ext = ".webm"
            elif "mp4" in ct or "m4a" in ct:
                ext = ".m4a"
        ext = ext or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            while chunk := await audio_file_obj.read(4 * 1024 * 1024):
                tmp_file.write(chunk)

        try:
            dest_filename = f"{voice_id}.wav"
            dest_path = voice_store.VOICES_DIR / dest_filename
            normalize_reference_audio(tmp_path, dest_path, provider=provider, start_time=start_time, end_time=end_time)
            ref_audio_rel = dest_filename
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to normalize audio", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Audio normalization failed: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)
    elif audio_bytes:
        ext = Path(audio_filename).suffix
        if not ext and fields.get("content_type"):
            ct = str(fields.get("content_type")).lower()
            if "mpeg" in ct or "mp3" in ct:
                ext = ".mp3"
            elif "ogg" in ct:
                ext = ".ogg"
            elif "flac" in ct:
                ext = ".flac"
            elif "webm" in ct:
                ext = ".webm"
            elif "mp4" in ct or "m4a" in ct:
                ext = ".m4a"
        ext = ext or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(audio_bytes)

        try:
            dest_filename = f"{voice_id}.wav"
            dest_path = voice_store.VOICES_DIR / dest_filename
            normalize_reference_audio(tmp_path, dest_path, provider=provider, start_time=start_time, end_time=end_time)
            ref_audio_rel = dest_filename
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to normalize audio", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Audio normalization failed: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)
    elif kind == "design":
        base_v = external_voice_id or str(tuning_params.get("base_voice") or "")
        if not base_v:
            raise HTTPException(status_code=400, detail="Base voice is required for Voice Design")
        external_voice_id = base_v
    elif provider == tts.ELEVENLABS_TTS and external_voice_id:
        pass
    else:
        raise HTTPException(status_code=400, detail="Audio file is required for voice cloning")

    # For VieNeu voice cloning, pre-cache speaker embedding for low-latency inference
    if ref_audio_rel and provider == tts.VIENEU_TTS:
        dest_path = voice_store.VOICES_DIR / ref_audio_rel
        try:
            from videotrans.services.model_cache import get_cached_vieneu_engine
            engine = get_cached_vieneu_engine(device="auto", backend="auto", max_batch_size=4)
            emb, codes = engine.encode_reference(dest_path, denoise=denoise_flag)
            voice_store.cache_voice_embedding(voice_id, emb, codes)
        except Exception as exc:
            logger.info("VieNeu pre-caching skipped or unavailable: %s", exc)

    # Check if a pre-generated preview audio was provided from preview-clone
    passed_preview = str(fields.get("preview_audio_path") or fields.get("preview_filename") or "").strip()
    preview_audio_rel = ""
    if passed_preview:
        try:
            clean_passed = Path(unquote(passed_preview.split("?")[0])).name
            p_check = voice_store.get_preview_audio_path(clean_passed)
            if p_check.is_file() and p_check.stat().st_size > 0:
                dest_preview = voice_store.PREVIEWS_DIR / f"{voice_id}_preview.wav"
                import shutil
                shutil.copyfile(p_check, dest_preview)
                preview_audio_rel = f"{voice_id}_preview.wav"
        except Exception as exc:
            logger.info("Could not reuse passed preview file: %s", exc)

    created = voice_store.create_voice(
        name=clean_name,
        provider=provider,
        voice_id=voice_id,
        description=description,
        kind=kind,
        language=language,
        ref_audio_path=ref_audio_rel,
        ref_text=ref_text,
        instruct=instruct,
        external_voice_id=external_voice_id,
        tuning_params=tuning_params,
        preview_audio_path=preview_audio_rel,
    )

    # Auto-generate initial preview if not yet present
    if not preview_audio_rel:
        try:
            gen_path = await synthesize_voice_preview(voice_id)
            preview_audio_rel = gen_path.name
            created["preview_audio_path"] = preview_audio_rel
            voice_store.update_voice(voice_id, preview_audio_path=preview_audio_rel)
        except Exception as exc:
            logger.warning("Auto-synthesizing preview on voice creation failed: %s", exc)

    created["preview_url"] = f"/api/custom-voices/{voice_id}/preview/audio"
    created["audio_url"] = f"/api/custom-voices/{voice_id}/audio"

    return JSONResponse(created, status_code=201)


@router.put("/api/custom-voices/{id}")
async def update_custom_voice_handler(
    id: str,
    payload: VoiceUpdateRequest,
    request: Request,
) -> JSONResponse:
    voice = voice_store.get_voice(id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice {id} not found")

    update_dict = payload.model_dump(exclude_unset=True)
    try:
        if "tuning_params" in update_dict and update_dict["tuning_params"] is not None:
            _validate_finite_json_value(update_dict["tuning_params"])
        updated = voice_store.update_voice(id, **update_dict)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not updated:
        raise HTTPException(status_code=404, detail=f"Voice {id} not found")

    return JSONResponse(updated)


@router.delete("/api/custom-voices/{id}")
async def delete_custom_voice_handler(
    id: str,
    hard: str = Query("false"),
) -> JSONResponse:
    voice = voice_store.get_voice(id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice {id} not found")

    hard_delete = hard.strip().lower() in {"true", "1", "yes"}
    success = voice_store.delete_voice(id, hard=hard_delete)
    if not success:
        raise HTTPException(status_code=404, detail=f"Voice {id} not found")

    return JSONResponse({"ok": True, "id": id, "hard": hard_delete})


class BulkDeleteVoicesRequest(BaseModel):
    ids: list[str]
    hard: Optional[bool] = False

    model_config = ConfigDict(populate_by_name=True, extra="allow")


@router.post("/api/custom-voices/bulk-delete")
@router.delete("/api/custom-voices/bulk-delete")
@router.post("/api/voices/bulk-delete")
@router.delete("/api/voices/bulk-delete")
async def bulk_delete_custom_voices_handler(payload: BulkDeleteVoicesRequest) -> JSONResponse:
    raw_ids = payload.ids or []
    deleted_ids = voice_store.bulk_delete_voices(raw_ids, hard=bool(payload.hard))
    return JSONResponse({"ok": True, "deleted": deleted_ids, "count": len(deleted_ids)})



@router.get("/api/custom-voices/{id}/audio")
async def get_custom_voice_audio_handler(id: str) -> FileResponse:
    voice = voice_store.get_voice(id)
    if not voice or not voice.get("ref_audio_path"):
        raise HTTPException(status_code=404, detail="Reference audio not found")

    try:
        audio_path = voice_store.get_voice_audio_path(voice["ref_audio_path"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Reference audio file missing from storage")

    return FileResponse(audio_path, media_type="audio/wav")


@router.post("/api/custom-voices/{id}/preview")
async def create_custom_voice_preview_handler(
    id: str,
    payload: Optional[VoicePreviewRequest] = None,
    request: Request = None,
) -> JSONResponse:
    voice = voice_store.get_voice(id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice {id} not found")

    text = payload.text if payload else None
    language = payload.language if payload else None
    force_refresh = bool(payload.force_refresh) if payload else False

    try:
        preview_path = await synthesize_voice_preview(
            id,
            text=text,
            language=language,
            force_refresh=force_refresh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Preview generation failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {exc}") from exc

    preview_url = f"/api/custom-voices/{id}/preview/audio"
    return JSONResponse({
        "ok": True,
        "voice_id": id,
        "preview_url": preview_url,
        "preview_filename": preview_path.name,
    })


@router.get("/api/custom-voices/{id}/preview/audio")
async def get_custom_voice_preview_audio_handler(id: str, request: Request = None) -> Response:
    voice = voice_store.get_voice(id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice {id} not found")

    raw_preview = str(voice.get("preview_audio_path") or "").strip()
    preview_filename = Path(unquote(raw_preview.split("?")[0])).name if raw_preview else f"{id}_preview.wav"
    try:
        preview_path = voice_store.get_preview_audio_path(preview_filename)
    except ValueError:
        preview_path = voice_store.get_preview_audio_path(f"{id}_preview.wav")

    resp = await _serve_locked_preview_response(preview_path, "audio/wav", request=request)
    if resp is not None:
        return resp

    try:
        preview_path = await synthesize_voice_preview(id)
        resp = await _serve_locked_preview_response(preview_path, "audio/wav", request=request)
        if resp is not None:
            return resp
    except Exception as exc:
        logger.warning("Auto-synthesizing preview audio for voice %s failed: %s", id, exc)
        ref_path = voice.get("ref_audio_path")
        if ref_path:
            try:
                clean_ref = Path(unquote(str(ref_path).split("?")[0])).name
                ref_audio_path = voice_store.get_voice_audio_path(clean_ref)
                media_t = "audio/mpeg" if clean_ref.lower().endswith(".mp3") else "audio/wav"
                resp = await _serve_locked_preview_response(ref_audio_path, media_t, request=request)
                if resp is not None:
                    return resp
            except Exception:
                pass
        raise HTTPException(status_code=404, detail="Preview audio not generated yet")

    raise HTTPException(status_code=404, detail="Preview audio not generated yet")


@router.post("/api/tts/preview")
async def create_tts_preview_handler(
    payload: Optional[UnifiedTTSPreviewRequest] = None,
    request: Request = None,
) -> JSONResponse:
    payload_obj = payload or UnifiedTTSPreviewRequest()
    voice = (payload_obj.voice or payload_obj.voice_id or "").strip()
    provider = _parse_provider_or_400(payload_obj.provider if payload_obj.provider is not None else payload_obj.ttsType)
    text = (payload_obj.text or "").strip()
    language = (payload_obj.language or "").strip()
    force_refresh = bool(payload_obj.force_refresh)

    try:
        preview_id, preview_path, applied_speed = await synthesize_unified_tts_preview(
            provider=provider,
            voice=voice,
            text=text,
            language=language,
            force_refresh=force_refresh,
            speed=payload_obj.speed,
            rate=payload_obj.rate,
            pitch=payload_obj.pitch,
            style=payload_obj.style,
            tuning_params=payload_obj.tuning_params,
            auto_speed=bool(payload_obj.auto_speed),
            slot_duration_s=payload_obj.slot_duration_s,
            max_speed_rate=payload_obj.max_speed_rate or 1.35,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Voice preview generation timed out. Try again or choose another voice.") from exc
    except Exception as exc:
        logger.exception("Unified TTS preview generation failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {exc}") from exc

    preview_url = f"/api/tts/preview/{preview_id}/audio"
    return JSONResponse({
        "ok": True,
        "id": preview_id,
        "preview_id": preview_id,
        "preview_url": preview_url,
        "audio_url": preview_url,
        "voice": voice,
        "voice_id": voice,
        "provider": provider,
        "applied_speed": applied_speed,
    })


@router.post("/api/tts/batch")
async def create_tts_batch_handler(
    payload: Optional[BatchTTSRequest] = None,
    request: Request = None,
) -> JSONResponse:
    payload_obj = payload or BatchTTSRequest()
    raw_items = payload_obj.segments if payload_obj.segments is not None else (payload_obj.items or [])
    if not raw_items:
        raise HTTPException(status_code=400, detail="segments list is required and must be non-empty")

    batch_id = str(payload_obj.batch_id or uuid.uuid4().hex[:12])

    default_provider_raw = payload_obj.provider if payload_obj.provider is not None else payload_obj.ttsType
    if default_provider_raw is not None:
        _parse_provider_or_400(default_provider_raw)

    normalized_items: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        prov_raw = item.provider if item.provider is not None else (item.ttsType if item.ttsType is not None else default_provider_raw)
        prov = _parse_provider_or_400(prov_raw)
        seg_id = item.segment_id if item.segment_id is not None else (item.id if item.id is not None else idx)
        voice_val = (item.voice or item.voice_id or payload_obj.voice or "").strip()
        lang_val = (item.language or payload_obj.language or "").strip() or None
        normalized_items.append({
            "segment_id": seg_id,
            "text": (item.text or "").strip(),
            "voice": voice_val,
            "provider": prov,
            "language": lang_val,
            "speed": item.speed if item.speed is not None else payload_obj.speed,
            "rate": item.rate,
            "pitch": item.pitch,
            "style": item.style if item.style is not None else payload_obj.style,
            "tuning_params": item.tuning_params if item.tuning_params is not None else payload_obj.tuning_params,
            "force_refresh": bool(item.force_refresh or payload_obj.force_refresh),
            "auto_speed": bool(item.auto_speed if item.auto_speed is not None else payload_obj.auto_speed),
            "slot_duration_s": item.slot_duration_s,
            "max_speed_rate": item.max_speed_rate if item.max_speed_rate is not None else (payload_obj.max_speed_rate if payload_obj.max_speed_rate is not None else 1.35),
        })

    async def _is_disconnected() -> bool:
        if request is None:
            return False
        try:
            return await request.is_disconnected()
        except Exception:
            return False

    cancel_ev = register_batch_cancel(batch_id)
    try:
        results = await synthesize_tts_batch(
            normalized_items,
            cancel_event=cancel_ev,
            is_disconnected=_is_disconnected,
        )
    except Exception as exc:
        logger.exception("Batch TTS synthesis failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch TTS generation failed: {exc}") from exc
    finally:
        unregister_batch_cancel(batch_id)

    succeeded = sum(1 for r in results if r.get("ok"))
    failed = len(results) - succeeded
    return JSONResponse({
        "ok": failed == 0,
        "batch_id": batch_id,
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    })


@router.post("/api/tts/batch/{batch_id}/cancel")
async def cancel_tts_batch_handler(batch_id: str) -> JSONResponse:
    cancelled = cancel_batch(batch_id)
    return JSONResponse({"ok": True, "batch_id": batch_id, "cancelled": cancelled})


@router.post("/api/voices/design-assist")
async def voice_design_assist_handler(payload: VoiceDesignAssistRequest) -> JSONResponse:
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    p_lower = prompt.lower()

    if any(k in p_lower for k in ["tin tức", "thời sự", "bản tin", "news", "formal", "phát thanh", "anchor", "chính luận"]):
        style = "tin_tuc"
        temp = 0.70
        rep_pen = 1.25
        desc = "Phong cách bản tin, thời sự trang trọng và rõ ràng"
    elif any(k in p_lower for k in ["kể chuyện", "đọc truyện", "truyện", "story", "audiobook", "tiểu thuyết", "bedtime", "cổ tích", "thơ"]):
        style = "doc_truyen"
        temp = 0.95
        rep_pen = 1.15
        desc = "Phong cách kể chuyện truyền cảm, giàu ngữ điệu và biểu cảm"
    else:
        style = "tu_nhien"
        temp = 0.80
        rep_pen = 1.20
        desc = "Phong cách tự nhiên, hội thoại chân thực và gần gũi"

    suggested_tags = []
    if any(k in p_lower for k in ["cười", "vui", "hài", "chuckle", "happy", "laugh", "thân thiện", "ấm áp"]):
        suggested_tags.append("[cười]")
    if any(k in p_lower for k in ["thở dài", "buồn", "trầm", "sigh", "sad", "suy tư", "lắng đọng"]):
        suggested_tags.append("[thở dài]")
    if any(k in p_lower for k in ["hắng giọng", "nghiêm", "rõ", "dõng dạc", "throat", "chú ý"]):
        suggested_tags.append("[hắng giọng]")

    suggested_name = "Giọng " + ("Tin tức" if style == "tin_tuc" else "Kể chuyện" if style == "doc_truyen" else "Tự nhiên")
    if suggested_tags:
        tag_str = ", ".join(suggested_tags)
        suggested_name += f" ({tag_str})"

    return JSONResponse({
        "style": style,
        "temperature": temp,
        "repetition_penalty": rep_pen,
        "suggested_tags": suggested_tags,
        "suggested_name": suggested_name,
        "description": desc,
    })


@router.get("/api/tts/preview/{id}/audio")
async def get_tts_preview_audio_handler(id: str, request: Request = None) -> Response:
    clean_id = Path(unquote(id)).name
    candidates = [
        clean_id if clean_id.endswith(".wav") else f"{clean_id}.wav",
        clean_id,
        f"{clean_id}_preview.wav",
    ]
    cv = voice_store.get_voice(clean_id)
    if cv and cv.get("preview_audio_path"):
        candidates.insert(0, cv["preview_audio_path"])

    headers = {"Cache-Control": "no-cache, must-revalidate"}
    for cand in candidates:
        try:
            p = voice_store.get_preview_audio_path(cand)
        except ValueError:
            continue
        resp = await _serve_locked_preview_response(p, "audio/wav", headers=headers, request=request)
        if resp is not None:
            return resp

    if cv and cv.get("ref_audio_path"):
        try:
            p = voice_store.get_voice_audio_path(cv["ref_audio_path"])
            resp = await _serve_locked_preview_response(p, "audio/wav", headers=headers, request=request)
            if resp is not None:
                return resp
        except ValueError:
            pass

    raise HTTPException(status_code=404, detail=f"Preview audio not found for ID: {id}")


@router.post("/api/voices/trim-audio")
async def trim_audio_handler(request: Request):
    """
    Trim or auto-detect speech segment for voice cloning.
    Accepts multipart audio file and optional start_time, end_time, target_duration, auto_detect.
    """
    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid multipart form: {exc}") from exc
    audio_file = form.get("audio") or form.get("file")
    if not audio_file:
        raise HTTPException(status_code=400, detail="Audio file is required for trimming")

    cut_start_raw = form.get("start_time") or form.get("cut_start")
    cut_end_raw = form.get("end_time") or form.get("cut_end")
    target_dur_raw = form.get("target_duration")
    auto_detect_raw = str(form.get("auto_detect", "false")).strip().lower() in {"true", "1", "yes"}

    try:
        provider = int(form.get("provider", tts.VIENEU_TTS))
    except (ValueError, TypeError):
        provider = tts.VIENEU_TTS

    try:
        target_duration = float(target_dur_raw) if target_dur_raw else 10.0
    except (ValueError, TypeError):
        target_duration = 10.0

    filename = getattr(audio_file, "filename", "audio.wav") or "audio.wav"
    ext = Path(filename).suffix or ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        if hasattr(audio_file, "read"):
            while chunk := await audio_file.read(4 * 1024 * 1024):
                tmp_file.write(chunk)
        elif isinstance(audio_file, bytes):
            tmp_file.write(audio_file)

    try:
        orig_dur = probe_audio_duration(tmp_path)

        if auto_detect_raw:
            seg = detect_optimal_speech_segment(tmp_path, target_duration=target_duration)
            start_time = seg["start_time"]
            end_time = seg["end_time"]
        else:
            try:
                start_time = float(cut_start_raw) if cut_start_raw is not None and str(cut_start_raw).strip() != "" else 0.0
            except (ValueError, TypeError):
                start_time = 0.0
            try:
                end_time = float(cut_end_raw) if cut_end_raw is not None and str(cut_end_raw).strip() != "" else min(orig_dur, start_time + target_duration)
            except (ValueError, TypeError):
                end_time = min(orig_dur, start_time + target_duration)

        trim_id = f"trim_{uuid.uuid4().hex[:10]}"
        trim_filename = f"{trim_id}.wav"
        trim_dest = voice_store.PREVIEWS_DIR / trim_filename

        meta = normalize_reference_audio(
            tmp_path,
            trim_dest,
            provider=provider,
            start_time=start_time,
            end_time=end_time,
        )

        return JSONResponse({
            "ok": True,
            "original_duration": round(orig_dur, 2),
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "duration": round(meta["duration"], 2),
            "filename": trim_filename,
            "audio_url": f"/api/voices/trimmed-audio/{trim_filename}",
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to trim audio: %s", exc)
        raise HTTPException(status_code=500, detail=f"Audio trimming failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/api/voices/trimmed-audio/{filename}")
async def get_trimmed_audio_handler(filename: str) -> FileResponse:
    clean_name = Path(unquote(filename)).name
    if not clean_name.startswith("trim_") or not clean_name.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid trimmed audio filename")

    target_path = voice_store.PREVIEWS_DIR / clean_name
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail="Trimmed audio file not found")

    return FileResponse(target_path, media_type="audio/wav")


@router.post("/api/voices/preview-clone")
@router.post("/api/custom-voices/preview-clone")
async def preview_clone_handler(request: Request) -> JSONResponse:
    """
    Generate an audio preview for a cloned voice before saving it.
    Accepts multipart/form-data or application/json.
    """
    voice_store.init_voice_dirs()
    fields: dict[str, Any] = {}
    audio_file_obj: Optional[UploadFile] = None
    audio_bytes: Optional[bytes] = None
    audio_filename: str = "audio.wav"

    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart" in content_type:
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid multipart form: {exc}") from exc
        for key, val in form.items():
            if key in {"audio", "file"}:
                if hasattr(val, "filename") and val.filename:
                    audio_file_obj = val
                    audio_filename = Path(unquote(val.filename)).name
                elif isinstance(val, (bytes, bytearray)):
                    audio_bytes = bytes(val)
                    audio_filename = "sample.wav"
            else:
                fields[key] = str(val)
    elif "json" in content_type:
        try:
            fields = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
        if not isinstance(fields, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object")
    else:
        raise HTTPException(status_code=400, detail="Expected multipart/form-data or application/json")

    provider = _parse_provider_or_400(fields.get("provider"))
    text = fields.get("text") or fields.get("sample_text") or None
    language = str(fields.get("language") or "vi").strip()
    ref_text = str(fields.get("ref_text") or "").strip()

    if provider == tts.OMNIVOICE_TTS and not ref_text:
        raise HTTPException(
            status_code=400,
            detail="OmniVoice requires reference text transcript to prevent acoustic hallucinations",
        )

    cut_start_raw = fields.get("cut_start") or fields.get("start_time")
    cut_end_raw = fields.get("cut_end") or fields.get("end_time")
    try:
        start_time = float(cut_start_raw) if cut_start_raw is not None and str(cut_start_raw).strip() != "" else None
    except (ValueError, TypeError):
        start_time = None
    try:
        end_time = float(cut_end_raw) if cut_end_raw is not None and str(cut_end_raw).strip() != "" else None
    except (ValueError, TypeError):
        end_time = None

    denoise_raw = fields.get("denoise")
    denoise = str(denoise_raw).lower() in ("true", "1", "yes") if denoise_raw is not None else False

    if not audio_file_obj and not audio_bytes and fields.get("audio"):
        raw_audio = fields.get("audio")
        if isinstance(raw_audio, str):
            import base64
            if raw_audio.startswith("data:audio"):
                try:
                    _, encoded = raw_audio.split(",", 1)
                    audio_bytes = base64.b64decode(encoded)
                except Exception:
                    pass
            elif len(raw_audio) > 100:
                try:
                    audio_bytes = base64.b64decode(raw_audio)
                except Exception:
                    pass

    preview_id = f"preview_clone_{uuid.uuid4().hex[:10]}"
    temp_ref_path = voice_store.PREVIEWS_DIR / f"{preview_id}_temp_ref.wav"

    if audio_file_obj:
        ext = Path(audio_filename).suffix
        if not ext and getattr(audio_file_obj, "content_type", None):
            ct = str(audio_file_obj.content_type).lower()
            if "mpeg" in ct or "mp3" in ct:
                ext = ".mp3"
            elif "ogg" in ct:
                ext = ".ogg"
            elif "flac" in ct:
                ext = ".flac"
            elif "webm" in ct:
                ext = ".webm"
            elif "mp4" in ct or "m4a" in ct:
                ext = ".m4a"
        ext = ext or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            while chunk := await audio_file_obj.read(4 * 1024 * 1024):
                tmp_file.write(chunk)
        try:
            normalize_reference_audio(tmp_path, temp_ref_path, provider=provider, start_time=start_time, end_time=end_time)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to normalize preview audio", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Audio normalization failed: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)
    elif audio_bytes:
        ext = Path(audio_filename).suffix
        if not ext and fields.get("content_type"):
            ct = str(fields.get("content_type")).lower()
            if "mpeg" in ct or "mp3" in ct:
                ext = ".mp3"
            elif "ogg" in ct:
                ext = ".ogg"
            elif "flac" in ct:
                ext = ".flac"
            elif "webm" in ct:
                ext = ".webm"
            elif "mp4" in ct or "m4a" in ct:
                ext = ".m4a"
        ext = ext or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(audio_bytes)
        try:
            normalize_reference_audio(tmp_path, temp_ref_path, provider=provider, start_time=start_time, end_time=end_time)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to normalize preview audio", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Audio normalization failed: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        raise HTTPException(status_code=400, detail="Audio file is required to preview cloned voice")

    try:
        pid, preview_path = await synthesize_clone_preview(
            temp_ref_path,
            provider=provider,
            text=text,
            language=language,
            ref_text=ref_text,
            preview_id=preview_id,
            denoise=denoise,
        )
    except Exception as exc:
        logger.exception("Clone preview synthesis failed, generating emergency fallback: %s", exc)
        preview_filename = f"{preview_id}.wav"
        preview_path = voice_store.get_preview_audio_path(preview_filename)
        generate_fallback_preview_wav(preview_path)
        pid = preview_id
    finally:
        temp_ref_path.unlink(missing_ok=True)

    preview_url = f"/api/voices/preview-audio/{preview_path.name}"
    return JSONResponse({
        "ok": True,
        "preview_id": pid,
        "preview_filename": preview_path.name,
        "preview_url": preview_url,
        "audio_url": preview_url,
    })


@router.get("/api/voices/preview-audio/{filename}")
async def get_preview_audio_file_handler(filename: str, request: Request = None) -> Response:
    unquoted = unquote(filename)
    if ".." in unquoted or "/" in unquoted or "\\" in unquoted:
        raise HTTPException(status_code=400, detail="Invalid audio filename or path traversal detected")

    clean_name = Path(unquoted.split("?")[0]).name
    if not clean_name.lower().endswith((".wav", ".mp3", ".ogg")):
        raise HTTPException(status_code=400, detail="Invalid audio filename")

    try:
        target_path = voice_store.get_preview_audio_path(clean_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    media_t = "audio/mpeg" if clean_name.lower().endswith(".mp3") else ("audio/ogg" if clean_name.lower().endswith(".ogg") else "audio/wav")
    resp = await _serve_locked_preview_response(target_path, media_t, request=request)
    if resp is None:
        raise HTTPException(status_code=404, detail="Preview audio file not found")
    return resp


def register_routes(app) -> None:
    app.include_router(router)
