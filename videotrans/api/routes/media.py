# -*- coding: utf-8 -*-
from __future__ import annotations

import uuid
from pathlib import Path
from urllib.parse import unquote
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from pydantic import BaseModel, ConfigDict, Field

from videotrans.configure.contants import AUDIO_EXITS, VIDEO_EXTS
from videotrans.core.edit_asset_store import EDIT_ASSET_STORE
from videotrans.core.media_store import MEDIA

# Compatibility aliases for legacy tests; request handling uses the injected service.
EDIT_ASSETS = EDIT_ASSET_STORE.items
EDIT_ASSETS_LOCK = EDIT_ASSET_STORE.lock

router = APIRouter()


class ScanFoldersRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    recursive: bool = True

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class RegisterLocalMediaRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, extra="allow")


@router.post("/api/media/scan-folders")
@router.post("/api/media/scan_folders")
async def scan_folders_handler(payload: ScanFoldersRequest) -> JSONResponse:
    target_paths = payload.paths or []
    recursive = payload.recursive
    found_map: dict[str, Path] = {}

    for raw in target_paths:
        clean = str(raw).strip().strip('"').strip("'")
        if not clean:
            continue
        p = Path(clean)
        if not p.exists():
            continue
        if p.is_file():
            if p.suffix.lower() == ".mp4":
                found_map[str(p.resolve())] = p.resolve()
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            try:
                for match in p.glob(pattern):
                    if match.is_file() and match.suffix.lower() == ".mp4":
                        found_map[str(match.resolve())] = match.resolve()
            except Exception:
                pass

    results = []
    for resolved_str, path_obj in sorted(found_map.items()):
        try:
            sz = path_obj.stat().st_size
            results.append({
                "path": resolved_str,
                "filename": path_obj.name,
                "sizeBytes": sz,
                "parentDir": str(path_obj.parent),
            })
        except Exception:
            pass

    return JSONResponse({"ok": True, "videos": results, "count": len(results)})


@router.post("/api/media/register-local", status_code=201)
@router.post("/api/media/register_local", status_code=201)
async def register_local_media_handler(
    payload: RegisterLocalMediaRequest,
    request: Request,
) -> JSONResponse:
    store = getattr(request.app.state, "media_store", None) or MEDIA
    records = []

    for raw in payload.paths:
        clean = str(raw).strip().strip('"').strip("'")
        if not clean:
            continue
        p = Path(clean)
        if not p.is_file():
            continue
        try:
            rec = store.inspect(p, filename=p.name)
            records.append(rec.snapshot())
        except Exception:
            pass

    return JSONResponse({"ok": True, "media": records, "count": len(records)}, status_code=201)


@router.post("/api/media/batch-upload", status_code=201)
@router.post("/api/media/batch_upload", status_code=201)
async def batch_upload_media_handler(
    request: Request,
) -> JSONResponse:
    store = getattr(request.app.state, "media_store", None) or MEDIA
    form = await request.form()
    uploaded_records = []

    for key, value in form.multi_items():
        if isinstance(value, (UploadFile, StarletteUploadFile)) and value.filename:
            filename = Path(unquote(value.filename)).name
            ext = Path(filename).suffix.lower().lstrip(".")
            if ext not in {*VIDEO_EXTS, *AUDIO_EXITS}:
                continue
            media_id = uuid.uuid4().hex
            upload_path = store.upload_dir / f"{media_id}-{filename}"
            try:
                with upload_path.open("wb") as output:
                    while chunk := await value.read(4 * 1024 * 1024):
                        output.write(chunk)
                if upload_path.is_file() and upload_path.stat().st_size > 0:
                    rec = store.inspect(upload_path, filename, media_id=media_id)
                    uploaded_records.append(rec.snapshot())
            except Exception:
                upload_path.unlink(missing_ok=True)

    return JSONResponse({"ok": True, "media": uploaded_records, "count": len(uploaded_records)}, status_code=201)


@router.post("/api/media", status_code=201)
async def media_handler(
    request: Request,
    file: Optional[UploadFile] = File(None),
) -> JSONResponse:
    store = getattr(request.app.state, "media_store", None) or MEDIA
    upload_path: Path | None = None
    filename = ""

    # Support file parameter or parse from form
    if file is None or not file.filename:
        try:
            form = await request.form()
            form_file = form.get("file")
            if isinstance(form_file, (UploadFile, StarletteUploadFile)) and form_file.filename:
                file = form_file
        except Exception:
            pass

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="A video or audio file is required")

    filename = Path(unquote(file.filename)).name
    if not filename:
        raise HTTPException(status_code=400, detail="A video or audio file is required")

    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in {*VIDEO_EXTS, *AUDIO_EXITS}:
        raise HTTPException(status_code=400, detail=f"Unsupported media type: .{extension or 'unknown'}")

    media_id = uuid.uuid4().hex
    upload_path = store.upload_dir / f"{media_id}-{filename}"
    try:
        with upload_path.open("wb") as output:
            while chunk := await file.read(4 * 1024 * 1024):
                output.write(chunk)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Unable to read media: {exc}") from exc

    if not upload_path.is_file() or upload_path.stat().st_size == 0:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="A video or audio file is required")

    try:
        media = store.inspect(upload_path, filename, media_id=media_id)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Unable to read media: {exc}") from exc

    return JSONResponse(media.snapshot(), status_code=201)


@router.get("/api/media/{media_id}/file")
@router.get("/api/media/{media_id}")
async def get_media_file_handler(media_id: str, request: Request) -> FileResponse:
    store = getattr(request.app.state, "media_store", None) or MEDIA
    record = store.get(media_id)
    if not record or not record.path.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(record.path, filename=record.filename)


@router.post("/api/assets/{kind}", status_code=201)
@router.post("/api/edit/assets", status_code=201)
async def edit_asset_handler(
    request: Request,
    kind: Optional[str] = None,
    file: Optional[UploadFile] = File(None, description="Asset file to upload"),
) -> JSONResponse:
    if kind is None:
        kind = request.query_params.get("kind") or "background-audio"

    allowed = {
        "background-audio": AUDIO_EXITS,
        "thumbnail": {"png", "jpg", "jpeg", "webp"},
    }
    if kind not in allowed:
        raise HTTPException(status_code=404, detail="Unknown edit asset type")

    saved: Path | None = None
    filename = ""
    store = getattr(request.app.state, "media_store", None) or MEDIA
    edit_asset_store = getattr(request.app.state, "edit_asset_store", None) or EDIT_ASSET_STORE

    content_type = (request.headers.get("content-type") or "").lower()
    if "x-www-form-urlencoded" in content_type:
        raise HTTPException(status_code=400, detail="An asset file is required")

    if "multipart" in content_type:
        file_obj = file
        if file_obj is None or not file_obj.filename:
            try:
                form = await request.form()
                form_file = form.get("file")
                if isinstance(form_file, (UploadFile, StarletteUploadFile)) and form_file.filename:
                    file_obj = form_file
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"An asset file is required: {exc}") from exc

        if isinstance(file_obj, (UploadFile, StarletteUploadFile)) and file_obj.filename:
            filename = Path(unquote(file_obj.filename)).name
            extension = Path(filename).suffix.lower().lstrip(".")
            if extension not in set(allowed[kind]):
                raise HTTPException(status_code=400, detail=f"Unsupported {kind} type: .{extension or 'unknown'}")
            saved = store.upload_dir / f"edit-{uuid.uuid4().hex}-{filename}"
            with saved.open("wb") as output:
                while chunk := await file_obj.read(4 * 1024 * 1024):
                    output.write(chunk)
    else:
        raw_name = request.headers.get("X-Filename") or request.query_params.get("filename") or ""
        if not raw_name:
            raise HTTPException(status_code=400, detail="An asset file is required")
        filename = Path(unquote(raw_name)).name
        if not filename:
            raise HTTPException(status_code=400, detail="A valid filename is required")
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in set(allowed[kind]):
            raise HTTPException(status_code=400, detail=f"Unsupported {kind} type: .{extension or 'unknown'}")
        saved = store.upload_dir / f"edit-{uuid.uuid4().hex}-{filename}"
        with saved.open("wb") as output:
            async for chunk in request.stream():
                output.write(chunk)

    if saved is None or not saved.is_file() or saved.stat().st_size == 0:
        if saved and saved.is_file():
            saved.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="An asset file is required")

    asset_id = uuid.uuid4().hex
    edit_asset_store.put(asset_id, saved)
    return JSONResponse({"id": asset_id, "name": filename}, status_code=201)


def register_routes(app) -> None:
    app.include_router(router)
