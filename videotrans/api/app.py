# -*- coding: utf-8 -*-
"""Web application factory and route assembly using FastAPI."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders, State

from videotrans.configure.config import app_cfg, params as app_params
from videotrans.core.db import init_db
from videotrans.core.job_store import sweep_orphans_on_startup
from videotrans.core.job_manager import JobManager, JOBS, run_prepare_review
from videotrans.core.media_store import MediaStore, MEDIA
from videotrans.core.edit_asset_store import EditAssetStore, EDIT_ASSET_STORE
from videotrans.util._ffprobe import get_video_info
from videotrans.api.catalog import (
    UPLOAD_DIR,
    get_frontend_dir,
)
from videotrans.api.provider_helpers import (
    ensure_asr_configured,
    ensure_translation_configured,
    test_asr_provider,
    test_translation_provider,
)
from videotrans.api.task_params import build_task_params
from videotrans.api.ocr_helpers import extract_ocr_segment_text
from videotrans.api.routes import projects, jobs, media, settings, stages, voices
from videotrans.core import voice_store
from videotrans.util.gpus import getset_gpu
from videotrans.util import help_role
from videotrans.util.help_role import role_menu

logger = logging.getLogger("videotrans.api.app")

# Ensure State supports .get()
State.get = lambda self, k, d=None: getattr(self, k, d) if hasattr(self, k) else self._state.get(k, d)

# Ensure FastAPI supports dict-like access for backwards compatibility
FastAPI.__getitem__ = lambda self, k: getattr(self.state, k)
FastAPI.__setitem__ = lambda self, k, v: setattr(self.state, k, v)
FastAPI.__contains__ = lambda self, k: hasattr(self.state, k)


class CallableRoutesList(list):
    def __call__(self):
        class RouteDefWrapper:
            def __init__(self, r: Any):
                self.resource = type("Resource", (), {"canonical": getattr(r, "path", "")})()
                self.path = getattr(r, "path", "")
        return [RouteDefWrapper(r) for r in self]


class NoCacheMiddleware:
    """Pure ASGI middleware (not BaseHTTPMiddleware) to avoid buffering SSE streams."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        async def send_with_no_cache(message: Any) -> None:
            if message["type"] == "http.response.start":
                if (
                    path == "/"
                    or path.endswith((".html", ".js", ".css"))
                    or path.startswith("/assets")
                ):
                    headers = MutableHeaders(scope=message)
                    headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                    headers["Pragma"] = "no-cache"
                    headers["Expires"] = "0"
            await send(message)

        await self.app(scope, receive, send_with_no_cache)


def frontend_version(frontend_dir: Path | None = None) -> str:
    digest = hashlib.sha1()
    frontend_dir = frontend_dir or get_frontend_dir()
    if not frontend_dir.is_dir():
        return digest.hexdigest()
    for path in sorted(frontend_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        digest.update(f"{path.relative_to(frontend_dir)}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return digest.hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence
    from videotrans.configure.config import init_run
    app_cfg.exec_mode = "web"
    init_run()
    init_db()
    try:
        from videotrans.core.secret_store import migrate_plaintext_secrets
        migrate_plaintext_secrets()
    except Exception:
        pass
    sweep_orphans_on_startup()
    voice_store.init_voice_dirs()
    voice_store.migrate_legacy_voices()
    try:
        import threading
        from videotrans.tts._vieneu_compat import setup_vieneu_environment
        threading.Thread(target=setup_vieneu_environment, daemon=True, name="vieneu-warmup").start()
    except Exception:
        pass
    yield
    # Shutdown sequence
    try:
        from videotrans.process.process_manager import GlobalProcessManager
        GlobalProcessManager.shutdown()
    except Exception:
        pass


def create_app(
    *,
    job_manager: JobManager | None = None,
    upload_dir: Path | None = None,
    media_probe: Callable[[str | Path], dict[str, Any]] = get_video_info,
    settings_store: Any = None,
    asr_tester: Callable[[int, str], str] = test_asr_provider,
    translation_tester: Callable[[int, bool | None], str] = test_translation_provider,
    translation_runner: Callable | None = None,
    ocr_extractor: Callable | None = None,
    edit_asset_store: EditAssetStore | None = None,
    frontend_dir: Path | None = None,
    task_params_builder: Callable = build_task_params,
    asr_validator: Callable = ensure_asr_configured,
    translation_validator: Callable = ensure_translation_configured,
    gpu_initializer: Callable = getset_gpu,
    role_provider: Callable | None = None,
    reload: bool = False,
) -> FastAPI:
    # Initialize DB, secrets, and voice stores early for synchronous and test environments
    init_db()
    try:
        from videotrans.core.secret_store import migrate_plaintext_secrets, sync_env_and_db_secrets
        migrate_plaintext_secrets()
        sync_env_and_db_secrets()
    except Exception:
        pass
    sweep_orphans_on_startup()
    voice_store.init_voice_dirs()
    voice_store.migrate_legacy_voices()

    frontend_root = frontend_dir or get_frontend_dir()
    if (frontend_root / "dist" / "index.html").is_file():
        frontend_dist = frontend_root / "dist"
    elif (frontend_root / "index.html").is_file():
        frontend_dist = frontend_root
    else:
        frontend_dist = frontend_root / "dist"

    app = FastAPI(
        title="pyVideoTrans WebUI",
        version="4.05",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Pure ASGI No-Cache Middleware
    app.add_middleware(NoCacheMiddleware)

    # Validation error handler returning 400 for consistency with tests/API contract
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        path = request.url.path
        body_err = any(err.get("loc", ())[:1] == ("body",) for err in exc.errors())
        if body_err:
            if path in {"/api/jobs", "/api/render", "/api/export"}:
                return JSONResponse(status_code=400, content={"detail": "A JSON job request is required"})
            if path == "/api/translate":
                return JSONResponse(status_code=400, content={"detail": "A JSON translation request is required"})
        errors = exc.errors()
        msg = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in errors)
        return JSONResponse(status_code=400, content={"detail": f"Validation error: {msg}"})

    if job_manager is None and translation_runner is not None:
        job_manager = JobManager(
            runner=run_prepare_review,
            translation_runner=translation_runner,
        )

    # Attach shared application state
    app.state.job_manager = job_manager or JOBS
    app.state.media_store = MEDIA if upload_dir is None and media_probe is get_video_info else MediaStore(upload_dir or UPLOAD_DIR, media_probe)
    app.state.settings_store = app_params if settings_store is None else settings_store
    app.state.asr_tester = asr_tester
    app.state.translation_tester = translation_tester
    app.state.ocr_extractor = ocr_extractor or extract_ocr_segment_text
    app.state.edit_asset_store = edit_asset_store or EDIT_ASSET_STORE
    app.state.frontend_dir = frontend_dist
    app.state.task_params_builder = task_params_builder
    app.state.asr_validator = asr_validator
    app.state.translation_validator = translation_validator
    app.state.gpu_initializer = gpu_initializer
    app.state.role_provider = role_provider if role_provider is not None else help_role.role_menu
    app.state.reload = reload

    # Include API routers
    app.include_router(settings.router)
    app.include_router(media.router)
    app.include_router(projects.router)
    app.include_router(jobs.router)
    app.include_router(stages.router)
    app.include_router(voices.router)

    # Static assets mounting and root handling
    assets_dir = frontend_dist / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=str(assets_dir), check_dir=False), name="assets")

    @app.get("/")
    @app.get("/index.html")
    async def index_handler(request: Request):
        index_file = frontend_dist / "index.html"
        if not index_file.is_file():
            logger.info("React build not found in %s; redirecting / to http://localhost:3000 (Vite dev server)", frontend_dist)
            return RedirectResponse(url="http://localhost:3000", status_code=307)

        if getattr(request.app.state, "reload", False):
            html = index_file.read_text(encoding="utf-8")
            script = f"""<script>
let frontendVersion = '{frontend_version(frontend_dist)}';
setInterval(async () => {{
  try {{
    const response = await fetch('/__dev_reload__', {{ cache: 'no-store' }});
    if ((await response.json()).version !== frontendVersion) location.reload();
  }} catch {{}}
}}, 500);
</script>"""
            return HTMLResponse(
                content=html.replace("</body>", f"{script}</body>"),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    if reload:
        @app.get("/__dev_reload__")
        async def dev_reload_handler(request: Request) -> JSONResponse:
            return JSONResponse(
                {"version": frontend_version(request.app.state.frontend_dir)},
                headers={"Cache-Control": "no-store"},
            )

    app.router.routes = CallableRoutesList(app.router.routes)
    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="pyVideoTrans Dubbing Video WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload", action="store_true", help="reload browser tabs when frontend files change")
    args = parser.parse_args()
    app_cfg.exec_mode = "web"
    import uvicorn
    uvicorn.run("videotrans.api.app:app", host=args.host, port=args.port, reload=args.reload)
