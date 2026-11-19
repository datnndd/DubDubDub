# -*- coding: utf-8 -*-
"""Offline workspace isolation and synthetic media/HTTP fixtures for E2E tests."""
from __future__ import annotations

import copy
import datetime
import functools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import uuid

import numpy as np
from PIL import Image, ImageDraw
import pytest
import soundfile as sf

# Force offline Hugging Face mode before any model/hub checks occur.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def make_wav(
    path: str | Path,
    duration_s: float = 1.5,
    sr: int = 16000,
    channels: int = 1,
    freq: float = 440.0,
    speech_like: bool = False,
    amplitude: float = 0.35,
    silent: bool = False,
) -> Path:
    """Create a real PCM16 WAV file via numpy + soundfile."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    num_samples = max(1, int(round(duration_s * sr)))
    if silent:
        mono = np.zeros(num_samples, dtype=np.float32)
    elif speech_like:
        # Build voiced-burst + silence pattern that reliably activates Ten-VAD.
        t = np.linspace(0.0, duration_s, num_samples, endpoint=False, dtype=np.float64)
        base = (
            0.45 * np.sin(2.0 * np.pi * 210.0 * t)
            + 0.30 * np.sin(2.0 * np.pi * 420.0 * t)
            + 0.15 * np.sin(2.0 * np.pi * 660.0 * t)
        )
        syllable_mod = 0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * t))
        envelope = np.zeros(num_samples, dtype=np.float64)
        # Put speech bursts with clear silence gaps (>= 0.4s) if duration >= 1.5s.
        if duration_s >= 1.5:
            s1, e1 = int(0.25 * sr), int(min(duration_s - 0.1, 1.1) * sr)
            envelope[s1:e1] = 1.0
            if duration_s >= 2.5:
                s2, e2 = int(1.6 * sr), int(min(duration_s - 0.15, 2.5) * sr)
                envelope[s2:e2] = 1.0
        else:
            s1, e1 = int(0.1 * sr), int(max(0.2, duration_s - 0.1) * sr)
            envelope[s1:e1] = 1.0
        mono = (amplitude * base * syllable_mod * envelope).astype(np.float32)
    else:
        t = np.linspace(0.0, duration_s, num_samples, endpoint=False, dtype=np.float64)
        mono = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    if channels > 1:
        data = np.column_stack([mono for _ in range(channels)])
    else:
        data = mono
    sf.write(str(out), data, sr, subtype="PCM_16")
    return out


def make_png(
    path: str | Path,
    width: int = 320,
    height: int = 180,
    text: str | None = "Sample Frame",
    bg_color: tuple[int, int, int] = (24, 32, 48),
) -> Path:
    """Create a real PNG image via PIL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=bg_color)
    if text:
        draw = ImageDraw.Draw(img)
        draw.text((20, height // 2 - 10), text, fill=(245, 245, 245))
    img.save(str(out), format="PNG")
    return out


def make_mp4(
    path: str | Path,
    duration_s: float = 2.0,
    width: int = 320,
    height: int = 240,
    fps: int = 10,
    with_audio: bool = True,
    sr: int = 16000,
    text_overlay: str | None = None,
    speech_like: bool = True,
) -> Path:
    """Create a real H.264 (+ optional AAC audio) MP4 using PIL + ffmpeg."""
    from videotrans.configure.config import FFMPEG_BIN

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem_dir = out.parent / f".synth_{uuid.uuid4().hex[:8]}"
    stem_dir.mkdir(parents=True, exist_ok=True)
    frame_path = stem_dir / "frame.png"
    wav_path = stem_dir / "audio.wav"
    try:
        make_png(frame_path, width=width, height=height, text=text_overlay or "pyvideotrans E2E")
        cmd = [
            FFMPEG_BIN or "ffmpeg",
            "-y",
            "-loop", "1",
            "-framerate", str(fps),
            "-t", f"{duration_s:.3f}",
            "-i", str(frame_path),
        ]
        if with_audio:
            make_wav(wav_path, duration_s=duration_s, sr=sr, channels=1, speech_like=speech_like)
            cmd.extend(["-i", str(wav_path)])
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
        ])
        if with_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "64k", "-shortest"])
        else:
            cmd.extend(["-an"])
        cmd.append(str(out))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr}")
        return out
    finally:
        shutil.rmtree(stem_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolate_e2e_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Completely isolate SQLite DB, storage dirs, TEMP_ROOT, secrets, and cfg/params JSON."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    for env_key in (
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "ELEVENLABS_API_KEY",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(env_key, raising=False)

    from videotrans.configure import _paths, config as runtime_config
    from videotrans.core import db, media_store, project_store, secret_store, storage_config, voice_store
    from videotrans.services import voice_preview
    from videotrans.util._ffprobe import get_video_info

    # 1. Isolate SQLite DB and secret store
    orig_db = db.get_db_path()
    e2e_db = tmp_path / "e2e.db"
    monkeypatch.setenv("DB_PATH", str(e2e_db))
    db.set_db_path(e2e_db)
    db.init_db(e2e_db)
    secret_store.invalidate()

    # 2. Protect videotrans/cfg.json & videotrans/params.json on disk and in memory
    orig_settings_json_path = runtime_config.settings._json_path
    orig_params_json_path = runtime_config.params._json_path
    orig_settings_state = copy.deepcopy(runtime_config.settings.__dict__)
    orig_params_state = copy.deepcopy(runtime_config.params.__dict__)
    orig_app_cfg_state = copy.deepcopy(runtime_config.app_cfg.__dict__)

    cfg_copy = tmp_path / "cfg.json"
    params_copy = tmp_path / "params.json"
    if Path(orig_settings_json_path).is_file():
        shutil.copy2(orig_settings_json_path, cfg_copy)
    if Path(orig_params_json_path).is_file():
        shutil.copy2(orig_params_json_path, params_copy)
    monkeypatch.setattr(runtime_config.settings, "_json_path", str(cfg_copy), raising=False)
    monkeypatch.setattr(runtime_config.params, "_json_path", str(params_copy), raising=False)

    # 3. Redirect all workspace directories to tmp_path
    out_dir = tmp_path / "output"
    projects_dir = out_dir / "projects"
    voices_dir = tmp_path / "data" / "voices"
    previews_dir = tmp_path / "tmp" / "voice_previews"
    uploads_dir = tmp_path / "tmp" / "webui_uploads"
    models_dir = tmp_path / "models"
    temp_root = tmp_path / "tmp"
    temp_dir = temp_root / str(os.getpid())
    translate_cache = temp_root / "translate_cache"
    dubbing_cache = temp_root / "dubbing_cache"
    logs_dir = tmp_path / "logs"

    for d in (
        out_dir,
        projects_dir,
        voices_dir,
        previews_dir,
        uploads_dir,
        models_dir,
        temp_root,
        temp_dir,
        translate_cache,
        dubbing_cache,
        logs_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(runtime_config.settings, "homedir", str(out_dir), raising=False)
    monkeypatch.setattr(runtime_config, "HOME_DIR", str(out_dir), raising=False)

    monkeypatch.setattr(
        storage_config,
        "DEFAULT_STORAGE",
        {
            "output_dir": out_dir,
            "uploads_dir": uploads_dir,
            "models_dir": models_dir,
            "temp_dir": temp_root,
        },
        raising=False,
    )

    monkeypatch.setattr(project_store, "DEFAULT_PROJECTS_DIR", projects_dir, raising=False)
    monkeypatch.setattr(project_store, "PROJECTS_OUTPUT_DIR", projects_dir, raising=False)
    monkeypatch.setattr(project_store, "get_projects_root", lambda: projects_dir)

    monkeypatch.setattr(voice_store, "VOICES_DIR", voices_dir, raising=False)
    monkeypatch.setattr(voice_store, "PREVIEWS_DIR", previews_dir, raising=False)

    temp_root_posix = temp_root.as_posix()
    temp_dir_posix = temp_dir.as_posix()
    trans_cache_posix = translate_cache.as_posix()
    dubb_cache_posix = dubbing_cache.as_posix()

    monkeypatch.setattr(_paths, "TEMP_ROOT", temp_root_posix, raising=False)
    monkeypatch.setattr(_paths, "TEMP_DIR", temp_dir_posix, raising=False)
    monkeypatch.setattr(_paths, "TRANSLATE_CACHE", trans_cache_posix, raising=False)
    monkeypatch.setattr(_paths, "DUBBING_CACHE", dubb_cache_posix, raising=False)
    monkeypatch.setattr(_paths, "LOGS_DIR", logs_dir.as_posix(), raising=False)

    monkeypatch.setattr(runtime_config, "TEMP_ROOT", temp_root_posix, raising=False)
    monkeypatch.setattr(runtime_config, "TEMP_DIR", temp_dir_posix, raising=False)
    monkeypatch.setattr(runtime_config, "TRANSLATE_CACHE", trans_cache_posix, raising=False)
    monkeypatch.setattr(runtime_config, "DUBBING_CACHE", dubb_cache_posix, raising=False)
    monkeypatch.setattr(runtime_config, "LOGS_DIR", logs_dir.as_posix(), raising=False)

    isolated_media = media_store.MediaStore(uploads_dir, get_video_info)
    monkeypatch.setattr(media_store, "UPLOAD_DIR", uploads_dir, raising=False)
    monkeypatch.setattr(media_store, "MEDIA", isolated_media, raising=False)

    from videotrans.api import catalog as api_catalog, task_params as api_task_params

    monkeypatch.setattr(api_catalog, "UPLOAD_DIR", uploads_dir, raising=False)
    monkeypatch.setattr(api_catalog, "OUTPUT_DIR", out_dir, raising=False)
    monkeypatch.setattr(api_task_params, "OUTPUT_DIR", out_dir, raising=False)
    monkeypatch.setattr(api_task_params, "TEMP_DIR", temp_dir_posix, raising=False)
    if getattr(api_task_params.build_task_params, "__kwdefaults__", None) is not None:
        monkeypatch.setitem(api_task_params.build_task_params.__kwdefaults__, "output_dir", out_dir)
        monkeypatch.setitem(api_task_params.build_task_params.__kwdefaults__, "temp_dir", Path(temp_dir_posix))

    for mod_name in (
        "videotrans.task.orchestrator",
        "videotrans.task._stage_dubbing",
        "videotrans.configure.base",
        "videotrans.api.provider_helpers",
        "videotrans.api.task_params",
        "videotrans.api.catalog",
        "videotrans.api.routes.jobs",
        "videotrans.api.routes.stages",
        "videotrans.api.routes.media",
        "videotrans.api.routes.projects",
        "videotrans.core.media_store",
        "videotrans.translator._base",
        "videotrans.recognition._base",
        "videotrans.tts._base",
        "videotrans.tts._omnivoice",
        "cli",
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if hasattr(mod, "TEMP_ROOT"):
            monkeypatch.setattr(mod, "TEMP_ROOT", temp_root_posix, raising=False)
        if hasattr(mod, "TEMP_DIR"):
            monkeypatch.setattr(mod, "TEMP_DIR", temp_dir_posix, raising=False)
        if hasattr(mod, "DUBBING_CACHE"):
            monkeypatch.setattr(mod, "DUBBING_CACHE", dubb_cache_posix, raising=False)
        if hasattr(mod, "TRANSLATE_CACHE"):
            monkeypatch.setattr(mod, "TRANSLATE_CACHE", trans_cache_posix, raising=False)
        if hasattr(mod, "UPLOAD_DIR"):
            monkeypatch.setattr(mod, "UPLOAD_DIR", uploads_dir, raising=False)
        if hasattr(mod, "OUTPUT_DIR"):
            monkeypatch.setattr(mod, "OUTPUT_DIR", out_dir, raising=False)
        if hasattr(mod, "MEDIA"):
            monkeypatch.setattr(mod, "MEDIA", isolated_media, raising=False)

    edit_asset_mod = sys.modules.get("videotrans.core.edit_asset_store")
    if edit_asset_mod is not None and hasattr(edit_asset_mod, "EDIT_ASSET_STORE"):
        with edit_asset_mod.EDIT_ASSET_STORE.lock:
            edit_asset_mod.EDIT_ASSET_STORE.items.clear()

    # Clear app_cfg transient sets/queues
    runtime_config.app_cfg.stoped_uuid_set.clear()
    runtime_config.app_cfg.queue_novice.clear()

    try:
        yield tmp_path
    finally:
        voice_preview.set_preview_synthesizer(None)
        cur_settings_json_path = runtime_config.settings._json_path
        runtime_config.settings.__dict__.clear()
        runtime_config.settings.__dict__.update(orig_settings_state)
        runtime_config.settings._json_path = cur_settings_json_path
        cur_params_json_path = runtime_config.params._json_path
        runtime_config.params.__dict__.clear()
        runtime_config.params.__dict__.update(orig_params_state)
        runtime_config.params._json_path = cur_params_json_path
        runtime_config.app_cfg.__dict__.clear()
        runtime_config.app_cfg.__dict__.update(orig_app_cfg_state)
        edit_asset_mod = sys.modules.get("videotrans.core.edit_asset_store")
        if edit_asset_mod is not None and hasattr(edit_asset_mod, "EDIT_ASSET_STORE"):
            with edit_asset_mod.EDIT_ASSET_STORE.lock:
                edit_asset_mod.EDIT_ASSET_STORE.items.clear()
        db.set_db_path(orig_db)
        secret_store.invalidate()


@pytest.fixture
def e2e_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create an isolated FastAPI app backed by tmp_path SQLite, MediaStore, and JobManager."""
    from videotrans.api import catalog as api_catalog, task_params as api_task_params
    from videotrans.api.app import create_app
    from videotrans.api.routes import jobs as jobs_routes, media as media_routes, projects as projects_routes
    from videotrans.api.task_params import build_task_params
    from videotrans.configure import config as runtime_config
    from videotrans.core.edit_asset_store import EditAssetStore
    from videotrans.core.job_manager import JobManager, run_prepare_review
    from videotrans.core import media_store
    from videotrans.task import orchestrator
    from videotrans.task.orchestrator import run_staged_translation
    from videotrans.translator import _base as trans_base

    # Ensure late-imported modules also point TEMP_DIR / TEMP_ROOT / MEDIA to tmp_path.
    temp_root_posix = runtime_config.TEMP_ROOT
    temp_dir_posix = runtime_config.TEMP_DIR
    uploads_dir = tmp_path / "tmp" / "webui_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(orchestrator, "TEMP_DIR", temp_dir_posix, raising=False)
    monkeypatch.setattr(api_task_params, "TEMP_DIR", temp_dir_posix, raising=False)
    monkeypatch.setattr(api_catalog, "UPLOAD_DIR", uploads_dir, raising=False)
    monkeypatch.setattr(api_catalog, "OUTPUT_DIR", tmp_path / "output", raising=False)
    monkeypatch.setattr(jobs_routes, "TEMP_DIR", temp_dir_posix, raising=False)
    monkeypatch.setattr(trans_base, "TEMP_ROOT", temp_root_posix, raising=False)

    manager = JobManager(
        runner=run_prepare_review,
        translation_runner=run_staged_translation,
    )
    app = create_app(
        job_manager=manager,
        upload_dir=uploads_dir,
        edit_asset_store=EditAssetStore(),
        task_params_builder=functools.partial(
            build_task_params,
            output_dir=tmp_path / "output",
            temp_dir=Path(temp_dir_posix),
        ),
    )
    monkeypatch.setattr(media_routes, "MEDIA", app.state.media_store, raising=False)
    monkeypatch.setattr(projects_routes, "MEDIA", app.state.media_store, raising=False)
    monkeypatch.setattr(jobs_routes, "MEDIA", app.state.media_store, raising=False)
    monkeypatch.setattr(media_store, "MEDIA", app.state.media_store, raising=False)
    return app


@pytest.fixture
def e2e_client(e2e_app):
    """Synchronous FastAPI TestClient without lifespan side effects."""
    from fastapi.testclient import TestClient

    return TestClient(e2e_app)


class LocalLlmServer:
    """In-process HTTP server speaking OpenAI /v1/chat/completions & Gemini generateContent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.delay_sec: float = 0.0
        self.failures_before_ok: int = 0
        self.failure_status: int = 429
        self.always_fail_status: int | None = None
        self.think_prefix: str = ""
        self.unclosed_tag: bool = False
        self.empty_response: bool = False
        self.custom_responder: Callable[[dict[str, Any], str], str] | None = None

        parent = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
                try:
                    payload = json.loads(raw_body) if raw_body else {}
                except json.JSONDecodeError:
                    payload = {"_raw": raw_body}

                with parent._lock:
                    parent.in_flight += 1
                    if parent.in_flight > parent.max_in_flight:
                        parent.max_in_flight = parent.in_flight
                    req_index = len(parent.requests)
                    parent.requests.append({
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": payload,
                        "timestamp": time.monotonic(),
                    })
                    should_fail_count = parent.failures_before_ok
                    if should_fail_count > 0:
                        parent.failures_before_ok -= 1
                    always_fail = parent.always_fail_status
                    fail_status = parent.failure_status
                    delay = parent.delay_sec

                try:
                    if delay > 0:
                        time.sleep(delay)

                    if always_fail is not None:
                        self._send_error_json(always_fail, f"Forced HTTP {always_fail}")
                        return
                    if should_fail_count > 0:
                        self._send_error_json(fail_status, f"Transient HTTP {fail_status} (#{req_index})")
                        return

                    text_out = parent._build_response_text(payload, self.path)
                    if ":streamGenerateContent" in self.path or "generateContent" in self.path:
                        if "alt=sse" in self.path or ":streamGenerateContent" in self.path:
                            sse_obj = {
                                "candidates": [
                                    {
                                        "content": {"parts": [{"text": text_out}], "role": "model"},
                                        "finishReason": "STOP",
                                    }
                                ]
                            }
                            body_bytes = f"data: {json.dumps(sse_obj)}\n\n".encode("utf-8")
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.send_header("Content-Length", str(len(body_bytes)))
                            self.end_headers()
                            self.wfile.write(body_bytes)
                        else:
                            resp_obj = {
                                "candidates": [
                                    {
                                        "content": {"parts": [{"text": text_out}], "role": "model"},
                                        "finishReason": "STOP",
                                    }
                                ]
                            }
                            body_bytes = json.dumps(resp_obj).encode("utf-8")
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Content-Length", str(len(body_bytes)))
                            self.end_headers()
                            self.wfile.write(body_bytes)
                    else:
                        resp_obj = {
                            "id": f"chatcmpl-e2e-{req_index}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": payload.get("model", "gpt-5-mini"),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": text_out},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                        body_bytes = json.dumps(resp_obj).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body_bytes)))
                        self.end_headers()
                        self.wfile.write(body_bytes)
                finally:
                    with parent._lock:
                        parent.in_flight -= 1

            def _send_error_json(self, status_code: int, message: str) -> None:
                err_obj = {
                    "error": {
                        "message": message,
                        "code": status_code,
                        "status": "RESOURCE_EXHAUSTED" if status_code == 429 else "ERROR",
                    }
                }
                body_bytes = json.dumps(err_obj).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self._server.server_port)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def openai_url(self) -> str:
        return f"{self.base_url}/v1"

    @property
    def gemini_url(self) -> str:
        return self.base_url

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self.in_flight = 0
            self.max_in_flight = 0
            self.delay_sec = 0.0
            self.failures_before_ok = 0
            self.failure_status = 429
            self.always_fail_status = None
            self.think_prefix = ""
            self.unclosed_tag = False
            self.empty_response = False
            self.custom_responder = None

    def _build_response_text(self, payload: dict[str, Any], path: str) -> str:
        if self.custom_responder is not None:
            return self.custom_responder(payload, path)
        if self.empty_response:
            return "<TRANSLATE_TEXT></TRANSLATE_TEXT>"

        user_text = ""
        if "messages" in payload and isinstance(payload["messages"], list):
            for msg in reversed(payload["messages"]):
                if msg.get("role") == "user":
                    user_text = str(msg.get("content") or "")
                    break
        elif "contents" in payload and isinstance(payload["contents"], list):
            parts = payload["contents"][-1].get("parts") or []
            user_text = "\n".join(str(p.get("text") or "") for p in parts if isinstance(p, dict))

        # Extract <INPUT>\n...\n</INPUT> block (avoiding inline `<INPUT>` mentions in prose)
        m_all = re.findall(r"<INPUT>\s*\r?\n([\s\S]*?)\r?\n\s*</INPUT>", user_text)
        if not m_all:
            m_all = re.findall(r"<INPUT>\s*([^`<][\s\S]*?)\s*</INPUT>", user_text)
        batch = m_all[-1] if m_all else user_text

        translated_body = self._translate_batch(batch)
        if self.unclosed_tag:
            return f"{self.think_prefix}<TRANSLATE_TEXT>\n{translated_body}"
        return f"{self.think_prefix}<TRANSLATE_TEXT>\n{translated_body}\n</TRANSLATE_TEXT>"

    @staticmethod
    def _translate_batch(batch: str) -> str:
        stripped = batch.strip()
        # Handle JSON payload if compact JSON format is used
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return json.dumps(
                        [
                            {**item, "text": f"[TR] {item.get('text', '')}"}
                            if isinstance(item, dict)
                            else f"[TR] {item}"
                            for item in parsed
                        ],
                        ensure_ascii=False,
                    )
                if isinstance(parsed, dict):
                    return json.dumps(
                        {k: f"[TR] {v}" for k, v in parsed.items()},
                        ensure_ascii=False,
                    )
            except Exception:
                pass

        # Handle SRT blocks
        if "-->" in stripped:
            blocks = re.split(r"\r?\n\s*\r?\n", stripped)
            out_blocks = []
            for blk in blocks:
                lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
                if len(lines) >= 3 and "-->" in lines[1]:
                    idx_line = lines[0]
                    time_line = lines[1]
                    text_lines = " ".join(lines[2:])
                    out_blocks.append(f"{idx_line}\n{time_line}\n[TR] {text_lines}")
                elif lines:
                    out_blocks.append("\n".join(f"[TR] {ln}" for ln in lines))
            return "\n\n".join(out_blocks)

        # Handle indexed lines `[1] ...` or plain lines
        out_lines = []
        for ln in stripped.splitlines():
            ln_s = ln.strip()
            if not ln_s:
                continue
            idx_m = re.match(r"^(\[\d+\]\s*)(.*)$", ln_s)
            if idx_m:
                out_lines.append(f"{idx_m.group(1)}[TR] {idx_m.group(2)}")
            else:
                out_lines.append(f"[TR] {ln_s}")
        return "\n".join(out_lines)

    def close(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass


@pytest.fixture
def local_llm_server():
    """Spin up an in-process OpenAI/Gemini-compatible HTTP server for offline translation E2E."""
    srv = LocalLlmServer()
    try:
        yield srv
    finally:
        srv.close()
