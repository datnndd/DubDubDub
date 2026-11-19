# -*- coding: utf-8 -*-
"""Synthetic media, PIL paragraph, FFmpeg video, local LLM server, and isolation helpers."""

from __future__ import annotations

import copy
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Optional
import uuid
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

_EN_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

_ZH_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
)


def find_en_font() -> str:
    for p in _EN_FONT_CANDIDATES:
        if Path(p).is_file():
            return p
    pytest.skip("No suitable English TTF font found for synthetic OCR rendering")
    return ""


def find_zh_font() -> str:
    for p in _ZH_FONT_CANDIDATES:
        if Path(p).is_file():
            return p
    pytest.skip("No suitable CJK font (msyh.ttc / simsun.ttc) found for synthetic OCR rendering")
    return ""


def load_font(lang: str = "en", size: int = 32) -> ImageFont.FreeTypeFont:
    font_path = find_zh_font() if str(lang).lower().startswith(("zh", "ch", "ja", "ko", "mix")) else find_en_font()
    return ImageFont.truetype(font_path, size)


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not available on PATH")
    return ffmpeg


def require_ocr_env() -> str:
    ffmpeg = require_ffmpeg()
    from videotrans.ocr._paddle import PaddleOcrProvider

    if not PaddleOcrProvider().is_available():
        pytest.skip("PaddleOcrProvider not available in this environment")
    return ffmpeg


def normalized_text_similarity(a: str, b: str) -> float:
    from videotrans.ocr._text import normalize_text

    na = normalize_text(a)
    nb = normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def make_blank_bgr(size: tuple[int, int] = (960, 260), bg: tuple[int, int, int] = (15, 18, 22)) -> np.ndarray:
    w, h = size
    img = Image.new("RGB", (w, h), bg)
    rgb = np.asarray(img, dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def make_paragraph_bgr(
    lines: Iterable[str],
    *,
    lang: str = "en",
    font_size: int = 32,
    size: tuple[int, int] = (1080, 280),
    bg: tuple[int, int, int] = (15, 18, 22),
    fg: tuple[int, int, int] = (255, 255, 255),
    x: int = 40,
    y_start: int = 24,
    line_spacing: int = 52,
) -> np.ndarray:
    """Render a multi-line paragraph onto a dark image and return BGR ndarray."""
    line_list = list(lines)
    w, h = size
    needed_h = y_start + max(1, len(line_list)) * line_spacing + 24
    if needed_h > h:
        h = needed_h
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    font = load_font(lang=lang, size=font_size)
    for idx, line in enumerate(line_list):
        draw.text((x, y_start + idx * line_spacing), line, font=font, fill=fg)
    rgb = np.asarray(img, dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def make_tiny_roi_bgr(
    text: str,
    *,
    lang: str = "en",
    size: tuple[int, int] = (720, 38),
    font_size: int = 22,
    bg: tuple[int, int, int] = (15, 18, 22),
    fg: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Render readable text inside a tiny (<48px tall) ROI strip."""
    w, h = size
    assert h < 48, f"Tiny ROI test helper expects height < 48px, got {h}"
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    font = load_font(lang=lang, size=font_size)
    draw.text((28, max(2, (h - font_size) // 2 - 2)), text, font=font, fill=fg)
    rgb = np.asarray(img, dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def make_jittered_sameline_bgr(
    *,
    lang: str = "en",
    font_size: int = 32,
    size: tuple[int, int] = (1000, 180),
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Render 2 rows where left column y is 4px LOWER than right column y.

    Catches raw (y, x) sorting that puts right-hand text before left-hand text.
    """
    rows = [
        ("Left alpha one", "Right beta one"),
        ("Left gamma two", "Right delta two"),
    ]
    img = Image.new("RGB", size, (15, 18, 22))
    draw = ImageDraw.Draw(img)
    font = load_font(lang=lang, size=font_size)
    row_ys = [40, 102]
    for (left_txt, right_txt), y_left in zip(rows, row_ys):
        y_right = y_left - 4
        draw.text((40, y_left), left_txt, font=font, fill=(255, 255, 255))
        draw.text((480, y_right), right_txt, font=font, fill=(255, 255, 255))
    rgb = np.asarray(img, dtype=np.uint8)
    return rgb[:, :, ::-1].copy(), rows


def make_two_column_bgr(
    *,
    lang: str = "en",
    font_size: int = 30,
    size: tuple[int, int] = (1020, 230),
) -> tuple[np.ndarray, list[str], list[str]]:
    """Render 3 lines in Col 1 (x=40) and 3 lines in Col 2 (x=520) separated by a wide gutter."""
    col1 = [
        "Column one line A",
        "Column one line B",
        "Column one line C",
    ]
    col2 = [
        "Column two line X",
        "Column two line Y",
        "Column two line Z",
    ]
    img = Image.new("RGB", size, (15, 18, 22))
    draw = ImageDraw.Draw(img)
    font = load_font(lang=lang, size=font_size)
    ys = [30, 86, 142]
    for line, y in zip(col1, ys):
        draw.text((40, y), line, font=font, fill=(255, 255, 255))
    for line, y in zip(col2, ys):
        draw.text((520, y), line, font=font, fill=(255, 255, 255))
    rgb = np.asarray(img, dtype=np.uint8)
    return rgb[:, :, ::-1].copy(), col1, col2


def render_video_frame_pil(
    lines: Iterable[str],
    *,
    lang: str = "en",
    width: int = 1280,
    height: int = 720,
    roi_box: tuple[float, float, float, float] = (0.05, 0.74, 0.90, 0.22),
    font_size: int = 32,
    watermark: Optional[str] = None,
) -> Image.Image:
    """Render one 720p video frame with dark subtitle band inside roi_box."""
    img = Image.new("RGB", (width, height), (24, 30, 40))
    draw = ImageDraw.Draw(img)

    rx, ry, rw, rh = roi_box
    x0 = int(round(rx * width))
    y0 = int(round(ry * height))
    x1 = min(width, int(round((rx + rw) * width)))
    y1 = min(height, int(round((ry + rh) * height)))
    draw.rectangle([x0, y0, x1, y1], fill=(12, 15, 20))

    line_list = [ln for ln in lines if ln]
    if line_list:
        font = load_font(lang=lang, size=font_size)
        band_h = max(1, y1 - y0)
        spacing = max(font_size + 10, min(48, (band_h - 12) // max(1, len(line_list))))
        total_h = font_size + (len(line_list) - 1) * spacing
        start_y = y0 + max(4, (band_h - total_h) // 2 - 2)
        for idx, line in enumerate(line_list):
            draw.text((x0 + 28, start_y + idx * spacing), line, font=font, fill=(255, 255, 255))

    if watermark:
        wm_font = load_font(lang="en", size=20)
        draw.text((max(x0 + 8, x1 - 150), y0 + 6), watermark, font=wm_font, fill=(210, 210, 210))

    return img


def build_segmented_video(
    out_path: Path | str,
    segments: list[tuple[float, list[str], str]],
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    with_audio: bool = False,
    watermark: Optional[str] = None,
    roi_box: tuple[float, float, float, float] = (0.05, 0.74, 0.90, 0.22),
    font_size: int = 32,
) -> Path:
    """Build a deterministic H.264 MP4 from (duration_sec, lines, lang_hint) segments."""
    ffmpeg = require_ffmpeg()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not segments:
        raise ValueError("segments must be non-empty")

    png_dir = out.parent / f".synth_{uuid.uuid4().hex[:8]}"
    png_dir.mkdir(parents=True, exist_ok=True)
    try:
        input_args: list[str] = []
        filter_inputs: list[str] = []
        total_dur = 0.0
        for idx, (dur_sec, lines, lang_hint) in enumerate(segments):
            dur = float(dur_sec)
            total_dur += dur
            img = render_video_frame_pil(
                lines,
                lang=lang_hint,
                width=width,
                height=height,
                roi_box=roi_box,
                font_size=font_size,
                watermark=watermark,
            )
            png_path = png_dir / f"seg_{idx:03d}.png"
            img.save(png_path)
            input_args.extend(["-loop", "1", "-t", f"{dur:.3f}", "-i", str(png_path)])
            filter_inputs.append(f"[{idx}:v]")

        n = len(segments)
        v_chain = f"{''.join(filter_inputs)}concat=n={n}:v=1:a=0,fps={fps},format=yuv420p[vout]"
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *input_args]
        if with_audio:
            cmd.extend([
                "-f", "lavfi",
                "-t", f"{total_dur:.3f}",
                "-i", "sine=frequency=440:sample_rate=16000",
                "-filter_complex", v_chain,
                "-map", "[vout]",
                "-map", f"{n}:a",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "stillimage",
                "-g", "15",
                "-c:a", "aac",
                "-ac", "1",
                "-ar", "16000",
                "-shortest",
                str(out),
            ])
        else:
            cmd.extend([
                "-filter_complex", v_chain,
                "-map", "[vout]",
                "-an",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "stillimage",
                "-g", "15",
                str(out),
            ])
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(png_dir, ignore_errors=True)
    return out


def make_wav(path: Path | str, duration_sec: float = 1.0, freq: float = 440.0, sample_rate: int = 24000) -> Path:
    """Create a deterministic mono PCM16 WAV file without external subprocesses."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    num_frames = max(1, int(round(duration_sec * sample_rate)))
    t = np.arange(num_frames, dtype=np.float64) / float(sample_rate)
    samples = (0.25 * 32767.0 * np.sin(2.0 * np.pi * freq * t)).astype(np.int16)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return p


class LocalOpenAITranslationServer:
    """In-process OpenAI-compatible HTTP server for offline AI translation E2E tests."""

    def __init__(
        self,
        *,
        fail_first_n_with_429: int = 0,
        wrap_think: bool = False,
        prefix: str = "[ZH]",
    ) -> None:
        self.fail_first_n_with_429 = fail_first_n_with_429
        self.wrap_think = wrap_think
        self.prefix = prefix
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._attempt = 0
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def _translate_text(self, text: str) -> str:
        clean = text.strip()
        if not clean:
            return clean
        return f"{self.prefix} {clean}"

    def _build_reply_content(self, messages: list[dict[str, Any]]) -> str:
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = str(m.get("content") or "")
                break

        inner = user_msg
        m_input = re.search(r"<INPUT>(.*?)</INPUT>", user_msg, flags=re.S | re.I)
        if m_input:
            inner = m_input.group(1).strip()

        # Detect SRT blocks: index\ntimestamp --> timestamp\ntext
        srt_blocks = re.findall(
            r"(\d+)\s*\r?\n(\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3})\s*\r?\n(.*?)(?=\r?\n\r?\n\d+\s*\r?\n\d{2}:\d{2}:\d{2},\d{3}\s*-->|\Z)",
            inner,
            flags=re.S,
        )
        if srt_blocks:
            out_blocks = []
            for idx, ts, body in srt_blocks:
                translated_lines = [self._translate_text(ln) for ln in body.strip().splitlines() if ln.strip()]
                out_blocks.append(f"{idx}\n{ts}\n" + "\n".join(translated_lines))
            body_text = "\n\n".join(out_blocks)
        else:
            # Compact [i] lines or plain lines
            out_lines = []
            for raw_line in inner.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                m_idx = re.match(r"^(\[\d+\])\s*(.*)$", line)
                if m_idx:
                    out_lines.append(f"{m_idx.group(1)} {self._translate_text(m_idx.group(2))}")
                else:
                    out_lines.append(self._translate_text(line))
            body_text = "\n".join(out_lines)

        reply = f"<TRANSLATE_TEXT>\n{body_text}\n</TRANSLATE_TEXT>"
        if self.wrap_think:
            reply = f"<think>\nStep 1: analyze lines\nStep 2: output strictly aligned\n</think>\n{reply}"
        return reply

    def __enter__(self) -> "LocalOpenAITranslationServer":
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path.rstrip("/") in {"/v1/models", "/models"}:
                    payload = json.dumps({"object": "list", "data": [{"id": "gpt-4o-mini", "object": "model"}]}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    data = {}

                with outer._lock:
                    outer._attempt += 1
                    attempt = outer._attempt
                    outer.requests.append(data)

                if attempt <= outer.fail_first_n_with_429:
                    err = json.dumps({
                        "error": {
                            "message": "Simulated 429 rate limit",
                            "type": "rate_limit_error",
                            "code": "rate_limit_exceeded",
                        }
                    }).encode("utf-8")
                    self.send_response(429)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Retry-After", "0")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return

                content = outer._build_reply_content(data.get("messages") or [])
                resp = {
                    "id": f"chatcmpl-{attempt}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": data.get("model", "gpt-4o-mini"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
                }
                payload = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def isolate_runtime_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate DB, TEMP_DIR/TEMP_ROOT, project output, voices, and config stores inside tmp_path."""
    from videotrans.api import catalog as api_catalog, task_params as api_task_params
    from videotrans.configure import _paths, config as runtime_config
    from videotrans.core import db as db_mod, media_store, project_store, secret_store, storage_config, voice_store
    from videotrans.task import orchestrator

    orig_db_path = db_mod.get_db_path()
    ws = tmp_path / "e2e_ws"
    temp_dir = ws / "tmp"
    output_dir = ws / "output"
    uploads_dir = ws / "uploads"
    voices_dir = ws / "voices"
    previews_dir = ws / "previews"
    models_dir = ws / "models"
    trans_cache_dir = temp_dir / "translate_cache"
    dub_cache_dir = temp_dir / "dubbing_cache"
    for d in (temp_dir, output_dir, uploads_dir, voices_dir, previews_dir, models_dir, trans_cache_dir, dub_cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    db_path = ws / "e2e.db"
    monkeypatch.setattr(db_mod, "_DB_PATH", db_path, raising=False)
    db_mod.set_db_path(db_path)
    db_mod.init_db(db_path)
    secret_store.invalidate()

    # Patch temp / storage roots so TaskRequest.normalize(), task_params, and project_store stay in tmp_path
    temp_str = str(temp_dir.resolve())
    monkeypatch.setattr(_paths, "TEMP_ROOT", temp_str, raising=False)
    monkeypatch.setattr(_paths, "TEMP_DIR", temp_str, raising=False)
    monkeypatch.setattr(_paths, "TRANSLATE_CACHE", str(trans_cache_dir.resolve()), raising=False)
    monkeypatch.setattr(_paths, "DUBBING_CACHE", str(dub_cache_dir.resolve()), raising=False)
    monkeypatch.setattr(runtime_config, "TEMP_ROOT", temp_str, raising=False)
    monkeypatch.setattr(runtime_config, "TEMP_DIR", temp_str, raising=False)
    monkeypatch.setattr(runtime_config, "TRANSLATE_CACHE", str(trans_cache_dir.resolve()), raising=False)
    monkeypatch.setattr(runtime_config, "DUBBING_CACHE", str(dub_cache_dir.resolve()), raising=False)
    monkeypatch.setattr(orchestrator, "TEMP_DIR", temp_str, raising=False)

    monkeypatch.setattr(api_task_params, "OUTPUT_DIR", output_dir, raising=False)
    monkeypatch.setattr(api_task_params, "TEMP_DIR", temp_str, raising=False)
    if getattr(api_task_params.build_task_params, "__kwdefaults__", None) is not None:
        monkeypatch.setitem(api_task_params.build_task_params.__kwdefaults__, "output_dir", output_dir)
        monkeypatch.setitem(api_task_params.build_task_params.__kwdefaults__, "temp_dir", temp_dir)
    monkeypatch.setattr(api_catalog, "OUTPUT_DIR", output_dir, raising=False)
    monkeypatch.setattr(api_catalog, "UPLOAD_DIR", uploads_dir, raising=False)
    monkeypatch.setattr(media_store, "UPLOAD_DIR", uploads_dir, raising=False)
    jobs_mod = sys.modules.get("videotrans.api.routes.jobs")
    if jobs_mod is not None and hasattr(jobs_mod, "TEMP_DIR"):
        monkeypatch.setattr(jobs_mod, "TEMP_DIR", temp_str, raising=False)

    # Isolate project_store output root and voice_store dirs
    projects_dir = output_dir / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(project_store, "DEFAULT_PROJECTS_DIR", projects_dir, raising=False)
    monkeypatch.setattr(project_store, "PROJECTS_OUTPUT_DIR", projects_dir, raising=False)
    monkeypatch.setattr(project_store, "get_projects_root", lambda: projects_dir)
    monkeypatch.setattr(voice_store, "VOICES_DIR", voices_dir, raising=False)
    monkeypatch.setattr(voice_store, "PREVIEWS_DIR", previews_dir, raising=False)
    monkeypatch.setattr(
        storage_config,
        "DEFAULT_STORAGE",
        {
            "output_dir": output_dir,
            "uploads_dir": uploads_dir,
            "models_dir": models_dir,
            "temp_dir": temp_dir,
        },
        raising=False,
    )

    # Isolate settings & params in-memory + redirect save paths to tmp_path without
    # conflicting with conftest.py's finally-block _json_path restoration order.
    orig_settings_data = copy.deepcopy(runtime_config.settings.to_dict())
    orig_params_data = copy.deepcopy(runtime_config.params.to_dict())
    orig_settings_json_path = getattr(runtime_config.settings, "_json_path", "")
    orig_params_json_path = getattr(runtime_config.params, "_json_path", "")
    ws_cfg = ws / "cfg.json"
    ws_params = ws / "params.json"
    if orig_settings_json_path and Path(orig_settings_json_path).is_file():
        shutil.copy2(orig_settings_json_path, ws_cfg)
    if orig_params_json_path and Path(orig_params_json_path).is_file():
        shutil.copy2(orig_params_json_path, ws_params)
    runtime_config.settings._json_path = str(ws_cfg)
    runtime_config.params._json_path = str(ws_params)

    # Isolate stage_dubbing DUBBING_CACHE if imported
    try:
        from videotrans.task import _stage_dubbing
        monkeypatch.setattr(_stage_dubbing, "DUBBING_CACHE", str(dub_cache_dir.resolve()), raising=False)
    except Exception:
        pass
    try:
        from videotrans.translator import _base as trans_base
        monkeypatch.setattr(trans_base, "TEMP_ROOT", temp_str, raising=False)
    except Exception:
        pass

    return {
        "root": ws,
        "db_path": db_path,
        "temp_dir": temp_dir,
        "output_dir": output_dir,
        "projects_dir": projects_dir,
        "uploads_dir": uploads_dir,
        "voices_dir": voices_dir,
        "previews_dir": previews_dir,
        "_orig_db": orig_db_path,  # type: ignore[dict-item]
        "_orig_settings": orig_settings_data,  # type: ignore[dict-item]
        "_orig_params": orig_params_data,  # type: ignore[dict-item]
        "_orig_settings_json_path": orig_settings_json_path,  # type: ignore[dict-item]
        "_orig_params_json_path": orig_params_json_path,  # type: ignore[dict-item]
    }


def restore_runtime_config(iso: dict[str, Any]) -> None:
    from videotrans.configure import config as runtime_config
    from videotrans.core import db as db_mod, secret_store

    db_mod.set_db_path(iso["_orig_db"])
    secret_store.invalidate()

    if "_orig_settings_json_path" in iso:
        runtime_config.settings._json_path = iso["_orig_settings_json_path"]
    if "_orig_params_json_path" in iso:
        runtime_config.params._json_path = iso["_orig_params_json_path"]

    for k, v in iso["_orig_settings"].items():
        if k == "hf_token":
            continue
        attr = "initial_prompt_zh_cn" if k == "initial_prompt_zh-cn" else ("initial_prompt_zh_tw" if k == "initial_prompt_zh-tw" else k)
        if hasattr(runtime_config.settings, attr):
            setattr(runtime_config.settings, attr, v)

    secret_keys = set(runtime_config.params._secret_keys()) if hasattr(runtime_config.params, "_secret_keys") else set()
    for k, v in iso["_orig_params"].items():
        if k in secret_keys:
            continue
        if hasattr(runtime_config.params, k):
            setattr(runtime_config.params, k, v)
