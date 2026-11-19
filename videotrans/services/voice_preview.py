# -*- coding: utf-8 -*-
"""Voice preview synthesis engine with concurrency lock, 30s timeout, and CPU fallback."""
from __future__ import annotations

import asyncio
from collections import deque
from functools import lru_cache
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import threading
import time
from typing import Any, Callable, Optional
import uuid
import wave
import weakref

from videotrans import tts
from videotrans.configure.config import params, ROOT_DIR
from videotrans.core import voice_store
from videotrans.tts._cache import prune_cache_dir, touch_cache_entry
from videotrans.util.help_ffmpeg import runffmpeg

logger = logging.getLogger("videotrans.voice_preview")

_gpu_thread_lock = threading.Lock()
_loop_provider_semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[int, asyncio.Semaphore]] = weakref.WeakKeyDictionary()

_active_batches_lock = threading.Lock()
_active_batch_cancels: dict[str, threading.Event] = {}

PREVIEW_TIMEOUT_SECONDS = 120.0
CLONE_TIMEOUT_SECONDS = 120.0
CLOUD_PROVIDER_CONCURRENCY = 4
VIENEU_BATCH_SIZE = 4
PREVIEWS_MAX_AGE_SECONDS = 7 * 86400
PREVIEWS_MAX_BYTES = 256 * 1024 * 1024
PREVIEWS_MAX_FILES = 500

_preview_synthesizer = None


class _CrossLoopAsyncLock:
    """Cancel-safe, non-blocking async mutex usable across multiple event loops."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locked = False
        self._waiters: deque[tuple[asyncio.AbstractEventLoop, asyncio.Future[None], list[str]]] = deque()

    async def acquire(self) -> bool:
        loop = asyncio.get_running_loop()
        fut: Optional[asyncio.Future[None]] = None
        state = ["waiting"]
        with self._guard:
            if not self._locked:
                self._locked = True
                state[0] = "acquired"
                return True
            fut = loop.create_future()
            self._waiters.append((loop, fut, state))

        try:
            await fut
            with self._guard:
                state[0] = "acquired"
            return True
        except BaseException:
            should_release = False
            with self._guard:
                if state[0] == "waiting":
                    state[0] = "aborted"
                    try:
                        self._waiters.remove((loop, fut, state))
                    except ValueError:
                        pass
                elif state[0] in ("waking", "acquired"):
                    state[0] = "aborted"
                    should_release = True
            if should_release:
                self.release()
            raise

    def release(self) -> None:
        while True:
            with self._guard:
                if not self._waiters:
                    self._locked = False
                    return
                loop, fut, state = self._waiters.popleft()
                if state[0] != "waiting":
                    continue
                state[0] = "waking"

            def _wake(f: asyncio.Future[None] = fut, st: list[str] = state) -> None:
                with self._guard:
                    if st[0] != "waking":
                        return
                    if f.done():
                        st[0] = "aborted"
                        should_release = True
                    else:
                        st[0] = "acquired"
                        f.set_result(None)
                        should_release = False
                if should_release:
                    self.release()

            try:
                loop.call_soon_threadsafe(_wake)
                return
            except RuntimeError:
                with self._guard:
                    state[0] = "aborted"
                continue

    async def __aenter__(self) -> _CrossLoopAsyncLock:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


_gpu_async_lock = _CrossLoopAsyncLock()
_path_locks_guard = threading.Lock()
_path_locks: weakref.WeakValueDictionary[str, _CrossLoopAsyncLock] = weakref.WeakValueDictionary()


def _get_gpu_lock() -> _CrossLoopAsyncLock:
    return _gpu_async_lock


def _get_provider_semaphore(provider: int, limit: int = CLOUD_PROVIDER_CONCURRENCY) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem_map = _loop_provider_semaphores.get(loop)
    if sem_map is None:
        sem_map = {}
        _loop_provider_semaphores[loop] = sem_map
    sem = sem_map.get(provider)
    if sem is None:
        sem = asyncio.Semaphore(max(1, int(limit)))
        sem_map[provider] = sem
    return sem


def get_path_lock(path: Path | str) -> _CrossLoopAsyncLock:
    key = os.path.normcase(str(Path(path).resolve()))
    with _path_locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = _CrossLoopAsyncLock()
            _path_locks[key] = lock
        return lock


_get_path_lock = get_path_lock


def register_batch_cancel(batch_id: str) -> threading.Event:
    ev = threading.Event()
    with _active_batches_lock:
        _active_batch_cancels[batch_id] = ev
    return ev


def unregister_batch_cancel(batch_id: str) -> None:
    with _active_batches_lock:
        _active_batch_cancels.pop(batch_id, None)


def cancel_batch(batch_id: str) -> bool:
    with _active_batches_lock:
        ev = _active_batch_cancels.get(batch_id)
        if ev is not None:
            ev.set()
            return True
    return False


def cleanup_previews_dir(exclude_names: Optional[set[str]] = None) -> int:
    """Apply bounded LRU/TTL eviction to voice_store.PREVIEWS_DIR while preserving saved voice previews."""
    try:
        previews_dir = Path(voice_store.PREVIEWS_DIR)
        return prune_cache_dir(
            previews_dir,
            max_age_seconds=PREVIEWS_MAX_AGE_SECONDS,
            max_bytes=PREVIEWS_MAX_BYTES,
            max_files=PREVIEWS_MAX_FILES,
            primary_suffixes=(".wav",),
            paired_suffixes=(".json",),
            exclude_names=exclude_names,
            preserve_prefixes=("voice_",),
            preserve_suffixes=("_preview.wav",),
        )
    except Exception as exc:
        logger.debug("Preview cache cleanup skipped: %s", exc)
        return 0


def set_preview_synthesizer(fn):
    """Set custom synthesizer function (useful for tests or mocking)."""
    global _preview_synthesizer
    _preview_synthesizer = fn

DEFAULT_TEXTS = {
    "vi": "Chào bạn, đây là bản nghe thử giọng nói trí tuệ nhân tạo được tổng hợp thành công.",
    "en": "Hello, this is a sample preview of your newly generated custom voice.",
}


def get_default_preview_text(language: str) -> str:
    lang = (language or "").strip().lower()
    if lang.startswith("vi"):
        return DEFAULT_TEXTS["vi"]
    return DEFAULT_TEXTS["en"]


def _is_valid_preview_wav(path: Path) -> bool:
    """Check if file exists, is a valid WAV, has > 0 frames, and is not completely silent."""
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        import wave
        with wave.open(str(path), "rb") as wf:
            nframes = wf.getnframes()
            if nframes <= 0:
                return False
            framerate = wf.getframerate()
            if framerate <= 0 or (nframes / framerate) < 0.1:
                return False
            frames = wf.readframes(min(nframes, framerate * 2))
            if not any(b != 0 for b in frames):
                return False
            return True
    except Exception:
        try:
            import soundfile as sf
            data, sr = sf.read(str(path))
            if len(data) == 0:
                return False
            import numpy as np
            peak = float(np.max(np.abs(data)))
            return peak > 1e-4
        except Exception:
            return False


def generate_snippet_preview(
    ref_audio_path: Path,
    output_path: Path,
    target_rate: int = 48000,
) -> bool:
    """Extract a 3-second normalized snippet from reference audio with fast fallback."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex[:6]}.tmp")
    cmd = [
        "-y",
        "-i", str(ref_audio_path),
        "-af", "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-40dB,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-t", "3",
        "-vn",
        "-ac", "1",
        "-ar", str(target_rate),
        "-acodec", "pcm_s16le",
        "-f", "wav",
        str(tmp_out),
    ]
    cmd_simple = [
        "-y",
        "-i", str(ref_audio_path),
        "-t", "3",
        "-vn",
        "-ac", "1",
        "-ar", str(target_rate),
        "-acodec", "pcm_s16le",
        "-f", "wav",
        str(tmp_out),
    ]
    try:
        try:
            runffmpeg(cmd, force_cpu=True)
        except Exception:
            pass

        if not _is_valid_preview_wav(tmp_out):
            try:
                runffmpeg(cmd_simple, force_cpu=True)
            except Exception:
                pass

        if _is_valid_preview_wav(tmp_out):
            _replace_file_with_retry(tmp_out, output_path)
            return True
        return False
    finally:
        _unlink_with_retry(tmp_out)


VIENEU_PRESET_MAP: dict[str, str] = {
    "bình": "Thanh Bình",
    "bình (nam miền bắc)": "Thanh Bình",
    "thanh bình": "Thanh Bình",
    "ly": "Trúc Ly",
    "ly (nữ miền bắc)": "Trúc Ly",
    "trúc ly": "Trúc Ly",
    "ngọc": "Ngọc Linh",
    "ngọc (nữ miền bắc)": "Ngọc Linh",
    "ngọc linh": "Ngọc Linh",
    "tuyên": "Phạm Tuyên",
    "tuyên (nam miền bắc)": "Phạm Tuyên",
    "phạm tuyên": "Phạm Tuyên",
    "vĩnh": "Xuân Vĩnh",
    "vĩnh (nam miền nam)": "Xuân Vĩnh",
    "xuân vĩnh": "Xuân Vĩnh",
    "đoan": "Thục Đoan",
    "đoan (nữ miền nam)": "Thục Đoan",
    "thục đoan": "Thục Đoan",
    "đoan trang": "Đoan Trang",
    "đức": "Minh Đức",
    "minh đức": "Minh Đức",
    "sơn": "Thái Sơn",
    "thái sơn": "Thái Sơn",
    "quang sơn": "Quang Sơn",
    "mai anh": "Mai Anh",
    "triết": "Minh Triết",
    "minh triết": "Minh Triết",
    "dung": "Thùy Dung",
    "thùy dung": "Thùy Dung",
    "trân": "Ngọc Trân",
    "ngọc trân": "Ngọc Trân",
}


def resolve_vieneu_preset_voice(voice_input: Any, engine: Any = None) -> Any:
    """Resolve regional, short, or full VieNeu preset voice names to valid preset objects or names."""
    if not voice_input:
        return None
    if isinstance(voice_input, dict):
        return voice_input
    if not isinstance(voice_input, str):
        return voice_input

    raw = voice_input.strip()
    norm = raw.lower()
    if norm in ("no", "default", "clone"):
        return None

    resolved_name: Optional[str] = VIENEU_PRESET_MAP.get(norm)
    if not resolved_name:
        for k, v in VIENEU_PRESET_MAP.items():
            if k in norm or norm in k:
                resolved_name = v
                break

    if not resolved_name:
        resolved_name = raw

    if engine and hasattr(engine, "get_preset_voice"):
        try:
            return engine.get_preset_voice(resolved_name)
        except Exception:
            # Fallback to available presets
            if hasattr(engine, "_preset_voices") and engine._preset_voices:
                avail = list(engine._preset_voices.keys())
                for a in avail:
                    if a.lower() in norm or norm in a.lower():
                        try:
                            return engine.get_preset_voice(a)
                        except Exception:
                            pass
                return engine.get_preset_voice(avail[0])
    return resolved_name


def _is_transient_lock_error(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32)


def _replace_file_with_retry(src: Path, dst: Path, retries: int = 8, delay: float = 0.015) -> None:
    """Replace ``dst`` with ``src`` retrying transient Windows PermissionError locks."""
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            if not _is_transient_lock_error(exc) or attempt + 1 >= retries:
                raise
            time.sleep(delay * (attempt + 1))


def _unlink_with_retry(path: Path, retries: int = 8, delay: float = 0.015, raise_on_error: bool = False) -> None:
    """Unlink ``path`` retrying transient Windows PermissionError locks."""
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError as exc:
            if not _is_transient_lock_error(exc) or attempt + 1 >= retries:
                if raise_on_error:
                    raise
                return
            time.sleep(delay * (attempt + 1))


def _save_normalized_wav(aud_item: Any, output_file: Path, sample_rate: int, label: str = "VieNeu") -> None:
    """Validate, detach tensor if needed, peak-normalize, and write a WAV file atomically."""
    import numpy as np
    import soundfile as sf

    if hasattr(aud_item, "detach"):
        aud_item = aud_item.detach().cpu().numpy()
    audio_data = np.asarray(aud_item, dtype=np.float32).squeeze()
    if audio_data.size == 0:
        raise RuntimeError(f"{label} returned 0 audio samples")
    if np.isnan(audio_data).any() or np.isinf(audio_data).any():
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
    peak = float(np.max(np.abs(audio_data))) if audio_data.size > 0 else 0.0
    if peak < 1e-4:
        raise RuntimeError(f"{label} returned silent audio (peak amplitude: {peak:.6f})")

    # Peak normalize to ~0.90 (-1 dBFS) for clear, audible voice
    audio_data = (audio_data / peak) * 0.90
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_file.with_name(f".{output_file.stem}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        sf.write(str(tmp_out), audio_data, int(sample_rate), format="WAV")
        _replace_file_with_retry(tmp_out, output_file)
    finally:
        _unlink_with_retry(tmp_out)


def _run_vieneu_synthesis(
    ref_audio: Optional[Path] = None,
    output_file: Optional[Path] = None,
    text: str = "",
    use_cuda: bool = True,
    voice: Optional[Any] = None,
    style: str = "tu_nhien",
    denoise: bool = True,
    temperature: float = 0.8,
    repetition_penalty: float = 1.2,
    top_p: float = 0.95,
    cached_embedding: Optional[tuple[Any, Optional[Any]]] = None,
    abort_event: Optional[threading.Event] = None,
) -> None:
    """Run VieNeu TTS inference synchronously with style, emotion tags, and tuning parameters using cached engine."""
    if output_file is None:
        raise ValueError("output_file is required")

    from videotrans.services.model_cache import get_cached_vieneu_engine, evict_vieneu_engine

    def _infer(device: str, backend: str):
        if abort_event is not None and abort_event.is_set():
            raise RuntimeError("Cancelled")
        with _gpu_thread_lock:
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("Cancelled")
            engine = get_cached_vieneu_engine(device=device, backend=backend, max_batch_size=VIENEU_BATCH_SIZE)
            kwargs: dict[str, Any] = {
                "style": style or "tu_nhien",
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "top_p": top_p,
            }
            if cached_embedding and cached_embedding[0] is not None:
                audios = engine.infer_batch(
                    [text],
                    voice={"speaker_emb": cached_embedding[0], "codes": cached_embedding[1]},
                    **kwargs,
                )
            elif ref_audio and ref_audio.is_file():
                audios = engine.infer_batch(
                    [text],
                    ref_audio=ref_audio.as_posix(),
                    denoise=denoise,
                    **kwargs,
                )
            elif voice:
                v_target = resolve_vieneu_preset_voice(voice, engine)
                audios = engine.infer_batch(
                    [text],
                    voice=v_target,
                    **kwargs,
                )
            else:
                audios = engine.infer_batch([text], **kwargs)

            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("Cancelled")
            if not audios or len(audios) == 0:
                raise RuntimeError("VieNeu returned empty audio list")
            _save_normalized_wav(audios[0], output_file, engine.sample_rate, label="VieNeu")

    can_use_cuda = use_cuda
    if can_use_cuda:
        try:
            import torch
            can_use_cuda = torch.cuda.is_available()
        except Exception:
            can_use_cuda = False

    try:
        if can_use_cuda:
            try:
                _infer("cuda", "pytorch")
                return
            except Exception as exc:
                if abort_event is not None and abort_event.is_set():
                    raise
                logger.warning("CUDA error during VieNeu preview synthesis (%s), falling back to CPU", exc)
                evict_vieneu_engine(device="cuda")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        # CPU fallback
        _infer("cpu", "onnx")
    except Exception as exc:
        if abort_event is None or not abort_event.is_set():
            logger.error("VieNeu preview synthesis failed: %s", exc, exc_info=True)
        raise
    finally:
        if abort_event is not None and abort_event.is_set():
            _unlink_with_retry(output_file)


def _run_vieneu_batch_synthesis(
    batch_items: list[dict[str, Any]],
    *,
    ref_audio: Optional[Path] = None,
    use_cuda: bool = True,
    voice: Optional[Any] = None,
    style: str = "tu_nhien",
    denoise: bool = True,
    temperature: float = 0.8,
    repetition_penalty: float = 1.2,
    top_p: float = 0.95,
    cached_embedding: Optional[tuple[Any, Optional[Any]]] = None,
    abort_event: Optional[threading.Event] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[tuple[bool, Optional[str]]]:
    """Run a homogeneous VieNeu batch (<=4 items sharing voice/style/tuning) synchronously."""
    if not batch_items:
        return []
    from videotrans.services.model_cache import get_cached_vieneu_engine, evict_vieneu_engine

    texts = [str(it["text"]) for it in batch_items]

    def _is_aborted_before_infer() -> bool:
        return (abort_event is not None and abort_event.is_set()) or (
            cancel_event is not None and cancel_event.is_set()
        )

    def _cleanup_batch_targets() -> None:
        for item in batch_items:
            _unlink_with_retry(item["preview_path"])

    def _infer_batch(device: str, backend: str) -> list[tuple[bool, Optional[str]]]:
        if _is_aborted_before_infer():
            _cleanup_batch_targets()
            return [(False, "Cancelled")] * len(batch_items)
        with _gpu_thread_lock:
            if _is_aborted_before_infer():
                _cleanup_batch_targets()
                return [(False, "Cancelled")] * len(batch_items)
            engine = get_cached_vieneu_engine(device=device, backend=backend, max_batch_size=VIENEU_BATCH_SIZE)
            kwargs: dict[str, Any] = {
                "style": style or "tu_nhien",
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "top_p": top_p,
                "batch_size": VIENEU_BATCH_SIZE,
            }
            if cached_embedding and cached_embedding[0] is not None:
                kwargs["voice"] = {"speaker_emb": cached_embedding[0], "codes": cached_embedding[1]}
            elif ref_audio and ref_audio.is_file():
                kwargs["ref_audio"] = ref_audio.as_posix()
                kwargs["denoise"] = denoise
            elif voice:
                kwargs["voice"] = resolve_vieneu_preset_voice(voice, engine)

            try:
                audios = engine.infer_batch(texts, **kwargs)
            except TypeError:
                # Some test doubles or older wrappers may not accept batch_size kwarg
                kwargs.pop("batch_size", None)
                audios = engine.infer_batch(texts, **kwargs)

            if abort_event is not None and abort_event.is_set():
                _cleanup_batch_targets()
                return [(False, "Cancelled")] * len(batch_items)

            if not audios or len(audios) != len(batch_items):
                raise RuntimeError(
                    f"VieNeu batch returned {len(audios) if audios else 0} clips for {len(batch_items)} texts"
                )
            sr = int(getattr(engine, "sample_rate", 24000) or 24000)
            outcomes: list[tuple[bool, Optional[str]]] = []
            try:
                for item, aud in zip(batch_items, audios):
                    if abort_event is not None and abort_event.is_set():
                        outcomes.append((False, "Cancelled"))
                        continue
                    out_path: Path = item["preview_path"]
                    try:
                        _save_normalized_wav(aud, out_path, sr, label="VieNeu")
                        outcomes.append((True, None))
                    except Exception as exc:
                        _unlink_with_retry(out_path)
                        outcomes.append((False, str(exc)))
                return outcomes
            finally:
                if abort_event is not None and abort_event.is_set():
                    _cleanup_batch_targets()

    can_use_cuda = use_cuda
    if can_use_cuda:
        try:
            import torch
            can_use_cuda = torch.cuda.is_available()
        except Exception:
            can_use_cuda = False

    if can_use_cuda:
        try:
            return _infer_batch("cuda", "pytorch")
        except Exception as exc:
            if _is_aborted_before_infer():
                _cleanup_batch_targets()
                return [(False, "Cancelled")] * len(batch_items)
            logger.warning("CUDA error during VieNeu batch synthesis (%s), falling back to CPU", exc)
            evict_vieneu_engine(device="cuda")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    return _infer_batch("cpu", "onnx")


def _run_omnivoice_synthesis(
    ref_audio: Path,
    ref_text: str,
    output_file: Path,
    text: str,
    use_cuda: bool = True,
    speed: float = 1.0,
    abort_event: Optional[threading.Event] = None,
) -> None:
    """Run OmniVoice TTS preview synthesis with cached model, falling back to CPU if needed."""
    from videotrans.configure.config import ROOT_DIR
    model_dir = Path(ROOT_DIR) / "models" / "models--k2-fsa--OmniVoice"
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"OmniVoice model not installed at {model_dir}")

    import numpy as np
    import soundfile as sf
    import torch
    from videotrans.services.model_cache import get_cached_omnivoice_model, evict_omnivoice_model
    from videotrans.util import gpus

    def _infer(device: str, dtype: Any):
        if abort_event is not None and abort_event.is_set():
            raise RuntimeError("Cancelled")
        with _gpu_thread_lock:
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("Cancelled")
            model = get_cached_omnivoice_model(model_dir=model_dir, device=device, dtype=dtype)
            wav = model.generate(
                text=text,
                ref_audio=ref_audio.as_posix(),
                ref_text=ref_text or None,
                speed=speed or 1.0,
            )
            if abort_event is not None and abort_event.is_set():
                raise RuntimeError("Cancelled")
            if not wav or len(wav) == 0:
                raise RuntimeError("OmniVoice returned empty audio list")
            _save_normalized_wav(wav[0], output_file, 24000, label="OmniVoice")

    is_gpu = use_cuda and torch.cuda.is_available()
    try:
        if is_gpu:
            try:
                _infer("cuda:0", torch.float16)
                return
            except Exception as exc:
                if abort_event is not None and abort_event.is_set():
                    raise
                logger.warning("CUDA error during OmniVoice preview synthesis (%s), falling back to CPU", exc)
                evict_omnivoice_model(model_dir=model_dir, device="cuda")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        _infer(gpus.mps_or_cpu(), torch.float32)
    finally:
        if abort_event is not None and abort_event.is_set():
            _unlink_with_retry(output_file)


async def synthesize_voice_preview(
    voice_id: str,
    text: Optional[str] = None,
    language: Optional[str] = None,
    use_cuda: bool = True,
    force_refresh: bool = False,
    db_path: Optional[str | Path] = None,
) -> Path:
    """
    Synthesize a single-utterance audio preview for a custom voice.
    Guarded by an asyncio concurrency lock and a 30s timeout.
    """
    voice = voice_store.get_voice(voice_id, db_path=db_path)
    if not voice:
        raise ValueError(f"Voice {voice_id} not found")

    voice_store.init_voice_dirs()
    preview_filename = f"{voice_id}_preview.wav"
    preview_path = voice_store.get_preview_audio_path(preview_filename)

    # 1. Determine sample utterance
    lang = language or voice.get("language") or "vi"
    sample_text = (text or "").strip() or get_default_preview_text(lang)

    # 2. Resolve voice hierarchy (handling designed voices)
    resolved = voice_store.resolve_voice_params(voice_id, db_path=db_path) or voice
    tuning = dict(resolved.get("tuning_params") or {})
    style = str(tuning.get("style") or "tu_nhien")
    temp = float(tuning.get("temperature", 0.8))
    rep_penalty = float(tuning.get("repetition_penalty", 1.2))
    top_p_val = float(tuning.get("top_p", 0.95))
    denoise_opt = bool(tuning.get("denoise", True))

    ref_audio_rel = resolved.get("ref_audio_path", "")
    ref_audio_path: Optional[Path] = None
    if ref_audio_rel:
        try:
            ref_audio_path = voice_store.get_voice_audio_path(ref_audio_rel)
            if not ref_audio_path.is_file():
                ref_audio_path = None
        except Exception:
            ref_audio_path = None

    eff_vid = resolved.get("effective_voice_id") or voice_id
    cached_emb = voice_store.get_voice_embedding(eff_vid)

    async with _get_path_lock(preview_path):
        # Check cache inside lock to avoid concurrent unlocked reads/writes
        if not force_refresh and _is_valid_preview_wav(preview_path):
            if preview_path.stat().st_mtime >= (voice.get("updated_at", 0) - 2.0):
                touch_cache_entry(preview_path)
                return preview_path

        provider = voice.get("provider", tts.VIENEU_TTS)
        call_abort = threading.Event()

        async def _do_synthesis() -> None:
            if _preview_synthesizer is not None:
                await asyncio.to_thread(_preview_synthesizer, voice, preview_path, sample_text)
                return

            # Check if neural engine can be loaded
            if provider == tts.VIENEU_TTS:
                try:
                    if resolved.get("base_type") == "preset":
                        preset_name = resolved.get("preset_voice") or voice.get("external_voice_id")
                        async with _get_gpu_lock():
                            await asyncio.to_thread(
                                _run_vieneu_synthesis,
                                None,
                                preview_path,
                                sample_text,
                                use_cuda,
                                voice=preset_name,
                                style=style,
                                temperature=temp,
                                repetition_penalty=rep_penalty,
                                top_p=top_p_val,
                                abort_event=call_abort,
                            )
                        return
                    elif ref_audio_path or cached_emb:
                        async with _get_gpu_lock():
                            await asyncio.to_thread(
                                _run_vieneu_synthesis,
                                ref_audio_path,
                                preview_path,
                                sample_text,
                                use_cuda,
                                style=style,
                                denoise=denoise_opt,
                                temperature=temp,
                                repetition_penalty=rep_penalty,
                                top_p=top_p_val,
                                cached_embedding=cached_emb,
                                abort_event=call_abort,
                            )
                        return
                except ImportError:
                    logger.info("vieneu package not installed, using fallback")
                except Exception as exc:
                    logger.warning("VieNeu synthesis error, falling back: %s", exc)

            if provider == tts.OMNIVOICE_TTS and ref_audio_path:
                try:
                    ref_text = voice.get("ref_text", "")
                    async with _get_gpu_lock():
                        await asyncio.to_thread(
                            _run_omnivoice_synthesis,
                            ref_audio_path,
                            ref_text,
                            preview_path,
                            sample_text,
                            use_cuda,
                            abort_event=call_abort,
                        )
                    return
                except ImportError:
                    logger.info("omnivoice package not installed, using reference audio snippet fallback")
                except Exception as exc:
                    logger.warning("OmniVoice synthesis error, falling back to snippet preview: %s", exc)

            # Robust fallback: generate 3-second preview snippet from reference audio
            if ref_audio_path and ref_audio_path.is_file():
                if await asyncio.to_thread(
                    generate_snippet_preview,
                    ref_audio_path,
                    preview_path,
                    48000 if provider == tts.VIENEU_TTS else 24000,
                ):
                    return

            # Pure synthetic wav fallback for design voices without source audio
            generate_fallback_preview_wav(preview_path)
            return

        try:
            await asyncio.wait_for(_do_synthesis(), timeout=PREVIEW_TIMEOUT_SECONDS)
        except (asyncio.TimeoutError, TimeoutError, asyncio.CancelledError, Exception) as exc:
            call_abort.set()
            logger.warning("Voice preview synthesis timed out or failed (%s), generating fallback preview audio", exc)
            if ref_audio_path and ref_audio_path.is_file():
                await asyncio.to_thread(
                    generate_snippet_preview,
                    ref_audio_path,
                    preview_path,
                    48000 if provider == tts.VIENEU_TTS else 24000,
                )
            if not _is_valid_preview_wav(preview_path):
                generate_fallback_preview_wav(preview_path)

        if not _is_valid_preview_wav(preview_path):
            generate_fallback_preview_wav(preview_path)

        # Update voice record with cached preview path if needed
        if voice.get("preview_audio_path") != preview_filename:
            voice_store.update_voice(voice_id, preview_audio_path=preview_filename, db_path=db_path)

        touch_cache_entry(preview_path)

    cleanup_previews_dir(exclude_names={preview_path.name})
    return preview_path


def generate_fallback_preview_wav(path: Path, duration_sec: float = 1.5, sample_rate: int = 24000) -> Path:
    """Generate a valid PCM 16-bit mono WAV preview file as safe fallback."""
    import array
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    tmp_out = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        with wave.open(str(tmp_out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            samples = [
                int(8000 * max(0.0, 1.0 - (i / num_samples)) * math.sin(2 * math.pi * 440.0 * (i / sample_rate)))
                for i in range(num_samples)
            ]
            wf.writeframes(array.array("h", samples).tobytes())
        _replace_file_with_retry(tmp_out, path)
    finally:
        _unlink_with_retry(tmp_out)
    return path


async def synthesize_clone_preview(
    audio_path: str | Path,
    provider: int = tts.VIENEU_TTS,
    text: Optional[str] = None,
    language: Optional[str] = None,
    use_cuda: bool = True,
    ref_text: Optional[str] = None,
    preview_id: Optional[str] = None,
    denoise: bool = False,
) -> tuple[str, Path]:
    """
    Synthesize a voice preview directly from a reference audio file before saving the voice.
    Returns (preview_id, preview_path).
    """
    voice_store.init_voice_dirs()
    pid = preview_id or f"preview_clone_{uuid.uuid4().hex[:10]}"
    clean_pid = "".join(c for c in pid if c.isalnum() or c in ("-", "_")) or f"preview_clone_{uuid.uuid4().hex[:10]}"
    preview_filename = f"{clean_pid}.wav"
    preview_path = voice_store.get_preview_audio_path(preview_filename)

    lang = language or "vi"
    sample_text = (text or "").strip() or get_default_preview_text(lang)
    ref_src = Path(audio_path).resolve()

    async with _get_path_lock(preview_path):
        call_abort = threading.Event()

        async def _do_synthesis() -> None:
            if _preview_synthesizer is not None:
                voice_dict = {
                    "id": pid,
                    "name": "Clone Preview",
                    "provider": provider,
                    "language": lang,
                    "ref_audio_path": str(ref_src),
                }
                await asyncio.to_thread(_preview_synthesizer, voice_dict, preview_path, sample_text)
                return

            if provider == tts.VIENEU_TTS and ref_src.is_file():
                try:
                    async with _get_gpu_lock():
                        await asyncio.to_thread(
                            _run_vieneu_synthesis,
                            ref_src,
                            preview_path,
                            sample_text,
                            use_cuda,
                            denoise=denoise,
                            abort_event=call_abort,
                        )
                    return
                except ImportError:
                    logger.info("vieneu package not installed, falling back to snippet preview")
                except Exception as exc:
                    logger.warning("VieNeu clone preview synthesis failed: %s", exc)

            elif provider == tts.OMNIVOICE_TTS and ref_src.is_file():
                try:
                    async with _get_gpu_lock():
                        await asyncio.to_thread(
                            _run_omnivoice_synthesis,
                            ref_src,
                            ref_text or "",
                            preview_path,
                            sample_text,
                            use_cuda,
                            abort_event=call_abort,
                        )
                    return
                except ImportError:
                    logger.info("omnivoice package not installed, falling back to snippet preview")
                except Exception as exc:
                    logger.warning("OmniVoice clone preview synthesis failed: %s", exc)

            # Robust snippet fallback from reference audio
            if ref_src.is_file():
                if await asyncio.to_thread(
                    generate_snippet_preview,
                    ref_src,
                    preview_path,
                    48000 if provider == tts.VIENEU_TTS else 24000,
                ):
                    return

            generate_fallback_preview_wav(preview_path)

        try:
            await asyncio.wait_for(_do_synthesis(), timeout=CLONE_TIMEOUT_SECONDS)
        except (asyncio.TimeoutError, TimeoutError, asyncio.CancelledError, Exception) as exc:
            call_abort.set()
            logger.warning("Clone preview synthesis timed out or failed (%s), generating fallback preview audio", exc)
            if ref_src and ref_src.is_file():
                await asyncio.to_thread(
                    generate_snippet_preview,
                    ref_src,
                    preview_path,
                    48000 if provider == tts.VIENEU_TTS else 24000,
                )
            if not _is_valid_preview_wav(preview_path):
                generate_fallback_preview_wav(preview_path)

        if not _is_valid_preview_wav(preview_path):
            generate_fallback_preview_wav(preview_path)

    cleanup_previews_dir(exclude_names={preview_path.name})
    return pid, preview_path


@lru_cache(maxsize=4)
def _load_elevenlabs_voice_map(json_path: str, mtime_ns: int, size: int) -> dict[str, str]:
    try:
        raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {
                str(name): str(info.get("voice_id"))
                for name, info in raw.items()
                if isinstance(info, dict) and info.get("voice_id")
            }
    except Exception:
        pass
    return {}


def _resolve_elevenlabs_voice_id(voice_str: str, cv_resolved: Optional[dict[str, Any]]) -> str:
    if cv_resolved and cv_resolved.get("external_voice_id"):
        return str(cv_resolved["external_voice_id"]).strip()
    if not voice_str or voice_str.lower() in ("no", "default"):
        return "21m00Tcm4TlvDq8ikWAM"
    json_file = Path(ROOT_DIR) / "videotrans" / "voicejson" / "elevenlabs.json"
    try:
        st = json_file.stat()
        vmap = _load_elevenlabs_voice_map(str(json_file), st.st_mtime_ns, st.st_size)
        if voice_str in vmap:
            return vmap[voice_str]
    except OSError:
        pass
    return voice_str


def _run_elevenlabs_preview(
    preview_path: Path,
    sample_text: str,
    target_voice: str,
    speed: Optional[float] = None,
) -> None:
    key = params.get("elevenlabstts_key", "")
    if not key:
        raise ValueError("ElevenLabs API key not configured")
    from elevenlabs import ElevenLabs, VoiceSettings

    client = ElevenLabs(api_key=key)
    el_model = (params.get("elevenlabstts_models") or "eleven_multilingual_v2").strip()
    if el_model.startswith("eleven_v3"):
        el_model = "eleven_multilingual_v2"
    voice_kwargs: dict[str, Any] = {
        "text": sample_text,
        "voice_id": target_voice,
        "model_id": el_model,
    }
    if speed is not None:
        voice_kwargs["voice_settings"] = VoiceSettings(speed=speed)

    resp = client.text_to_speech.convert(**voice_kwargs)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_raw = preview_path.with_name(f".{preview_path.stem}.{uuid.uuid4().hex[:6]}.raw")
    tmp_wav = preview_path.with_name(f".{preview_path.stem}.{uuid.uuid4().hex[:6]}.wav")
    try:
        with open(tmp_raw, "wb") as f:
            for chunk in resp:
                if chunk:
                    f.write(chunk)
        if not tmp_raw.is_file() or tmp_raw.stat().st_size == 0:
            raise RuntimeError("ElevenLabs returned empty audio")
        with open(tmp_raw, "rb") as fh:
            magic = fh.read(4)
        if magic == b"RIFF":
            _replace_file_with_retry(tmp_raw, preview_path)
        else:
            runffmpeg(
                ["-y", "-i", str(tmp_raw), "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(tmp_wav)],
                force_cpu=True,
            )
            if not tmp_wav.is_file() or tmp_wav.stat().st_size <= 44:
                raise RuntimeError("Failed to transcode ElevenLabs preview audio to WAV")
            _replace_file_with_retry(tmp_wav, preview_path)
    finally:
        _unlink_with_retry(tmp_raw)
        _unlink_with_retry(tmp_wav)


def _run_gemini_preview(
    preview_path: Path,
    sample_text: str,
    target_voice: str,
    language: Optional[str] = None,
) -> None:
    key = params.get("gemini_key", "")
    if not key:
        raise ValueError("Gemini API key not configured")
    from videotrans.tts._geminitts import GEMINITTS

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = preview_path.with_name(f".{preview_path.stem}.{uuid.uuid4().hex[:6]}.tmp")
    tts_inst = GEMINITTS(language=language or "vi", uuid="preview")
    try:
        gem_tts_model = (params.get("gemini_ttsmodel") or "gemini-2.5-flash-preview-tts").strip()
        if gem_tts_model.startswith("gemini-3.1"):
            gem_tts_model = "gemini-2.5-flash-preview-tts"
        tts_inst.generate_tts_segment(
            sample_text,
            target_voice,
            gem_tts_model,
            str(tmp_out),
        )
        _replace_file_with_retry(tmp_out, preview_path)
    finally:
        _unlink_with_retry(tmp_out)
        if hasattr(tts_inst, "_close_clients"):
            tts_inst._close_clients()


def _write_sidecar_atomic(path: Path, applied_speed: float) -> None:
    sidecar = path.with_suffix(".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp_sidecar = sidecar.with_name(f".{sidecar.stem}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        tmp_sidecar.write_text(json.dumps({"applied_speed": applied_speed}), encoding="utf-8")
        _replace_file_with_retry(tmp_sidecar, sidecar)
    finally:
        _unlink_with_retry(tmp_sidecar)


def _read_cached_speed(preview_path: Path) -> Optional[float]:
    try:
        return float(json.loads(preview_path.with_suffix(".json").read_text(encoding="utf-8"))["applied_speed"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _resolve_preview_context(
    *,
    provider: int,
    voice: str,
    text: Optional[str] = None,
    language: Optional[str] = None,
    speed: Optional[float] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
    style: Optional[str] = None,
    tuning_params: Optional[dict[str, Any]] = None,
    db_path: Optional[str | Path] = None,
    auto_speed: bool = False,
    slot_duration_s: Optional[float] = None,
    max_speed_rate: Optional[float] = 1.35,
) -> dict[str, Any]:
    sample_text = (text or "").strip() or get_default_preview_text(language or "vi")
    voice_str = str(voice or "").strip()
    tp = dict(tuning_params or {})

    clean_voice = voice_str[len("Custom: "):].strip() if voice_str.startswith("Custom: ") else voice_str
    cv = voice_store.get_voice(clean_voice, db_path=db_path) if clean_voice else None
    if not cv and clean_voice:
        cv = voice_store.find_voice_by_name(clean_voice, provider=provider, db_path=db_path)
    if not cv and clean_voice and provider is not None:
        cv = voice_store.find_voice_by_name(clean_voice, provider=None, db_path=db_path)

    cv_resolved = None
    cv_ref_path: Optional[Path] = None
    cv_cached_emb = None
    if cv:
        cv_resolved = voice_store.resolve_voice_params(cv["id"], db_path=db_path) or cv
        cv_tp = dict(cv_resolved.get("tuning_params") or {})
        for k, v in cv_tp.items():
            if k not in tp and v is not None:
                tp[k] = v

        ref_rel = cv_resolved.get("ref_audio_path", "")
        if ref_rel:
            try:
                p = voice_store.get_voice_audio_path(ref_rel)
                if p.is_file():
                    cv_ref_path = p
            except Exception:
                pass

        eff_vid = cv_resolved.get("effective_voice_id") or cv["id"]
        cv_cached_emb = voice_store.get_voice_embedding(eff_vid)
        if not language and cv.get("language"):
            language = cv["language"]
        if cv.get("provider") is not None:
            provider = cv["provider"]
        voice_str = cv["id"]

    if _preview_synthesizer is not None:
        key_raw = f"{provider}_{voice_str}_{sample_text}_{language}_{speed}_{style}_{auto_speed}_{slot_duration_s}_{max_speed_rate}"
    else:
        tuning_key = json.dumps(tp, sort_keys=True)
        key_raw = f"{provider}_{voice_str}_{sample_text}_{language}_{speed}_{rate}_{pitch}_{style}_{tuning_key}_{auto_speed}_{slot_duration_s}_{max_speed_rate}"

    preview_id = f"prev_{hashlib.md5(key_raw.encode('utf-8')).hexdigest()[:12]}"
    preview_path = voice_store.get_preview_audio_path(f"{preview_id}.wav")

    return {
        "provider": provider,
        "voice_str": voice_str,
        "raw_voice": str(voice or "").strip(),
        "sample_text": sample_text,
        "language": language,
        "speed": speed,
        "rate": rate,
        "pitch": pitch,
        "style": style,
        "tp": tp,
        "cv": cv,
        "cv_resolved": cv_resolved,
        "cv_ref_path": cv_ref_path,
        "cv_cached_emb": cv_cached_emb,
        "auto_speed": bool(auto_speed),
        "slot_duration_s": slot_duration_s,
        "max_speed_rate": max_speed_rate or 1.35,
        "preview_id": preview_id,
        "preview_path": preview_path,
    }


async def _finish_preview(
    path: Path,
    preview_id: str,
    *,
    auto_speed: bool = False,
    slot_duration_s: Optional[float] = None,
    max_speed_rate: Optional[float] = 1.35,
) -> tuple[str, Path, float]:
    applied_speed = 1.0
    try:
        if auto_speed and slot_duration_s and slot_duration_s > 0:
            from videotrans.services.audio_fit import smart_fit_audio_file
            _, applied_speed, fit_status = await asyncio.to_thread(
                smart_fit_audio_file, path, slot_duration_s, max_speed_rate or 1.35,
            )
            if fit_status in {"error", "not_found"}:
                raise RuntimeError("Could not adjust preview audio to fit the subtitle timing")
        await asyncio.to_thread(_write_sidecar_atomic, path, applied_speed)
    except BaseException:
        _unlink_with_retry(path)
        _unlink_with_retry(path.with_suffix(".json"))
        raise
    return preview_id, path, applied_speed


async def _synthesize_ctx_uncached(
    ctx: dict[str, Any],
    use_cuda: bool = True,
    abort_event: Optional[threading.Event] = None,
) -> None:
    provider = ctx["provider"]
    voice_str = ctx["voice_str"]
    sample_text = ctx["sample_text"]
    language = ctx["language"]
    speed = ctx["speed"]
    style = ctx["style"]
    tp = ctx["tp"]
    cv = ctx["cv"]
    cv_resolved = ctx["cv_resolved"]
    cv_ref_path = ctx["cv_ref_path"]
    cv_cached_emb = ctx["cv_cached_emb"]
    preview_path: Path = ctx["preview_path"]

    if provider == tts.VIENEU_TTS:
        from videotrans.util.help_role import get_vieneu_custom_voice_path
        custom_ref = cv_ref_path or get_vieneu_custom_voice_path(voice_str)
        selected_style = style or tp.get("style", "tu_nhien")
        temp_val = float(tp.get("temperature", 0.8))
        rep_pen = float(tp.get("repetition_penalty", 1.2))
        top_p_val = float(tp.get("top_p", 0.95))
        denoise_opt = bool(tp.get("denoise", True))

        if cv_resolved and cv_resolved.get("base_type") == "preset":
            preset_name = cv_resolved.get("preset_voice") or (cv.get("external_voice_id") if cv else None)
            async with _get_gpu_lock():
                await asyncio.to_thread(
                    _run_vieneu_synthesis,
                    None,
                    preview_path,
                    sample_text,
                    use_cuda,
                    voice=preset_name,
                    style=selected_style,
                    temperature=temp_val,
                    repetition_penalty=rep_pen,
                    top_p=top_p_val,
                    abort_event=abort_event,
                )
            return
        elif (custom_ref and Path(custom_ref).is_file()) or cv_cached_emb:
            async with _get_gpu_lock():
                await asyncio.to_thread(
                    _run_vieneu_synthesis,
                    Path(custom_ref) if custom_ref else None,
                    preview_path,
                    sample_text,
                    use_cuda,
                    style=selected_style,
                    denoise=denoise_opt,
                    temperature=temp_val,
                    repetition_penalty=rep_pen,
                    top_p=top_p_val,
                    cached_embedding=cv_cached_emb,
                    abort_event=abort_event,
                )
            return

        preset_v = voice_str if (voice_str and voice_str.lower() not in ("no", "default", "clone")) else None
        async with _get_gpu_lock():
            await asyncio.to_thread(
                _run_vieneu_synthesis,
                None,
                preview_path,
                sample_text,
                use_cuda,
                voice=preset_v,
                style=selected_style,
                temperature=temp_val,
                repetition_penalty=rep_pen,
                top_p=top_p_val,
                abort_event=abort_event,
            )
        return

    if provider == tts.OMNIVOICE_TTS:
        from videotrans.configure.config import ROOT_DIR as CFG_ROOT_DIR
        model_dir = Path(CFG_ROOT_DIR) / "models" / "models--k2-fsa--OmniVoice"
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"OmniVoice model not installed at {model_dir}")
        omnivoice_ref = cv_ref_path
        if omnivoice_ref and Path(omnivoice_ref).is_file():
            ref_text = (cv.get("ref_text") if cv else "") or ""
            async with _get_gpu_lock():
                await asyncio.to_thread(
                    _run_omnivoice_synthesis,
                    Path(omnivoice_ref),
                    ref_text,
                    preview_path,
                    sample_text,
                    use_cuda,
                    speed=speed or 1.0,
                    abort_event=abort_event,
                )
            return
        raise NotImplementedError("OmniVoice preset preview requires clone audio")

    if provider == tts.ELEVENLABS_TTS:
        target_voice = _resolve_elevenlabs_voice_id(voice_str, cv_resolved)
        async with _get_provider_semaphore(tts.ELEVENLABS_TTS, CLOUD_PROVIDER_CONCURRENCY):
            await asyncio.to_thread(
                _run_elevenlabs_preview,
                preview_path,
                sample_text,
                target_voice,
                speed,
            )
        return

    if provider == tts.GEMINI_TTS:
        target_voice = voice_str if (voice_str and voice_str.lower() not in ("no", "default")) else "Puck"
        async with _get_provider_semaphore(tts.GEMINI_TTS, CLOUD_PROVIDER_CONCURRENCY):
            await asyncio.to_thread(
                _run_gemini_preview,
                preview_path,
                sample_text,
                target_voice,
                language,
            )
        return

    raise RuntimeError(f"Unsupported or unconfigured provider {provider}")


async def synthesize_unified_tts_preview(
    provider: int,
    voice: str,
    text: Optional[str] = None,
    language: Optional[str] = None,
    force_refresh: bool = False,
    speed: Optional[float] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
    style: Optional[str] = None,
    tuning_params: Optional[dict[str, Any]] = None,
    use_cuda: bool = True,
    db_path: Optional[str | Path] = None,
    auto_speed: bool = False,
    slot_duration_s: Optional[float] = None,
    max_speed_rate: Optional[float] = 1.35,
) -> tuple[str, Path, float]:
    """
    Unified voice preview generator supporting all 4 TTS providers
    (0: ElevenLabs, 1: OmniVoice, 2: VieNeu-TTS, 3: Gemini TTS) and custom voices.
    Returns (preview_id, preview_file_path, applied_speed).
    """
    voice_store.init_voice_dirs()
    ctx = _resolve_preview_context(
        provider=provider,
        voice=voice,
        text=text,
        language=language,
        speed=speed,
        rate=rate,
        pitch=pitch,
        style=style,
        tuning_params=tuning_params,
        db_path=db_path,
        auto_speed=auto_speed,
        slot_duration_s=slot_duration_s,
        max_speed_rate=max_speed_rate,
    )
    preview_id = ctx["preview_id"]
    preview_path: Path = ctx["preview_path"]

    if _preview_synthesizer is not None:
        voice_dict = {
            "id": preview_id,
            "name": (ctx["cv"].get("name") if ctx["cv"] else None) or ctx["voice_str"] or "Default",
            "provider": ctx["provider"],
            "language": ctx["language"] or "vi",
            "style": ctx["style"],
        }
        if ctx["cv_ref_path"]:
            voice_dict["ref_audio_path"] = str(ctx["cv_ref_path"])
        async with _get_path_lock(preview_path):
            await asyncio.to_thread(_preview_synthesizer, voice_dict, preview_path, ctx["sample_text"])
            if not preview_path.is_file() or preview_path.stat().st_size == 0:
                generate_fallback_preview_wav(preview_path)
            res = await _finish_preview(
                preview_path,
                preview_id,
                auto_speed=ctx["auto_speed"],
                slot_duration_s=ctx["slot_duration_s"],
                max_speed_rate=ctx["max_speed_rate"],
            )
        cleanup_previews_dir(exclude_names={preview_path.name})
        return res

    async with _get_path_lock(preview_path):
        if not force_refresh and _is_valid_preview_wav(preview_path):
            speed_from_cache = _read_cached_speed(preview_path)
            if speed_from_cache is not None:
                touch_cache_entry(preview_path, (".json",))
                return preview_id, preview_path, speed_from_cache

        _unlink_with_retry(preview_path)
        _unlink_with_retry(preview_path.with_suffix(".json"))

        call_abort = threading.Event()
        try:
            await asyncio.wait_for(
                _synthesize_ctx_uncached(ctx, use_cuda=use_cuda, abort_event=call_abort),
                timeout=PREVIEW_TIMEOUT_SECONDS,
            )
        except BaseException as exc:
            call_abort.set()
            _unlink_with_retry(preview_path)
            _unlink_with_retry(preview_path.with_suffix(".json"))
            if isinstance(exc, Exception):
                logger.exception("TTS preview synthesis failed")
            raise

        if not _is_valid_preview_wav(preview_path):
            _unlink_with_retry(preview_path)
            _unlink_with_retry(preview_path.with_suffix(".json"))
            raise RuntimeError("TTS preview did not produce audible audio")

        res = await _finish_preview(
            preview_path,
            preview_id,
            auto_speed=ctx["auto_speed"],
            slot_duration_s=ctx["slot_duration_s"],
            max_speed_rate=ctx["max_speed_rate"],
        )
        touch_cache_entry(preview_path, (".json",))

    cleanup_previews_dir(exclude_names={preview_path.name})
    return res


async def synthesize_tts_batch(
    items: list[dict[str, Any]],
    *,
    use_cuda: bool = True,
    db_path: Optional[str | Path] = None,
    cancel_event: Optional[threading.Event] = None,
    is_disconnected: Optional[Callable[[], Any]] = None,
) -> list[dict[str, Any]]:
    """Synthesize a batch of TTS segments efficiently.

    - Reuses disk cache (touching mtime).
    - Groups uncached VieNeu items by resolved voice/style/tuning and invokes
      ``_run_vieneu_batch_synthesis`` in sub-batches of ``<= 4`` (with per-item
      fallback if a multi-item sub-batch fails).
    - Runs cloud providers (ElevenLabs/Gemini) concurrently under bounded semaphores.
    - Runs ``smart_fit_audio_file`` outside the GPU lock.
    - Surfaces per-item errors without masking surviving segments.
    """
    from videotrans.services.audio_fit import get_audio_duration_seconds

    voice_store.init_voice_dirs()
    results: list[Optional[dict[str, Any]]] = [None] * len(items)
    if not items:
        return []

    async def _is_cancelled() -> bool:
        if cancel_event is not None and cancel_event.is_set():
            return True
        if is_disconnected is not None:
            try:
                if await is_disconnected():
                    if cancel_event is not None:
                        cancel_event.set()
                    return True
            except Exception:
                pass
        return False

    def _ok_result(seg_id: Any, ctx: dict[str, Any], applied_speed: float) -> dict[str, Any]:
        p_id = ctx["preview_id"]
        p_path: Path = ctx["preview_path"]
        url = f"/api/tts/preview/{p_id}/audio"
        dur = get_audio_duration_seconds(p_path)
        return {
            "segmentId": seg_id,
            "ok": True,
            "id": p_id,
            "preview_id": p_id,
            "previewUrl": url,
            "preview_url": url,
            "audio_url": url,
            "durationSec": round(dur, 4),
            "appliedSpeed": applied_speed,
            "applied_speed": applied_speed,
            "voice": ctx["raw_voice"] or ctx["voice_str"],
            "provider": ctx["provider"],
            "error": None,
        }

    def _err_result(seg_id: Any, ctx: Optional[dict[str, Any]], msg: str) -> dict[str, Any]:
        return {
            "segmentId": seg_id,
            "ok": False,
            "id": None,
            "preview_id": None,
            "previewUrl": None,
            "preview_url": None,
            "audio_url": None,
            "durationSec": 0.0,
            "appliedSpeed": 1.0,
            "applied_speed": 1.0,
            "voice": (ctx["raw_voice"] or ctx["voice_str"]) if ctx else None,
            "provider": ctx["provider"] if ctx else None,
            "error": msg,
        }

    pending_entries: list[tuple[int, Any, dict[str, Any], bool]] = []
    seen_paths: set[str] = set()
    in_flight_by_path: dict[Path, list[tuple[int, Any, dict[str, Any]]]] = {}
    meta_by_idx: dict[int, tuple[Any, Optional[dict[str, Any]]]] = {}

    for idx, raw_item in enumerate(items):
        seg_id = raw_item.get("segment_id") if raw_item.get("segment_id") is not None else raw_item.get("segmentId")
        if seg_id is None:
            seg_id = raw_item.get("id", idx)
        meta_by_idx[idx] = (seg_id, None)
        raw_text = str(raw_item.get("text") or "").strip()
        if not raw_text:
            results[idx] = _err_result(seg_id, None, "Segment text is empty")
            continue

        ctx = _resolve_preview_context(
            provider=raw_item["provider"],
            voice=raw_item.get("voice") or "",
            text=raw_text,
            language=raw_item.get("language"),
            speed=raw_item.get("speed"),
            rate=raw_item.get("rate"),
            pitch=raw_item.get("pitch"),
            style=raw_item.get("style"),
            tuning_params=raw_item.get("tuning_params"),
            db_path=db_path,
            auto_speed=bool(raw_item.get("auto_speed", False)),
            slot_duration_s=raw_item.get("slot_duration_s"),
            max_speed_rate=raw_item.get("max_speed_rate", 1.35),
        )
        meta_by_idx[idx] = (seg_id, ctx)
        force_refresh = bool(raw_item.get("force_refresh", False))
        p_path = ctx["preview_path"]
        seen_paths.add(p_path.name)

        if p_path in in_flight_by_path:
            in_flight_by_path[p_path].append((idx, seg_id, ctx))
            continue

        if _preview_synthesizer is None and not force_refresh:
            async with _get_path_lock(p_path):
                if _is_valid_preview_wav(p_path):
                    cached_spd = _read_cached_speed(p_path)
                    if cached_spd is not None:
                        touch_cache_entry(p_path, (".json",))
                        results[idx] = _ok_result(seg_id, ctx, cached_spd)
                        continue

        in_flight_by_path[p_path] = []
        pending_entries.append((idx, seg_id, ctx, force_refresh))

    def _propagate_duplicate_paths() -> None:
        for idx, _, ctx, _ in pending_entries:
            p_path = ctx["preview_path"]
            dups = in_flight_by_path.get(p_path)
            if not dups:
                continue
            leader_res = results[idx]
            for dup_idx, dup_seg_id, dup_ctx in dups:
                if leader_res is not None and leader_res.get("ok"):
                    results[dup_idx] = {**leader_res, "segmentId": dup_seg_id}
                else:
                    results[dup_idx] = _err_result(
                        dup_seg_id,
                        dup_ctx,
                        (leader_res or {}).get("error") or "Batch synthesis failed",
                    )

    def _finalize_results() -> list[dict[str, Any]]:
        final: list[dict[str, Any]] = []
        for idx, r in enumerate(results):
            if r is not None:
                final.append(r)
            else:
                seg_id, ctx = meta_by_idx.get(idx, (idx, None))
                err_msg = "Cancelled" if (cancel_event is not None and cancel_event.is_set()) else "Unknown batch error"
                final.append(_err_result(seg_id, ctx, err_msg))
        return final

    if not pending_entries:
        cleanup_previews_dir(exclude_names=seen_paths)
        return _finalize_results()

    # Test-double fast path (preserves set_preview_synthesizer contracts)
    if _preview_synthesizer is not None:
        sem = asyncio.Semaphore(4)

        async def _run_fake_entry(idx: int, seg_id: Any, ctx: dict[str, Any]) -> None:
            if await _is_cancelled():
                results[idx] = _err_result(seg_id, ctx, "Cancelled")
                return
            async with sem:
                if await _is_cancelled():
                    results[idx] = _err_result(seg_id, ctx, "Cancelled")
                    return
                p_path = ctx["preview_path"]
                voice_dict = {
                    "id": ctx["preview_id"],
                    "name": (ctx["cv"].get("name") if ctx["cv"] else None) or ctx["voice_str"] or "Default",
                    "provider": ctx["provider"],
                    "language": ctx["language"] or "vi",
                    "style": ctx["style"],
                }
                if ctx["cv_ref_path"]:
                    voice_dict["ref_audio_path"] = str(ctx["cv_ref_path"])
                try:
                    async with _get_path_lock(p_path):
                        await asyncio.to_thread(_preview_synthesizer, voice_dict, p_path, ctx["sample_text"])
                        if not p_path.is_file() or p_path.stat().st_size == 0:
                            generate_fallback_preview_wav(p_path)
                        _, _, spd = await _finish_preview(
                            p_path,
                            ctx["preview_id"],
                            auto_speed=ctx["auto_speed"],
                            slot_duration_s=ctx["slot_duration_s"],
                            max_speed_rate=ctx["max_speed_rate"],
                        )
                    results[idx] = _ok_result(seg_id, ctx, spd)
                except Exception as exc:
                    results[idx] = _err_result(seg_id, ctx, str(exc))

        await asyncio.gather(*(_run_fake_entry(i, sid, c) for i, sid, c, _ in pending_entries))
        _propagate_duplicate_paths()
        cleanup_previews_dir(exclude_names=seen_paths)
        return _finalize_results()

    # Partition pending items into VieNeu batchable vs OmniVoice vs Cloud
    vieneu_groups: dict[tuple[Any, ...], list[tuple[int, Any, dict[str, Any], bool, dict[str, Any]]]] = {}
    other_entries: list[tuple[int, Any, dict[str, Any], bool]] = []

    from videotrans.util.help_role import get_vieneu_custom_voice_path

    for idx, seg_id, ctx, force_refresh in pending_entries:
        if ctx["provider"] == tts.VIENEU_TTS:
            tp = ctx["tp"]
            voice_str = ctx["voice_str"]
            cv = ctx["cv"]
            cv_resolved = ctx["cv_resolved"]
            cv_ref_path = ctx["cv_ref_path"]
            cv_cached_emb = ctx["cv_cached_emb"]
            custom_ref = cv_ref_path or get_vieneu_custom_voice_path(voice_str)
            selected_style = str(ctx["style"] or tp.get("style", "tu_nhien"))
            temp_val = float(tp.get("temperature", 0.8))
            rep_pen = float(tp.get("repetition_penalty", 1.2))
            top_p_val = float(tp.get("top_p", 0.95))
            denoise_opt = bool(tp.get("denoise", True))

            if cv_resolved and cv_resolved.get("base_type") == "preset":
                preset_name = cv_resolved.get("preset_voice") or (cv.get("external_voice_id") if cv else None)
                eff_key = ("preset_cv", str(preset_name or ""))
                batch_kwargs = {"ref_audio": None, "voice": preset_name, "cached_embedding": None}
            elif (custom_ref and Path(custom_ref).is_file()) or cv_cached_emb:
                ref_p = Path(custom_ref) if custom_ref else None
                eff_key = ("ref_or_emb", ref_p.as_posix() if ref_p else "", bool(cv_cached_emb), voice_str)
                batch_kwargs = {"ref_audio": ref_p, "voice": None, "cached_embedding": cv_cached_emb}
            else:
                preset_v = voice_str if (voice_str and voice_str.lower() not in ("no", "default", "clone")) else None
                eff_key = ("preset", str(preset_v or ""))
                batch_kwargs = {"ref_audio": None, "voice": preset_v, "cached_embedding": None}

            batch_kwargs.update({
                "style": selected_style,
                "temperature": temp_val,
                "repetition_penalty": rep_pen,
                "top_p": top_p_val,
                "denoise": denoise_opt,
                "use_cuda": use_cuda,
            })
            group_key = (eff_key, selected_style, temp_val, rep_pen, top_p_val, denoise_opt, bool(use_cuda))
            vieneu_groups.setdefault(group_key, []).append((idx, seg_id, ctx, force_refresh, batch_kwargs))
        else:
            other_entries.append((idx, seg_id, ctx, force_refresh))

    async def _finalize_item(
        idx: int,
        seg_id: Any,
        ctx: dict[str, Any],
        source_path: Optional[Path] = None,
    ) -> None:
        p_path: Path = ctx["preview_path"]
        try:
            async with _get_path_lock(p_path):
                if source_path is not None:
                    if not _is_valid_preview_wav(source_path):
                        _unlink_with_retry(source_path)
                        results[idx] = _err_result(seg_id, ctx, "TTS preview did not produce audible audio")
                        return
                    p_path.parent.mkdir(parents=True, exist_ok=True)
                    _unlink_with_retry(p_path.with_suffix(".json"))
                    await asyncio.to_thread(_replace_file_with_retry, source_path, p_path)
                if not _is_valid_preview_wav(p_path):
                    _unlink_with_retry(p_path)
                    _unlink_with_retry(p_path.with_suffix(".json"))
                    results[idx] = _err_result(seg_id, ctx, "TTS preview did not produce audible audio")
                    return
                _, _, spd = await _finish_preview(
                    p_path,
                    ctx["preview_id"],
                    auto_speed=ctx["auto_speed"],
                    slot_duration_s=ctx["slot_duration_s"],
                    max_speed_rate=ctx["max_speed_rate"],
                )
                touch_cache_entry(p_path, (".json",))
                results[idx] = _ok_result(seg_id, ctx, spd)
        except Exception as exc:
            results[idx] = _err_result(seg_id, ctx, str(exc))
        finally:
            if source_path is not None:
                _unlink_with_retry(source_path)

    async def _process_vieneu_groups() -> None:
        cancelled = False
        for group_items in vieneu_groups.values():
            if cancelled or await _is_cancelled():
                cancelled = True
                for idx, seg_id, ctx, _, _ in group_items:
                    if results[idx] is None:
                        results[idx] = _err_result(seg_id, ctx, "Cancelled")
                continue
            b_kwargs = group_items[0][4]
            for chunk_start in range(0, len(group_items), VIENEU_BATCH_SIZE):
                if cancelled or await _is_cancelled():
                    cancelled = True
                    for idx, seg_id, ctx, _, _ in group_items[chunk_start:]:
                        if results[idx] is None:
                            results[idx] = _err_result(seg_id, ctx, "Cancelled")
                    break
                chunk = group_items[chunk_start : chunk_start + VIENEU_BATCH_SIZE]
                # Synthesize into per-request/per-item unique .tmp files inside GPU lock,
                # then atomically move & stretch under _get_path_lock(preview_path) outside GPU lock.
                tmp_paths: list[Path] = []
                batch_payload: list[dict[str, Any]] = []
                for _, _, ctx, _, _ in chunk:
                    p_path: Path = ctx["preview_path"]
                    p_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_wav = p_path.with_name(f".{p_path.stem}.{uuid.uuid4().hex[:8]}.tmp")
                    tmp_paths.append(tmp_wav)
                    batch_payload.append({"text": ctx["sample_text"], "preview_path": tmp_wav})

                batch_failed_exc: Optional[Exception] = None
                outcomes: Optional[list[tuple[bool, Optional[str]]]] = None
                call_abort = threading.Event()
                try:
                    async with _get_gpu_lock():
                        if await _is_cancelled():
                            cancelled = True
                            call_abort.set()
                            for tp_path in tmp_paths:
                                _unlink_with_retry(tp_path)
                            for idx, seg_id, ctx, _, _ in chunk:
                                if results[idx] is None:
                                    results[idx] = _err_result(seg_id, ctx, "Cancelled")
                            continue
                        outcomes = await asyncio.wait_for(
                            asyncio.to_thread(
                                _run_vieneu_batch_synthesis,
                                batch_payload,
                                abort_event=call_abort,
                                cancel_event=cancel_event,
                                **b_kwargs,
                            ),
                            timeout=PREVIEW_TIMEOUT_SECONDS,
                        )
                except Exception as exc:
                    call_abort.set()
                    batch_failed_exc = exc
                except BaseException:
                    call_abort.set()
                    for tp_path in tmp_paths:
                        _unlink_with_retry(tp_path)
                    raise

                if batch_failed_exc is not None or outcomes is None:
                    for tp_path in tmp_paths:
                        _unlink_with_retry(tp_path)
                    # Fallback per-item if multi-item batch failed so surviving segments succeed
                    if len(chunk) == 1:
                        idx, seg_id, ctx, _, _ = chunk[0]
                        if await _is_cancelled():
                            cancelled = True
                            results[idx] = _err_result(seg_id, ctx, "Cancelled")
                        else:
                            results[idx] = _err_result(seg_id, ctx, str(batch_failed_exc or "VieNeu batch synthesis failed"))
                        continue

                    for idx, seg_id, ctx, _, _ in chunk:
                        if cancelled or await _is_cancelled():
                            cancelled = True
                            results[idx] = _err_result(seg_id, ctx, "Cancelled")
                            continue
                        p_path = ctx["preview_path"]
                        try:
                            async with _get_path_lock(p_path):
                                _unlink_with_retry(p_path)
                                _unlink_with_retry(p_path.with_suffix(".json"))
                                item_abort = threading.Event()
                                try:
                                    await asyncio.wait_for(
                                        _synthesize_ctx_uncached(ctx, use_cuda=use_cuda, abort_event=item_abort),
                                        timeout=PREVIEW_TIMEOUT_SECONDS,
                                    )
                                except BaseException:
                                    item_abort.set()
                                    _unlink_with_retry(p_path)
                                    _unlink_with_retry(p_path.with_suffix(".json"))
                                    raise
                                if not _is_valid_preview_wav(p_path):
                                    _unlink_with_retry(p_path)
                                    _unlink_with_retry(p_path.with_suffix(".json"))
                                    results[idx] = _err_result(seg_id, ctx, "TTS preview did not produce audible audio")
                                    continue
                                _, _, spd = await _finish_preview(
                                    p_path,
                                    ctx["preview_id"],
                                    auto_speed=ctx["auto_speed"],
                                    slot_duration_s=ctx["slot_duration_s"],
                                    max_speed_rate=ctx["max_speed_rate"],
                                )
                                touch_cache_entry(p_path, (".json",))
                                results[idx] = _ok_result(seg_id, ctx, spd)
                        except Exception as item_exc:
                            results[idx] = _err_result(seg_id, ctx, str(item_exc))
                    continue

                # Run smart-fit / sidecar write outside GPU lock concurrently across the chunk
                fit_tasks = []
                for (idx, seg_id, ctx, _, _), (ok, err_msg), tmp_wav in zip(chunk, outcomes, tmp_paths):
                    if not ok:
                        _unlink_with_retry(tmp_wav)
                        results[idx] = _err_result(seg_id, ctx, err_msg or "VieNeu synthesis failed")
                    else:
                        fit_tasks.append(_finalize_item(idx, seg_id, ctx, source_path=tmp_wav))
                if fit_tasks:
                    try:
                        await asyncio.gather(*fit_tasks)
                    finally:
                        for tp_path in tmp_paths:
                            _unlink_with_retry(tp_path)

    async def _process_other_entry(idx: int, seg_id: Any, ctx: dict[str, Any], force_refresh: bool) -> None:
        if await _is_cancelled():
            results[idx] = _err_result(seg_id, ctx, "Cancelled")
            return
        p_path: Path = ctx["preview_path"]
        try:
            async with _get_path_lock(p_path):
                item_abort = threading.Event()
                try:
                    if not force_refresh and _is_valid_preview_wav(p_path):
                        spd = _read_cached_speed(p_path)
                        if spd is not None:
                            touch_cache_entry(p_path, (".json",))
                            results[idx] = _ok_result(seg_id, ctx, spd)
                            return
                    _unlink_with_retry(p_path)
                    _unlink_with_retry(p_path.with_suffix(".json"))
                    await asyncio.wait_for(
                        _synthesize_ctx_uncached(ctx, use_cuda=use_cuda, abort_event=item_abort),
                        timeout=PREVIEW_TIMEOUT_SECONDS,
                    )
                    if not _is_valid_preview_wav(p_path):
                        _unlink_with_retry(p_path)
                        _unlink_with_retry(p_path.with_suffix(".json"))
                        results[idx] = _err_result(seg_id, ctx, "TTS preview did not produce audible audio")
                        return
                    _, _, spd = await _finish_preview(
                        p_path,
                        ctx["preview_id"],
                        auto_speed=ctx["auto_speed"],
                        slot_duration_s=ctx["slot_duration_s"],
                        max_speed_rate=ctx["max_speed_rate"],
                    )
                    touch_cache_entry(p_path, (".json",))
                    results[idx] = _ok_result(seg_id, ctx, spd)
                except BaseException:
                    item_abort.set()
                    _unlink_with_retry(p_path)
                    _unlink_with_retry(p_path.with_suffix(".json"))
                    raise
        except Exception as exc:
            results[idx] = _err_result(seg_id, ctx, str(exc))

    tasks = []
    if vieneu_groups:
        tasks.append(_process_vieneu_groups())
    for idx, seg_id, ctx, force_refresh in other_entries:
        tasks.append(_process_other_entry(idx, seg_id, ctx, force_refresh))

    if tasks:
        await asyncio.gather(*tasks)

    # Propagate results for any duplicate preview_paths in the same batch
    _propagate_duplicate_paths()

    cleanup_previews_dir(exclude_names=seen_paths)
    return _finalize_results()
