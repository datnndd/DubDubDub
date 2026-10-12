# -*- coding: utf-8 -*-
"""Voice preview synthesis engine with concurrency lock, 30s timeout, and CPU fallback."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Optional
import uuid
import wave

from videotrans import tts
from videotrans.configure.config import params
from videotrans.core import voice_store
from videotrans.util.help_ffmpeg import runffmpeg

logger = logging.getLogger("videotrans.voice_preview")

_preview_lock = asyncio.Lock()
PREVIEW_TIMEOUT_SECONDS = 120.0
CLONE_TIMEOUT_SECONDS = 120.0

_preview_synthesizer = None


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
    cmd = [
        "-y",
        "-i", str(ref_audio_path),
        "-af", "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-40dB,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-t", "3",
        "-vn",
        "-ac", "1",
        "-ar", str(target_rate),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    cmd_simple = [
        "-y",
        "-i", str(ref_audio_path),
        "-t", "3",
        "-vn",
        "-ac", "1",
        "-ar", str(target_rate),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    try:
        runffmpeg(cmd, force_cpu=True)
    except Exception:
        pass

    if not _is_valid_preview_wav(output_path):
        try:
            runffmpeg(cmd_simple, force_cpu=True)
        except Exception:
            pass

    return _is_valid_preview_wav(output_path)


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
) -> None:
    """Run VieNeu TTS inference synchronously with style, emotion tags, and tuning parameters using cached engine."""
    if output_file is None:
        raise ValueError("output_file is required")

    import numpy as np
    import soundfile as sf
    from videotrans.services.model_cache import get_cached_vieneu_engine, evict_vieneu_engine

    def _infer(device: str, backend: str):
        engine = get_cached_vieneu_engine(device=device, backend=backend, max_batch_size=1)
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

        if not audios or len(audios) == 0:
            raise RuntimeError("VieNeu returned empty audio list")
        aud_item = audios[0]
        if hasattr(aud_item, "detach"):
            aud_item = aud_item.detach().cpu().numpy()
        audio_data = np.asarray(aud_item, dtype=np.float32).squeeze()
        if audio_data.size == 0:
            raise RuntimeError("VieNeu returned 0 audio samples")
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
        peak = float(np.max(np.abs(audio_data))) if audio_data.size > 0 else 0.0
        if peak < 1e-4:
            raise RuntimeError(f"VieNeu returned silent audio (peak amplitude: {peak:.6f})")

        # Peak normalize to ~0.90 (-1 dBFS) for clear, audible voice
        audio_data = (audio_data / peak) * 0.90
        sf.write(str(output_file), audio_data, engine.sample_rate)

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
        logger.error("VieNeu preview synthesis failed: %s", exc, exc_info=True)
        raise


def _run_omnivoice_synthesis(ref_audio: Path, ref_text: str, output_file: Path, text: str, use_cuda: bool = True, speed: float = 1.0) -> None:
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
        model = get_cached_omnivoice_model(model_dir=model_dir, device=device, dtype=dtype)
        wav = model.generate(
            text=text,
            ref_audio=ref_audio.as_posix(),
            ref_text=ref_text or None,
            speed=speed or 1.0,
        )
        if not wav or len(wav) == 0:
            raise RuntimeError("OmniVoice returned empty audio list")
        wav_item = wav[0]
        if hasattr(wav_item, "detach"):
            wav_item = wav_item.detach().cpu().numpy()
        audio_data = np.asarray(wav_item, dtype=np.float32).squeeze()
        if audio_data.size == 0:
            raise RuntimeError("OmniVoice returned 0 audio samples")
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
        peak = float(np.max(np.abs(audio_data))) if audio_data.size > 0 else 0.0
        if peak < 1e-4:
            raise RuntimeError(f"OmniVoice returned silent audio (peak amplitude: {peak:.6f})")

        # Peak normalize to ~0.90 (-1 dBFS) for clear, audible voice
        audio_data = (audio_data / peak) * 0.90
        sf.write(str(output_file), audio_data, 24000)

    is_gpu = use_cuda and torch.cuda.is_available()
    if is_gpu:
        try:
            _infer("cuda:0", torch.float16)
            return
        except Exception as exc:
            logger.warning("CUDA error during OmniVoice preview synthesis (%s), falling back to CPU", exc)
            evict_omnivoice_model(model_dir=model_dir, device="cuda")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    _infer(gpus.mps_or_cpu(), torch.float32)


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

    # 1. Return cached preview if valid and not force_refresh
    if not force_refresh and _is_valid_preview_wav(preview_path):
        if preview_path.stat().st_mtime >= (voice.get("updated_at", 0) - 2.0):
            return preview_path

    # 2. Determine sample utterance
    lang = language or voice.get("language") or "vi"
    sample_text = (text or "").strip() or get_default_preview_text(lang)

    # 3. Resolve voice hierarchy (handling designed voices)
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

    async with _preview_lock:
        # Re-check cache inside lock to avoid redundant concurrent synthesis
        if not force_refresh and _is_valid_preview_wav(preview_path):
            if preview_path.stat().st_mtime >= (voice.get("updated_at", 0) - 2.0):
                return preview_path

        provider = voice.get("provider", tts.VIENEU_TTS)

        async def _do_synthesis() -> None:
            if _preview_synthesizer is not None:
                await asyncio.to_thread(_preview_synthesizer, voice, preview_path, sample_text)
                return

            # Check if neural engine can be loaded
            if provider == tts.VIENEU_TTS:
                try:
                    if resolved.get("base_type") == "preset":
                        preset_name = resolved.get("preset_voice") or voice.get("external_voice_id")
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
                        )
                        return
                    elif ref_audio_path or cached_emb:
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
                        )
                        return
                except ImportError:
                    logger.info("vieneu package not installed, using fallback")
                except Exception as exc:
                    logger.warning("VieNeu synthesis error, falling back: %s", exc)

            if provider == tts.OMNIVOICE_TTS and ref_audio_path:
                try:
                    ref_text = voice.get("ref_text", "")
                    await asyncio.to_thread(
                        _run_omnivoice_synthesis,
                        ref_audio_path,
                        ref_text,
                        preview_path,
                        sample_text,
                        use_cuda,
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

        return preview_path


def generate_fallback_preview_wav(path: Path, duration_sec: float = 1.5, sample_rate: int = 24000) -> Path:
    """Generate a valid PCM 16-bit mono WAV preview file as safe fallback."""
    import array
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = [
            int(8000 * max(0.0, 1.0 - (i / num_samples)) * math.sin(2 * math.pi * 440.0 * (i / sample_rate)))
            for i in range(num_samples)
        ]
        wf.writeframes(array.array("h", samples).tobytes())
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

    async with _preview_lock:
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
                    await asyncio.to_thread(
                        _run_vieneu_synthesis,
                        ref_src,
                        preview_path,
                        sample_text,
                        use_cuda,
                        denoise=denoise,
                    )
                    return
                except ImportError:
                    logger.info("vieneu package not installed, falling back to snippet preview")
                except Exception as exc:
                    logger.warning("VieNeu clone preview synthesis failed: %s", exc)

            elif provider == tts.OMNIVOICE_TTS and ref_src.is_file():
                try:
                    await asyncio.to_thread(
                        _run_omnivoice_synthesis,
                        ref_src,
                        ref_text or "",
                        preview_path,
                        sample_text,
                        use_cuda,
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

        return pid, preview_path


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
    sample_text = (text or "").strip() or get_default_preview_text(language or "vi")
    voice_str = str(voice or "").strip()
    tp = dict(tuning_params or {})

    async def finish_preview(path: Path, preview_id: str) -> tuple[str, Path, float]:
        applied_speed = 1.0
        if auto_speed and slot_duration_s and slot_duration_s > 0:
            from videotrans.services.audio_fit import smart_fit_audio_file
            _, applied_speed, fit_status = await asyncio.to_thread(
                smart_fit_audio_file, path, slot_duration_s, max_speed_rate or 1.35,
            )
            if fit_status in {"error", "not_found"}:
                raise RuntimeError("Could not adjust preview audio to fit the subtitle timing")
        path.with_suffix(".json").write_text(json.dumps({"applied_speed": applied_speed}), encoding="utf-8")
        return preview_id, path, applied_speed

    # 1. Check if voice refers to an existing custom voice
    cv = voice_store.get_voice(voice_str, db_path=db_path) if voice_str else None
    if not cv and voice_str:
        cv = voice_store.find_voice_by_name(voice_str, provider=provider, db_path=db_path)

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

    # 2. If test double synthesizer is set, invoke it
    if _preview_synthesizer is not None:
        key_raw = f"{provider}_{voice_str}_{sample_text}_{language}_{speed}_{style}_{auto_speed}_{slot_duration_s}_{max_speed_rate}"
        p_id = f"prev_{hashlib.md5(key_raw.encode('utf-8')).hexdigest()[:12]}"
        p_path = voice_store.get_preview_audio_path(f"{p_id}.wav")
        voice_dict = {
            "id": p_id,
            "name": (cv.get("name") if cv else None) or voice_str or "Default",
            "provider": provider,
            "language": language or "vi",
            "style": style,
        }
        if cv_ref_path:
            voice_dict["ref_audio_path"] = str(cv_ref_path)
        await asyncio.to_thread(_preview_synthesizer, voice_dict, p_path, sample_text)
        if not p_path.is_file() or p_path.stat().st_size == 0:
            generate_fallback_preview_wav(p_path)
        return await finish_preview(p_path, p_id)

    # 3. Standard/Preset or Custom voice synthesis with deterministic cache
    tuning_key = json.dumps(tp, sort_keys=True)
    cache_seed = f"{provider}_{voice_str}_{sample_text}_{language}_{speed}_{rate}_{pitch}_{style}_{tuning_key}_{auto_speed}_{slot_duration_s}_{max_speed_rate}"
    preview_id = f"prev_{hashlib.md5(cache_seed.encode('utf-8')).hexdigest()[:12]}"
    preview_filename = f"{preview_id}.wav"
    preview_path = voice_store.get_preview_audio_path(preview_filename)

    # Return cached if valid and not force_refresh
    def cached_speed() -> float | None:
        try:
            return float(json.loads(preview_path.with_suffix(".json").read_text(encoding="utf-8"))["applied_speed"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    if not force_refresh and _is_valid_preview_wav(preview_path):
        speed_from_cache = cached_speed()
        if speed_from_cache is not None:
            return preview_id, preview_path, speed_from_cache

    async with _preview_lock:
        if not force_refresh and _is_valid_preview_wav(preview_path):
            speed_from_cache = cached_speed()
            if speed_from_cache is not None:
                return preview_id, preview_path, speed_from_cache

        preview_path.unlink(missing_ok=True)
        preview_path.with_suffix(".json").unlink(missing_ok=True)

        async def _do_synthesis() -> None:
            # VieNeu-TTS
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
                    )
                    return
                elif (custom_ref and Path(custom_ref).is_file()) or cv_cached_emb:
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
                    )
                    return

                # Preset voice: leverage cached VieNeu engine with CUDA-to-CPU fallback
                preset_v = voice_str if (voice_str and voice_str.lower() not in ("no", "default", "clone")) else None
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
                )
                return

            # OmniVoice
            if provider == tts.OMNIVOICE_TTS:
                from videotrans.configure.config import ROOT_DIR
                model_dir = Path(ROOT_DIR) / "models" / "models--k2-fsa--OmniVoice"
                if not (model_dir / "model.safetensors").is_file():
                    raise FileNotFoundError(f"OmniVoice model not installed at {model_dir}")
                omnivoice_ref = cv_ref_path
                if omnivoice_ref and Path(omnivoice_ref).is_file():
                    ref_text = (cv.get("ref_text") if cv else "") or ""
                    await asyncio.to_thread(
                        _run_omnivoice_synthesis,
                        Path(omnivoice_ref),
                        ref_text,
                        preview_path,
                        sample_text,
                        use_cuda,
                        speed=speed or 1.0,
                    )
                    return
                # OmniVoice requires reference audio, generate if available
                raise NotImplementedError("OmniVoice preset preview requires clone audio")

            # ElevenLabs
            if provider == tts.ELEVENLABS_TTS:
                key = params.get("elevenlabstts_key", "")
                if not key:
                    raise ValueError("ElevenLabs API key not configured")
                from elevenlabs import ElevenLabs, VoiceSettings
                client = ElevenLabs(api_key=key)
                target_voice = voice_str if (voice_str and voice_str.lower() not in ("no", "default")) else "21m00Tcm4TlvDq8ikWAM"
                voice_kwargs: dict[str, Any] = {
                    "text": sample_text,
                    "voice_id": target_voice,
                    "model_id": params.get("elevenlabstts_models", "eleven_multilingual_v2"),
                }
                if speed is not None:
                    voice_kwargs["voice_settings"] = VoiceSettings(speed=speed)
                resp = client.text_to_speech.convert(**voice_kwargs)
                with open(preview_path, "wb") as f:
                    for chunk in resp:
                        f.write(chunk)
                return

            # Gemini TTS
            if provider == tts.GEMINI_TTS:
                key = params.get("gemini_key", "")
                if not key:
                    raise ValueError("Gemini API key not configured")
                from videotrans.tts._geminitts import GEMINITTS
                tts_inst = GEMINITTS(language=language or "vi")
                target_voice = voice_str if (voice_str and voice_str.lower() not in ("no", "default")) else "Puck"
                tts_inst.generate_tts_segment(
                    sample_text,
                    target_voice,
                    params.get("gemini_ttsmodel", "gemini-2.5-flash-preview-tts"),
                    str(preview_path),
                )
                return

            raise RuntimeError(f"Unsupported or unconfigured provider {provider}")

        try:
            await asyncio.wait_for(_do_synthesis(), timeout=PREVIEW_TIMEOUT_SECONDS)
        except Exception:
            preview_path.unlink(missing_ok=True)
            logger.exception("TTS preview synthesis failed")
            raise

        if not _is_valid_preview_wav(preview_path):
            preview_path.unlink(missing_ok=True)
            raise RuntimeError("TTS preview did not produce audible audio")

        return await finish_preview(preview_path, preview_id)
