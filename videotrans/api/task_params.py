# -*- coding: utf-8 -*-
"""Helpers for translating WebUI frontend options into task configuration dictionaries."""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from videotrans import recognition, translator, tts
from videotrans.configure import config as runtime_config
from videotrans.configure.config import TEMP_DIR
from videotrans.util._ffmpeg_misc import format_video
from videotrans.util.help_role import role_menu
from videotrans.api.catalog import (
    ASR_BY_TYPE,
    ASR_PROVIDERS,
    asr_models,
    OUTPUT_DIR,
    TIMING_MODES,
    TRANSLATION_BY_TYPE,
    TRANSLATION_MODES,
)


def _required_index(value: Any, size: int, label: str) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"A valid {label} is required") from None
    if not 0 <= index < size:
        raise ValueError(f"Unknown {label}: {value}")
    return index


def _optional_index(value: Any, size: int, default: int = 0) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return default
    return index if 0 <= index < size else default


def _safe_volume(value: Any, default: float) -> float:
    """Safely coerce volume float value clamped to [0.0, 1.5], with fallback default."""
    if value is None or isinstance(value, bool):
        return default
    try:
        val = float(value)
        if math.isnan(val):
            return default
        return max(0.0, min(1.5, val))
    except (TypeError, ValueError, OverflowError):
        return default


def _translation_mode(value: Any = None) -> tuple[str, bool]:
    mode_id = str(value or ("srt" if runtime_config.settings.get("aisendsrt", True) else "line"))
    mode = TRANSLATION_MODES.get(mode_id)
    if mode is None:
        raise ValueError(f"Unknown translation mode: {mode_id}")
    return mode_id, bool(mode["aisendsrt"])


def _resolve_voice_id_to_role(voice_id_or_role: str, tts_type: int) -> str:
    s = str(voice_id_or_role).strip()
    if not s or s in {"No", "clone"}:
        return s
    if s.startswith("voice_"):
        try:
            from videotrans.core import voice_store
            from videotrans import tts
            v = voice_store.get_voice(s)
            if v:
                if tts_type == tts.VIENEU_TTS:
                    return f"Custom: {v['name']}"
                elif tts_type == tts.OMNIVOICE_TTS:
                    return v["name"]
                elif tts_type == tts.ELEVENLABS_TTS:
                    return v.get("external_voice_id") or v["name"]
                return v["name"]
        except Exception:
            pass
    return s


def build_task_params(
    input_path: Path,
    options: dict[str, Any],
    job_type: str = "full",
    project_id: str | None = None,
    *,
    format_video_fn=format_video,
    output_dir: Path | str = OUTPUT_DIR,
    temp_dir: Path | str = TEMP_DIR,
    role_provider=role_menu,
) -> dict[str, Any]:
    """Translate supported frontend fields into the existing task configuration."""
    file_info = format_video_fn(input_path.resolve().as_posix())
    safe_stem = re.sub(r"[^\w.-]+", "-", file_info.basename, flags=re.UNICODE).strip("-")
    pid = project_id or options.get("projectId") or options.get("project_id")
    if pid:
        from videotrans.core.project_store import get_project_dir
        target_dir = get_project_dir(str(pid)) / "exports"
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = Path(output_dir) / (safe_stem or file_info.uuid)
    cache_dir = Path(temp_dir) / file_info.uuid

    if job_type in {"render", "translation"}:
        recogn_type = _optional_index(options.get("recognType"), len(recognition.RECOGN_NAME_LIST), 0)
        asr_provider = ASR_BY_TYPE.get(recogn_type) or ASR_PROVIDERS[0]
        model_name = str(options.get("modelName") or asr_provider["models"][0])
    else:
        recogn_type = _required_index(options.get("recognType"), len(recognition.RECOGN_NAME_LIST), "ASR engine")
        asr_provider = ASR_BY_TYPE.get(recogn_type)
        if asr_provider is None:
            raise ValueError(f"Unsupported ASR engine: {recogn_type}")
        model_name = str(options.get("modelName") or asr_provider["models"][0])
        if recogn_type == recognition.Deepgram:
            if not model_name.strip():
                model_name = "nova-3"
        elif asr_provider.get("thirdParty"):
            if not model_name.strip():
                model_name = asr_provider["models"][0]
        else:
            if model_name not in asr_models(asr_provider):
                raise ValueError(f"Model {model_name} is not supported by {asr_provider['label']}")

    source_language = str(options.get("sourceLanguage") or "zh-cn").strip()
    if source_language.lower() in ("auto", "detect") or source_language not in translator.LANGNAME_DICT:
        source_language = "zh-cn"

    is_asr_only = (job_type == "asr")
    if is_asr_only or job_type == "render":
        translate_type = int(options.get("translateType") or 0)
        aisendsrt = False
        tts_type = int(options.get("ttsType") or 0)
        target_language = str(options.get("targetLanguage") or source_language)
        if target_language not in translator.LANGNAME_DICT:
            target_language = source_language
        timing_mode = str(options.get("timingMode") or "voice")
        timing_flags = TIMING_MODES.get(timing_mode, TIMING_MODES["voice"])
        # Stage 1 is ASR-only. Voice assignment belongs to Stage 3 and must
        # not activate dubbing/render preparation while transcription is running.
        voice_role = "No"
    else:
        translate_type = _required_index(options.get("translateType"), len(translator.TRANSLASTE_NAME_LIST), "translation engine")
        if translate_type not in TRANSLATION_BY_TYPE:
            raise ValueError(f"Unsupported translation engine: {translate_type}")
        _, aisendsrt = _translation_mode(options.get("translationMode"))
        tts_type = _optional_index(
            options.get("ttsType", tts.DEFAULT_TTS), len(tts.TTS_NAME_LIST), default=tts.DEFAULT_TTS
        )
        target_language = str(options.get("targetLanguage") or "")
        if target_language not in translator.LANGNAME_DICT:
            raise ValueError(f"Unknown target language: {target_language or 'missing'}")
        timing_mode = str(options.get("timingMode") or "voice")
        timing_flags = TIMING_MODES.get(timing_mode)
        if timing_flags is None:
            timing_flags = TIMING_MODES["voice"]
        voice_role = str(options.get("voiceRole") or "")
        if not voice_role and job_type != "translation":
            try:
                voice_role = next((voice for voice in role_provider(tts_type, langcode=target_language) if voice != "No"), "No")
            except Exception:
                voice_role = "No"
        elif not voice_role:
            voice_role = "No"
    voice_role = _resolve_voice_id_to_role(voice_role, tts_type)

    raw_vol = options.get("volume")
    if isinstance(raw_vol, (int, float)) and not isinstance(raw_vol, bool):
        try:
            diff = int(round((float(raw_vol) - 1.0) * 100))
            norm_volume = f"{diff:+d}%"
        except (OverflowError, ValueError):
            norm_volume = "+0%"
    elif raw_vol is not None and str(raw_vol).strip():
        s = str(raw_vol).strip()
        if re.match(r'^[+-]?\d+(\.\d+)?%$', s):
            norm_volume = s if s.startswith(("+", "-")) else f"+{s}"
        else:
            norm_volume = s
    else:
        norm_volume = "+0%"

    segments = options.get("segments") if isinstance(options.get("segments"), list) else []
    speaker_voice_map = options.get("speakerVoiceMap") if isinstance(options.get("speakerVoiceMap"), dict) else {}
    segment_voice_overrides = options.get("segmentVoiceOverrides") if isinstance(options.get("segmentVoiceOverrides"), dict) else {}
    line_roles = {}
    for index, segment in enumerate(segments, 1):
        if not isinstance(segment, dict):
            continue
        raw_spk = segment.get("speakerId")
        voice = (
            segment.get("voiceOverride")
            or segment_voice_overrides.get(str(segment.get("id")))
            or segment_voice_overrides.get(segment.get("id"))
            or (speaker_voice_map.get(raw_spk) if raw_spk is not None else None)
            or (speaker_voice_map.get(str(raw_spk)) if raw_spk is not None else None)
        )
        if voice:
            line_roles[str(index)] = _resolve_voice_id_to_role(str(voice), tts_type)

    params = asdict(file_info)
    params.update({
        "name": input_path.resolve().as_posix(),
        "target_dir": target_dir.resolve().as_posix(),
        "cache_folder": cache_dir.resolve().as_posix(),
        "source_language_code": source_language,
        "target_language_code": target_language,
        "recogn_type": recogn_type,
        "model_name": model_name,
        "translate_type": translate_type,
        "aisendsrt": aisendsrt,
        "tts_type": tts_type,
        "voice_role": voice_role,
        "line_roles": line_roles,
        "is_cuda": bool(options.get("useCuda", False)),
        "enable_diariz": bool(options.get("speakerDiarization", False)),
        "nums_diariz": int(options.get("speakerCount", 0) or 0),
        "voice_rate": str(options.get("voiceRate") or "+0%"),
        "volume": norm_volume,
        "pitch": "+0Hz",
        "deepgram_options": options.get("deepgramOptions") or options.get("deepgram_options") or {},
        **timing_flags,
        # ASR-only jobs should extract audio, transcribe, and diarize only.
        # Prevent PrepareMixin from spawning the no-audio video render thread,
        # which otherwise competes with Deepgram for CPU/GPU/disk resources.
        "video_autorate": False if is_asr_only else timing_flags.get("video_autorate", False),
        "subtitle_type": 0 if is_asr_only else 1,
        "only_out_dubbed_audio": bool(is_asr_only),
        "subtitles": str(options.get("subtitles") or ""),
        "segments": list(segments),
        "background_music": None if is_asr_only else options.get("backgroundMusicPath"),
        "backaudio_volume": _safe_volume(options.get("backgroundAudioVolume"), 0.8),
        "source_audio_volume": _safe_volume(options.get("originalAudioVolume"), 0.0),
        "thumbnail": None if is_asr_only else options.get("thumbnailPath"),
        "subtitle_style": options.get("subtitleStyle") if isinstance(options.get("subtitleStyle"), dict) else None,
        "force_recogn": bool(options.get("forceAsr", False)) if is_asr_only else False,
        "clear_cache": job_type not in {"render", "translation"},
        "embed_bgm": not is_asr_only,
        "project_id": str(pid) if pid else None,
    })
    if not is_asr_only:
        params["remove_noise"] = bool(options.get("removeNoise", False))
    return params
