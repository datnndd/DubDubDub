# -*- coding: utf-8 -*-
"""Helpers for translating WebUI frontend options into task configuration dictionaries."""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

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


def _pick_defined(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first mapping value whose key is present and not None."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


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


def _parse_ocr_roi(raw_roi: Any) -> tuple[float, float, float, float] | None:
    if raw_roi is None:
        return None
    if isinstance(raw_roi, dict):
        vals = (
            _pick_defined(raw_roi, "x"),
            _pick_defined(raw_roi, "y"),
            _pick_defined(raw_roi, "width", "w"),
            _pick_defined(raw_roi, "height", "h"),
        )
    elif isinstance(raw_roi, (list, tuple)) and len(raw_roi) == 4:
        vals = tuple(raw_roi)
    else:
        raise ValueError("ocrRoi must be a 4-element [x, y, w, h] region")

    parsed: list[float] = []
    for v in vals:
        if isinstance(v, bool):
            raise ValueError("ocrRoi must be a 4-element [x, y, w, h] region")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError("ocrRoi must be a 4-element [x, y, w, h] region") from None
        if not math.isfinite(f):
            raise ValueError("ocrRoi must be a 4-element [x, y, w, h] region")
        parsed.append(f)

    x, y, w, h = parsed
    if x < 0.0 or y < 0.0 or w <= 0.0 or h <= 0.0 or x + w > 1.001 or y + h > 1.001:
        raise ValueError("ocrRoi coordinates must be normalized within [0, 1] with positive width and height")
    if x + w > 1.0:
        w = 1.0 - x
    if y + h > 1.0:
        h = 1.0 - y
    if w <= 0.0 or h <= 0.0:
        raise ValueError("ocrRoi coordinates must be normalized within [0, 1] with positive width and height")
    return (x, y, w, h)


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

    subtitle_source = str(
        options.get("subtitleSource")
        or options.get("subtitle_source")
        or "audio_asr"
    ).strip().lower()
    if subtitle_source not in {"audio_asr", "video_ocr"}:
        raise ValueError(f"Unsupported subtitle source: {subtitle_source}")

    raw_roi = _pick_defined(options, "ocrRoi", "ocr_roi")
    ocr_roi = _parse_ocr_roi(raw_roi)
    ocr_roi_confirmed = bool(_pick_defined(options, "ocrRoiConfirmed", "ocr_roi_confirmed", default=False))
    if subtitle_source == "video_ocr" and job_type not in {"render", "translation"}:
        if ocr_roi is None:
            raise ValueError("ocrRoi is required when subtitleSource is video_ocr")
        if not ocr_roi_confirmed:
            raise ValueError("ocrRoiConfirmed must be true when subtitleSource is video_ocr")

    raw_ocr_opts_camel = options.get("ocrOptions") if isinstance(options.get("ocrOptions"), dict) else {}
    raw_ocr_opts_snake = options.get("ocr_options") if isinstance(options.get("ocr_options"), dict) else {}
    raw_ocr_opts = {**raw_ocr_opts_snake, **raw_ocr_opts_camel}
    ocr_options: dict[str, Any] = {}
    if raw_ocr_opts:
        raw_interval = _pick_defined(raw_ocr_opts, "coarseIntervalMs", "coarse_interval_ms")
        if isinstance(raw_interval, bool):
            raise ValueError("ocrOptions.coarseIntervalMs must be an integer")
        if raw_interval is not None:
            try:
                interval_val = int(raw_interval)
            except (TypeError, ValueError):
                raise ValueError("ocrOptions.coarseIntervalMs must be an integer") from None
            ocr_options["coarse_interval_ms"] = max(100, min(5000, interval_val))
        raw_use_angle_cls = _pick_defined(raw_ocr_opts, "useAngleCls", "use_angle_cls")
        if raw_use_angle_cls is not None:
            ocr_options["use_angle_cls"] = bool(raw_use_angle_cls)
        raw_pm = _pick_defined(raw_ocr_opts, "paragraphMode", "paragraph_mode")
        if raw_pm is not None:
            pm = str(raw_pm).strip()
            if pm not in {"preserve_lines", "unwrap"}:
                raise ValueError(f"Unsupported OCR paragraphMode: {pm}")
            ocr_options["paragraph_mode"] = pm

    if job_type in {"render", "translation"} or subtitle_source == "video_ocr":
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
            model_name = model_name.strip() or "nova-3"
            dg_models = set(recognition.get_model_by_type(recognition.Deepgram))
            raw_dg_setting = runtime_config.settings.get("deepgram_model", "")
            if isinstance(raw_dg_setting, (list, tuple, set)):
                dg_models.update(str(m).strip() for m in raw_dg_setting if str(m).strip())
            else:
                dg_models.update(m.strip() for m in str(raw_dg_setting).split(",") if m.strip())
            if model_name not in dg_models and not model_name.startswith(("nova-", "enhanced", "base")):
                raise ValueError(f"Model {model_name} is not supported by {asr_provider['label']}")
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
        translate_type = _optional_index(options.get("translateType"), len(translator.TRANSLASTE_NAME_LIST), default=0)
        _, aisendsrt = _translation_mode(options.get("translationMode")) if options.get("translationMode") else ("srt", False)
        tts_type = _optional_index(
            options.get("ttsType", tts.DEFAULT_TTS), len(tts.TTS_NAME_LIST), default=tts.DEFAULT_TTS
        )
        target_language = str(options.get("targetLanguage") or source_language)
        if target_language not in translator.LANGNAME_DICT:
            target_language = source_language
        timing_mode = str(options.get("timingMode") or "voice")
        timing_flags = TIMING_MODES.get(timing_mode, TIMING_MODES["voice"])
        if is_asr_only:
            # Stage 1 is ASR-only. Voice assignment belongs to Stage 3 and must
            # not activate dubbing/render preparation while transcription is running.
            voice_role = "No"
        else:
            voice_role = str(options.get("voiceRole") or "").strip() or "No"
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

    audio_mix = options.get("audioMix") if isinstance(options.get("audioMix"), Mapping) else {}
    raw_vol = options.get("volume")
    if raw_vol is None:
        raw_vol = options.get("dubbedAudioVolume")
    if raw_vol is None and "dubbed" in audio_mix and audio_mix["dubbed"] is not None and not isinstance(audio_mix["dubbed"], bool):
        try:
            dub_pct = float(audio_mix["dubbed"])
            if math.isfinite(dub_pct):
                raw_vol = max(0.0, min(1.5, dub_pct / 100.0))
        except (TypeError, ValueError, OverflowError):
            pass

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

    if job_type == "render" and (not voice_role or voice_role == "No") and options.get("voiceRole") != "No":
        assigned_role = next((v for v in line_roles.values() if v and v != "No"), None)
        if not assigned_role and speaker_voice_map:
            for spk_v in speaker_voice_map.values():
                resolved_spk_v = _resolve_voice_id_to_role(str(spk_v), tts_type)
                if resolved_spk_v and resolved_spk_v != "No":
                    assigned_role = resolved_spk_v
                    break
        if assigned_role:
            voice_role = assigned_role

    subtitles_str = str(options.get("subtitles") or "")
    if not subtitles_str.strip() and job_type == "render" and segments:
        from videotrans.util._srt_parse import ms_to_time_string
        from videotrans.util.segment_ops import parse_timestamp
        srt_blocks: list[str] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            txt = str(
                seg.get("targetText")
                if seg.get("targetText") is not None and str(seg.get("targetText")).strip()
                else (seg.get("sourceText") if seg.get("sourceText") is not None else (seg.get("text") or ""))
            ).strip()
            if not txt:
                continue
            has_timing = any(
                seg.get(k) is not None for k in ("startSec", "endSec", "startTime", "endTime")
            )
            if not has_timing:
                continue
            raw_start = seg.get("startSec") if seg.get("startSec") is not None else seg.get("startTime", 0.0)
            raw_end = seg.get("endSec") if seg.get("endSec") is not None else seg.get("endTime", 0.0)
            try:
                s_sec = max(0.0, parse_timestamp(raw_start))
                e_sec = max(s_sec + 0.1, parse_timestamp(raw_end))
            except Exception:
                s_sec, e_sec = 0.0, 1.0
            s_str = ms_to_time_string(ms=int(round(s_sec * 1000)))
            e_str = ms_to_time_string(ms=int(round(e_sec * 1000)))
            srt_blocks.append(f"{len(srt_blocks) + 1}\n{s_str} --> {e_str}\n{txt}")
        if srt_blocks:
            subtitles_str = "\n\n".join(srt_blocks)

    raw_orig_vol = options.get("originalAudioVolume")
    if raw_orig_vol is None and "original" in audio_mix and audio_mix["original"] is not None and not isinstance(audio_mix["original"], bool):
        try:
            raw_orig_vol = float(audio_mix["original"]) / 100.0
        except (TypeError, ValueError, OverflowError):
            pass

    raw_bgm_vol = options.get("backgroundAudioVolume")
    if raw_bgm_vol is None and "background" in audio_mix and audio_mix["background"] is not None and not isinstance(audio_mix["background"], bool):
        try:
            raw_bgm_vol = float(audio_mix["background"]) / 100.0
        except (TypeError, ValueError, OverflowError):
            pass

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
        "enable_diariz": False if subtitle_source == "video_ocr" else bool(options.get("speakerDiarization", False)),
        "nums_diariz": 0 if subtitle_source == "video_ocr" else int(options.get("speakerCount", 0) or 0),
        "voice_rate": str(options.get("voiceRate") or "+0%"),
        "volume": norm_volume,
        "pitch": "+0Hz",
        "deepgram_options": options.get("deepgramOptions") or options.get("deepgram_options") or {},
        "subtitle_source": subtitle_source,
        "ocr_roi": ocr_roi,
        "ocr_roi_confirmed": ocr_roi_confirmed,
        "ocr_options": ocr_options or None,
        **timing_flags,
        # ASR-only jobs should extract audio, transcribe, and diarize only.
        # Prevent PrepareMixin from spawning the no-audio video render thread,
        # which otherwise competes with Deepgram for CPU/GPU/disk resources.
        "video_autorate": False if is_asr_only else timing_flags.get("video_autorate", False),
        "subtitle_type": 0 if is_asr_only else 1,
        "only_out_dubbed_audio": bool(is_asr_only),
        "subtitles": subtitles_str,
        "segments": list(segments),
        "background_music": None if is_asr_only else options.get("backgroundMusicPath"),
        "backaudio_volume": _safe_volume(raw_bgm_vol, 0.8),
        "source_audio_volume": _safe_volume(raw_orig_vol, 0.0),
        "thumbnail": None if is_asr_only else options.get("thumbnailPath"),
        "subtitle_style": options.get("subtitleStyle") if isinstance(options.get("subtitleStyle"), dict) else None,
        "annotations": (
            [dict(a) for a in options.get("annotations", []) if isinstance(a, dict)]
            if (not is_asr_only and isinstance(options.get("annotations"), list))
            else None
        ),
        "burn_annotations": (
            False
            if is_asr_only
            else bool(_pick_defined(options, "burnAnnotations", "burn_annotations", default=False))
        ),
        "force_recogn": bool(options.get("forceAsr", False)) if is_asr_only else False,
        "clear_cache": job_type not in {"render", "translation"},
        "embed_bgm": not is_asr_only,
        "project_id": str(pid) if pid else None,
    })
    if not is_asr_only:
        params["remove_noise"] = False if subtitle_source == "video_ocr" else bool(options.get("removeNoise", False))
    return params
