# -*- coding: utf-8 -*-
"""Audio normalization and duration validation service for custom voice cloning."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Optional

from videotrans import tts
from videotrans.configure.excepts import FFmpegError
from videotrans.util.help_ffmpeg import runffmpeg, runffprobe, get_audio_time

logger = logging.getLogger("videotrans.audio_normalizer")

# Model-specific audio standards
PROVIDER_SAMPLE_RATES = {
    tts.ELEVENLABS_TTS: 44100,
    tts.OMNIVOICE_TTS: 24000,
    tts.VIENEU_TTS: 48000,
    tts.GEMINI_TTS: 24000,
}
DEFAULT_SAMPLE_RATE = 48000

MIN_REFERENCE_DURATION_SEC = 2.0
MAX_REFERENCE_DURATION_SEC = 30.0


def get_target_sample_rate(provider: int) -> int:
    """Return target sample rate based on TTS engine requirements."""
    return PROVIDER_SAMPLE_RATES.get(provider, DEFAULT_SAMPLE_RATE)


def probe_audio_duration(file_path: str | Path) -> float:
    """
    Measure audio file duration in seconds using soundfile or ffprobe.
    Raises ValueError if duration cannot be determined.
    """
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {p}")

    if p.suffix.lower() == ".mp3":
        # MP3 duration estimates can include ID3 metadata or assume a constant bitrate.
        # Count decoded samples from the audio stream that FFmpeg will normalize.
        try:
            audio = json.loads(runffprobe([
                "-v", "error",
                "-select_streams", "a:0",
                "-show_frames",
                "-show_entries", "frame=nb_samples:stream=sample_rate",
                "-of", "json",
                str(p),
            ]))
            sample_rate = int(audio["streams"][0]["sample_rate"])
            samples = sum(int(frame["nb_samples"]) for frame in audio["frames"])
            if sample_rate > 0 and samples > 0:
                return samples / sample_rate
        except (FFmpegError, ValueError, TypeError, KeyError, IndexError) as exc:
            logger.debug("Could not count decoded MP3 samples in %s: %s", p, exc)

    # Method 1: Try soundfile first (fast, header-based, no process spawn)
    try:
        import soundfile as sf
        with sf.SoundFile(str(p)) as f:
            if f.samplerate > 0 and len(f) > 0:
                duration = len(f) / float(f.samplerate)
                if duration > 0:
                    return duration
    except Exception:
        pass

    # Method 2: Try built-in get_audio_time (ms)
    try:
        ms = get_audio_time(str(p))
        if ms > 0:
            return ms / 1000.0
    except Exception:
        pass

    # Method 3: Direct ffprobe format=duration fallback
    try:
        out = runffprobe([
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(p),
        ])
        if out and out.strip():
            val = float(out.strip())
            if val > 0:
                return val
    except Exception as exc:
        logger.warning(f"Direct ffprobe duration check failed for {p}: {exc}")

    raise ValueError(f"Could not determine audio duration for {p.name}")


def detect_optimal_speech_segment(
    input_path: str | Path,
    target_duration: float = 10.0,
    min_duration: float = MIN_REFERENCE_DURATION_SEC,
    max_duration: float = MAX_REFERENCE_DURATION_SEC,
) -> dict:
    """
    Detect voice activity / nonsilent speech chunks and return recommended [start_time, end_time].
    Finds the cleanest speech segment within [min_duration, max_duration], targeting ~10s.
    """
    src = Path(input_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source audio file not found: {src}")

    total_duration = probe_audio_duration(src)
    if total_duration <= min_duration:
        return {
            "start_time": 0.0,
            "end_time": round(total_duration, 2),
            "duration": round(total_duration, 2),
            "original_duration": round(total_duration, 2),
        }

    # Attempt pydub silence detection
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        ext = src.suffix.lower().lstrip(".")
        format_name = "mp4" if ext == "m4a" else (ext if ext else None)
        audio = AudioSegment.from_file(str(src), format=format_name)
        silence_thresh = max(-50.0, audio.dBFS - 16.0) if audio.dBFS > -60 else -40.0
        nonsilent_chunks = detect_nonsilent(audio, min_silence_len=250, silence_thresh=silence_thresh, seek_step=25)

        if nonsilent_chunks:
            if total_duration <= target_duration:
                start_sec = max(0.0, (nonsilent_chunks[0][0] - 50) / 1000.0)
                end_sec = min(total_duration, (nonsilent_chunks[-1][1] + 150) / 1000.0)
                dur = end_sec - start_sec
                if dur >= min_duration:
                    return {
                        "start_time": round(start_sec, 2),
                        "end_time": round(end_sec, 2),
                        "duration": round(dur, 2),
                        "original_duration": round(total_duration, 2),
                    }

            step = 0.5
            best_start = 0.0
            best_speech_ms = 0
            max_start = max(0.0, total_duration - target_duration)
            curr = 0.0
            while curr <= max_start:
                curr_end = curr + target_duration
                curr_start_ms = curr * 1000.0
                curr_end_ms = curr_end * 1000.0
                speech_ms = 0
                for c_start, c_end in nonsilent_chunks:
                    overlap_start = max(curr_start_ms, c_start)
                    overlap_end = min(curr_end_ms, c_end)
                    if overlap_end > overlap_start:
                        speech_ms += (overlap_end - overlap_start)
                if speech_ms > best_speech_ms:
                    best_speech_ms = speech_ms
                    best_start = curr
                curr += step

            best_end = min(total_duration, best_start + target_duration)
            return {
                "start_time": round(best_start, 2),
                "end_time": round(best_end, 2),
                "duration": round(best_end - best_start, 2),
                "original_duration": round(total_duration, 2),
            }
    except Exception as exc:
        logger.debug("Speech detection fallback due to: %s", exc)

    chosen_end = min(total_duration, target_duration)
    return {
        "start_time": 0.0,
        "end_time": round(chosen_end, 2),
        "duration": round(chosen_end, 2),
        "original_duration": round(total_duration, 2),
    }


def normalize_reference_audio(
    input_path: str | Path,
    output_path: str | Path,
    provider: int = tts.VIENEU_TTS,
    min_duration: float = MIN_REFERENCE_DURATION_SEC,
    max_duration: float = MAX_REFERENCE_DURATION_SEC,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> dict:
    """
    Validate audio duration (2.0s - 30.0s) and transcode to pristine 16-bit mono PCM WAV
    at the target sample rate appropriate for the given TTS provider.
    Supports trimming via start_time and end_time.

    Returns dict with audio metadata:
        {"duration": float, "sample_rate": int, "channels": 1, "format": "pcm_s16le"}
    """
    src = Path(input_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source audio file not found: {src}")

    target_rate = get_target_sample_rate(provider)
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    has_trim = start_time is not None or end_time is not None
    if has_trim:
        orig_dur = probe_audio_duration(src)
        ss = max(0.0, float(start_time or 0.0))
        to = min(orig_dur, float(end_time)) if end_time is not None else orig_dur
        if to <= ss:
            raise ValueError(f"End time ({to:.2f}s) must be greater than start time ({ss:.2f}s)")
        expected_dur = to - ss
        if expected_dur < min_duration or expected_dur > max_duration:
            raise ValueError(
                f"Audio duration must be between {min_duration:.1f} and {max_duration:.1f} seconds (got {expected_dur:.2f}s)"
            )
        cmd = [
            "-y",
            "-i", str(src),
            "-ss", f"{ss:.3f}",
            "-to", f"{to:.3f}",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-vn",
            "-ac", "1",
            "-ar", str(target_rate),
            "-acodec", "pcm_s16le",
            str(dest),
        ]
        cmd_simple = [
            "-y",
            "-i", str(src),
            "-ss", f"{ss:.3f}",
            "-to", f"{to:.3f}",
            "-vn",
            "-ac", "1",
            "-ar", str(target_rate),
            "-acodec", "pcm_s16le",
            str(dest),
        ]
    else:
        duration = probe_audio_duration(src)
        if duration < min_duration or duration > max_duration:
            raise ValueError(
                f"Audio duration must be between {min_duration:.1f} and {max_duration:.1f} seconds (got {duration:.2f}s)"
            )

        cmd = [
            "-y",
            "-i", str(src),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-vn",
            "-ac", "1",
            "-ar", str(target_rate),
            "-acodec", "pcm_s16le",
            str(dest),
        ]
        cmd_simple = [
            "-y",
            "-i", str(src),
            "-vn",
            "-ac", "1",
            "-ar", str(target_rate),
            "-acodec", "pcm_s16le",
            str(dest),
        ]

    try:
        runffmpeg(cmd, force_cpu=True)
    except Exception:
        try:
            runffmpeg(cmd_simple, force_cpu=True)
        except Exception as exc:
            logger.exception(f"Audio normalization failed for {src}: {exc}", exc_info=True)
            raise FFmpegError(f"Audio normalization failed: {exc}") from exc

    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"Transcoded output WAV was not generated: {dest}")

    out_dur = probe_audio_duration(dest)
    if out_dur < (min_duration - 0.2) or out_dur > (max_duration + 0.5):
        raise ValueError(
            f"Audio duration must be between {min_duration:.1f} and {max_duration:.1f} seconds (got {out_dur:.2f}s)"
        )

    return {
        "duration": round(out_dur, 3),
        "sample_rate": target_rate,
        "channels": 1,
        "format": "pcm_s16le",
    }
