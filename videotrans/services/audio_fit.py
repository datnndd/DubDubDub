"""Smart Fit and pitch-preserving audio timing adjustment.

Inspired by VoiceStudio-0.5.0 fit_planner and ffmpeg atempo chaining.
Provides slot calculation with slack absorption, rate determination,
and pitch-preserving audio stretching.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _is_transient_lock_error(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32)


def _replace_file_with_retry(src: Path, dst: Path, retries: int = 8, delay: float = 0.01) -> None:
    for attempt in range(retries):
        try:
            os.replace(str(src), str(dst))
            return
        except OSError as exc:
            if not _is_transient_lock_error(exc) or attempt + 1 >= retries:
                raise
            time.sleep(delay * (attempt + 1))


def _unlink_with_retry(path: Path, retries: int = 5, delay: float = 0.005) -> None:
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError as exc:
            if not _is_transient_lock_error(exc) or attempt + 1 >= retries:
                return
            time.sleep(delay * (attempt + 1))


@dataclass(frozen=True)
class FitParams:
    """Parameters for Smart Fit audio speed adjustment."""
    max_audio_rate: float = 1.35
    gap_guard_s: float = 0.15
    min_audio_rate: float = 1.0  # 1.0 keeps natural speed for shorter audio
    max_audio_rate_hard: float = 2.0


def calculate_slot_duration(
    start_s: float,
    end_s: float,
    next_start_s: Optional[float] = None,
    total_dur_s: Optional[float] = None,
    gap_guard_s: float = 0.15,
) -> float:
    """Calculate the usable audio slot duration with gap slack absorption.

    The slot extends past the segment's end into the silent gap before
    the next segment, leaving gap_guard_s clear of the next onset.
    """
    start = max(0.0, float(start_s))
    end = max(start, float(end_s))
    base_slot = max(0.001, end - start)

    if next_start_s is not None:
        next_start = float(next_start_s)
        effective_end = max(end, next_start - gap_guard_s)
        effective_end = min(max(effective_end, start), max(next_start, end))
        return max(0.001, round(effective_end - start, 4))

    if total_dur_s is not None and total_dur_s > 0:
        total_dur = float(total_dur_s)
        effective_end = max(end, total_dur)
        return max(0.001, round(effective_end - start, 4))

    return round(base_slot, 4)


def calculate_fit_rate(
    natural_duration_s: float,
    slot_duration_s: float,
    params: Optional[FitParams] = None,
) -> tuple[float, str]:
    """Calculate required audio playback speed factor and status.

    Returns:
        (rate, status) where:
        - rate: playback speed multiplier (>1.0 speeds up, 1.0 unchanged)
        - status: "fits" | "audio_stretched" | "overflow_trimmed" | "audio_slowed"
    """
    params = params or FitParams()
    slot = max(0.001, float(slot_duration_s))
    natural = max(0.0, float(natural_duration_s))

    if natural <= _EPS:
        return 1.0, "fits"

    need = natural / slot

    # Fits within slot naturally
    if need <= 1.0 + _EPS:
        if need > _EPS and need < 0.95 and params.min_audio_rate < 1.0 - _EPS:
            return round(max(need, params.min_audio_rate), 4), "audio_slowed"
        return 1.0, "fits"

    # Fits within user-configured max audio rate
    if need <= params.max_audio_rate + _EPS:
        return round(need, 4), "audio_stretched"

    # Exceeds max rate: cap at max_audio_rate, residual overflows
    return round(params.max_audio_rate, 4), "overflow_trimmed"


def build_atempo_chain(ratio: float) -> str:
    """Build an atempo filter chain for arbitrary ratios.

    ffmpeg's atempo filter is limited to [0.5, 2.0] per stage.
    Chaining multiple stages multiplies the effective ratio while
    keeping each individual stage in the valid range, preserving pitch.
    """
    if ratio <= 0:
        return "atempo=1.0"

    stages: list[str] = []
    remaining = float(ratio)

    while remaining > 2.0:
        stages.append("atempo=2.0")
        remaining /= 2.0

    while remaining < 0.5:
        stages.append("atempo=0.5")
        remaining /= 0.5

    stages.append(f"atempo={remaining:.6f}")
    return ",".join(stages)


def get_audio_duration_seconds(audio_path: Path | str) -> float:
    """Get the duration of an audio file in seconds quickly."""
    p = Path(audio_path)
    if not p.is_file() or p.stat().st_size == 0:
        return 0.0

    # Try soundfile header first (very fast, no full decode)
    try:
        import soundfile as sf
        info = sf.info(str(p))
        if info.samplerate > 0 and info.frames >= 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    # Try wave for wav files
    try:
        import wave
        with wave.open(str(p), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return float(frames) / float(rate)
    except Exception:
        pass

    # Fallback to ffprobe
    try:
        from videotrans.util.help_ffmpeg import runffprobe
        data = runffprobe(str(p))
        if data and "format" in data and "duration" in data["format"]:
            return float(data["format"]["duration"])
    except Exception:
        pass

    return 0.0


def apply_pitch_preserving_stretch(
    input_wav: Path | str,
    output_wav: Path | str,
    speed_factor: float,
    max_duration_s: Optional[float] = None,
) -> bool:
    """Apply pitch-preserving time stretch to an audio file using ffmpeg atempo.

    Returns True on success, False on failure.
    """
    in_path = Path(input_wav).resolve()
    out_path = Path(output_wav).resolve()

    if not in_path.is_file():
        logger.error("Input audio file not found: %s", in_path)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If speed factor is essentially 1.0 and no trimming requested
    if abs(speed_factor - 1.0) < 0.005 and (max_duration_s is None or max_duration_s <= 0):
        if in_path != out_path:
            shutil.copy2(str(in_path), str(out_path))
        return True

    filter_str = build_atempo_chain(speed_factor)

    with tempfile.NamedTemporaryFile(
        prefix=f".{out_path.stem}.",
        suffix=".tmp",
        delete=False,
        dir=str(out_path.parent),
    ) as tmp_f:
        tmp_output = Path(tmp_f.name)

    cmd = [
        "-y",
        "-i",
        str(in_path),
        "-filter:a",
        filter_str,
        "-ar",
        "48000",
        "-c:a",
        "pcm_s16le",
    ]
    if max_duration_s is not None and max_duration_s > 0:
        cmd.extend(["-t", f"{max_duration_s:.6f}"])
    cmd.extend(["-f", "wav", str(tmp_output)])

    try:
        from videotrans.util.help_ffmpeg import runffmpeg
        runffmpeg(cmd, force_cpu=True)
        if tmp_output.is_file() and tmp_output.stat().st_size > 0:
            _replace_file_with_retry(tmp_output, out_path)
            return True
        logger.warning("ffmpeg speedup produced empty output: %s", tmp_output)
        return False
    except Exception as exc:
        logger.exception("Failed to apply atempo stretch: %s", exc)
        return False
    finally:
        _unlink_with_retry(tmp_output)


def smart_fit_audio_file(
    input_wav: Path | str,
    slot_duration_s: float,
    max_speed_rate: float = 1.35,
    output_wav: Optional[Path | str] = None,
    min_audio_rate: float = 1.0,
) -> tuple[Path, float, str]:
    """Determine fit rate and adjust audio playback speed if needed.

    Returns:
        (result_path, applied_speed_factor, status)
    """
    in_p = Path(input_wav).resolve()
    out_p = Path(output_wav).resolve() if output_wav else in_p

    if not in_p.is_file():
        return in_p, 1.0, "not_found"

    if slot_duration_s <= 0:
        return in_p, 1.0, "fits"

    natural_dur = get_audio_duration_seconds(in_p)
    params = FitParams(
        max_audio_rate=max_speed_rate,
        min_audio_rate=min_audio_rate,
    )
    speed_factor, status = calculate_fit_rate(natural_dur, slot_duration_s, params)

    if status == "fits" and abs(speed_factor - 1.0) < 0.005:
        if out_p != in_p:
            shutil.copy2(str(in_p), str(out_p))
        return out_p, 1.0, "fits"

    # Need speedup or slowdown
    # Do not hard-truncate with max_duration_s (-t) to prevent filter buffer truncation
    success = apply_pitch_preserving_stretch(
        in_p,
        out_p,
        speed_factor=speed_factor,
        max_duration_s=None,
    )

    if success:
        return out_p, speed_factor, status
    return in_p, 1.0, "error"
