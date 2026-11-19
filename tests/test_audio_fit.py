import math
import wave
from pathlib import Path
import pytest

from videotrans.services.audio_fit import (
    FitParams,
    calculate_slot_duration,
    calculate_fit_rate,
    build_atempo_chain,
    get_audio_duration_seconds,
    apply_pitch_preserving_stretch,
    smart_fit_audio_file,
)


def test_calculate_slot_duration_base():
    slot = calculate_slot_duration(0.0, 2.5)
    assert slot == 2.5


def test_calculate_slot_duration_with_slack():
    # Slot absorbs gap up to next_start - gap_guard (0.05)
    slot = calculate_slot_duration(1.0, 3.0, next_start_s=4.0, gap_guard_s=0.05)
    assert slot == pytest.approx(2.95, rel=1e-3)


def test_calculate_slot_duration_tail():
    # Last segment absorbs slack up to total video duration
    slot = calculate_slot_duration(5.0, 7.0, total_dur_s=10.0)
    assert slot == 5.0


def test_calculate_fit_rate_fits():
    params = FitParams(max_audio_rate=1.25, min_audio_rate=1.0)
    rate, status = calculate_fit_rate(2.0, 2.5, params)
    assert rate == 1.0
    assert status == "fits"


def test_calculate_fit_rate_stretched():
    params = FitParams(max_audio_rate=1.3)
    rate, status = calculate_fit_rate(2.4, 2.0, params)
    assert rate == pytest.approx(1.2, rel=1e-3)
    assert status == "audio_stretched"


def test_calculate_fit_rate_overflow_capped():
    params = FitParams(max_audio_rate=1.25)
    rate, status = calculate_fit_rate(3.5, 2.0, params)
    # Capped at max_audio_rate
    assert rate == 1.25
    assert status == "overflow_trimmed"


def test_build_atempo_chain_single():
    res = build_atempo_chain(1.25)
    assert res == "atempo=1.250000"


def test_build_atempo_chain_multi():
    res = build_atempo_chain(2.5)
    assert "atempo=2.0" in res
    assert "atempo=1.250000" in res


def _create_synthetic_wav(file_path: Path, duration_s: float, sample_rate: int = 16000):
    num_samples = int(duration_s * sample_rate)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Generate simple sine wave
        samples = bytearray()
        for i in range(num_samples):
            val = int(3000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            samples.extend(val.to_bytes(2, byteorder="little", signed=True))
        wf.writeframes(samples)


def test_smart_fit_audio_file_execution(tmp_path):
    input_wav = tmp_path / "test_input.wav"
    output_wav = tmp_path / "test_output.wav"
    # Create 2.0s audio
    _create_synthetic_wav(input_wav, duration_s=2.0)

    dur = get_audio_duration_seconds(input_wav)
    assert dur == pytest.approx(2.0, rel=0.05)

    # Fit into 1.6s slot (need = 2.0 / 1.6 = 1.25x)
    res_path, applied_rate, status = smart_fit_audio_file(
        input_wav,
        slot_duration_s=1.6,
        max_speed_rate=1.25,
        output_wav=output_wav,
    )
    assert status == "audio_stretched"
    assert applied_rate == pytest.approx(1.25, rel=0.01)
    assert res_path.is_file()

    new_dur = get_audio_duration_seconds(res_path)
    assert new_dur == pytest.approx(1.6, rel=0.1)


def test_apply_pitch_preserving_stretch_temp_name_dash_f_wav_and_prune_safety(tmp_path, monkeypatch):
    import os
    import time
    from videotrans.tts._cache import _is_temp_cache_name, prune_cache_dir
    from videotrans.util import help_ffmpeg

    prev_dir = tmp_path / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    in_wav = prev_dir / "prev_src.wav"
    out_wav = prev_dir / "prev_target.wav"
    _create_synthetic_wav(in_wav, duration_s=1.0, sample_rate=24000)

    old_ts = time.time() - 3600
    for stale_name in ("stale_a.wav", "stale_b.wav"):
        stale = prev_dir / stale_name
        _create_synthetic_wav(stale, duration_s=0.3, sample_rate=24000)
        os.utime(stale, (old_ts, old_ts))

    real_runffmpeg = help_ffmpeg.runffmpeg
    observed: dict[str, object] = {}

    def wrapped_runffmpeg(cmd, *args, **kwargs):
        observed["cmd"] = list(cmd)
        tmp_file = Path(cmd[-1])
        assert cmd[-3:-1] == ["-f", "wav"]
        assert tmp_file.name.startswith(".prev_target.") and tmp_file.name.endswith(".tmp")
        assert _is_temp_cache_name(tmp_file.name) is True

        res = real_runffmpeg(cmd, *args, **kwargs)
        assert tmp_file.is_file() and tmp_file.stat().st_size > 44
        prune_cache_dir(
            prev_dir,
            max_age_seconds=86400,
            max_bytes=10 * 1024 * 1024,
            max_files=1,
            primary_suffixes=(".wav",),
        )
        observed["survived_prune"] = tmp_file.is_file()
        return res

    monkeypatch.setattr(help_ffmpeg, "runffmpeg", wrapped_runffmpeg)
    assert apply_pitch_preserving_stretch(in_wav, out_wav, speed_factor=1.5) is True
    assert observed.get("survived_prune") is True
    assert get_audio_duration_seconds(out_wav) == pytest.approx(1.0 / 1.5, abs=0.08)
    leftover_tmps = [p.name for p in prev_dir.iterdir() if p.name.startswith(".") or ".tmp" in p.name.lower()]
    assert leftover_tmps == []


def test_apply_pitch_preserving_stretch_retries_permission_error_replace_and_finally_unlink(tmp_path, monkeypatch):
    import os
    from videotrans.services import audio_fit

    in_wav = tmp_path / "src.wav"
    out_wav = tmp_path / "dst.wav"
    _create_synthetic_wav(in_wav, duration_s=0.9, sample_rate=24000)

    real_replace = os.replace
    real_unlink = Path.unlink
    sleep_calls: list[float] = []
    monkeypatch.setattr(audio_fit.time, "sleep", lambda s: sleep_calls.append(s))

    # Part A: os.replace retries on PermissionError / WinError 32
    replace_attempts = {"n": 0}

    def flaky_replace(src, dst):
        replace_attempts["n"] += 1
        if replace_attempts["n"] <= 3:
            exc = PermissionError(13, "Sharing violation")
            if replace_attempts["n"] == 2:
                exc.winerror = 32
            raise exc
        return real_replace(src, dst)

    monkeypatch.setattr(audio_fit.os, "replace", flaky_replace)
    assert apply_pitch_preserving_stretch(in_wav, out_wav, speed_factor=1.5) is True
    assert replace_attempts["n"] == 4
    assert len(sleep_calls) == 3
    assert out_wav.is_file() and out_wav.stat().st_size > 44
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name or p.name.startswith(".")] == []

    # Part B: finally unlink retries PermissionError without raising
    def always_fail_replace(src, dst):
        exc = PermissionError(13, "Locked")
        exc.winerror = 5
        raise exc

    unlink_attempts = {"n": 0}

    def flaky_unlink(self, missing_ok=False):
        if ".tmp" in self.name:
            unlink_attempts["n"] += 1
            if unlink_attempts["n"] == 1:
                exc = PermissionError(13, "Reader still open")
                exc.winerror = 32
                raise exc
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(audio_fit.os, "replace", always_fail_replace)
    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    assert apply_pitch_preserving_stretch(in_wav, out_wav, speed_factor=1.5) is False
    assert unlink_attempts["n"] >= 2
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name or p.name.startswith(".")] == []
