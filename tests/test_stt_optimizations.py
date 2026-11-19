# -*- coding: utf-8 -*-
"""Comprehensive offline tests for Milestone M3 STT optimizations and bug fixes."""

import functools
import importlib.util
import json
import sys
import threading
import time
import wave
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from videotrans.configure import config
from videotrans.configure.excepts import (
    SpeechToTextError,
    StopRetry,
    StopTask,
    SttTimeoutError,
)
from videotrans.task.taskcfg import SrtItem
from videotrans.util.help_srt import get_srt_from_list, ms_to_time_string


def _write_pcm_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> Path:
    sf.write(str(path), samples.astype(np.int16), sr, subtype="PCM_16")
    return path


def _make_srt(text: str, start: int, end: int, line: int = 1) -> SrtItem:
    s_raw = ms_to_time_string(ms=start)
    e_raw = ms_to_time_string(ms=end)
    return SrtItem(
        line=line,
        text=text,
        start_time=start,
        end_time=end,
        startraw=s_raw,
        endraw=e_raw,
        time=f"{s_raw} --> {e_raw}",
    )


def _make_deepgram_api_error(exc_cls, message: str, status: str):
    try:
        err = exc_cls(message, status)
    except TypeError:
        err = exc_cls(message)
    if not hasattr(err, "status"):
        err.status = status
    return err


def _make_elevenlabs_api_error(exc_cls, status_code: int, body: dict):
    try:
        err = exc_cls(status_code=status_code, body=body)
    except TypeError:
        err = exc_cls(str(body))
        err.body = body
    if not hasattr(err, "status_code"):
        err.status_code = status_code
    return err


@functools.lru_cache(maxsize=1)
def _load_qwen_asr_utils():
    pkg_spec = importlib.util.find_spec("qwen_asr")
    if pkg_spec is None or not pkg_spec.submodule_search_locations:
        pytest.skip("qwen_asr package is not installed")
    utils_path = Path(pkg_spec.submodule_search_locations[0]) / "inference" / "utils.py"
    if not utils_path.is_file():
        pytest.skip(f"qwen_asr inference/utils.py not found at {utils_path}")
    spec = importlib.util.spec_from_file_location("_qwen_asr_inference_utils_direct", utils_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# 1. Shared runtime retry helper (videotrans/recognition/_retry.py)
# ============================================================================

def test_retry_resolve_max_attempts_runtime(monkeypatch):
    from videotrans.recognition import _retry

    monkeypatch.setitem(config.settings, "retry_nums", 1)
    assert _retry.resolve_max_attempts() == 3

    monkeypatch.setitem(config.settings, "retry_nums", 4)
    assert _retry.resolve_max_attempts() == 5

    assert _retry.resolve_max_attempts(2) == 2
    assert _retry.resolve_max_attempts(0) == 1

    monkeypatch.setitem(config.settings, "retry_nums", "invalid")
    assert _retry.resolve_max_attempts() == 3


def test_retry_exponential_backoff_cancel_and_no_retry(monkeypatch):
    from videotrans.recognition import _retry

    monkeypatch.setitem(config.settings, "retry_nums", 3)  # max_attempts = 4
    sleeps = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient timeout")
        return "ok"

    res = _retry.run_with_retry(
        flaky,
        should_retry=lambda exc: isinstance(exc, TimeoutError),
        base_delay=1.0,
        max_delay=4.0,
        sleep_fn=sleeps.append,
        label="test",
    )
    assert res == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2
    assert 0.75 <= sleeps[0] <= 1.25
    assert 1.5 <= sleeps[1] <= 2.5

    # Cancellation during retry stops immediately with StopTask
    cancel_flag = {"cancelled": False}
    calls["n"] = 0

    def always_timeout():
        calls["n"] += 1
        cancel_flag["cancelled"] = True
        raise TimeoutError("boom")

    with pytest.raises(StopTask):
        _retry.run_with_retry(
            always_timeout,
            should_retry=lambda exc: True,
            is_cancelled=lambda: cancel_flag["cancelled"],
            sleep_fn=lambda _: None,
            label="cancel-test",
        )
    assert calls["n"] == 1

    # NO_RETRY_EXCEPT bypasses retries immediately
    calls["n"] = 0

    def abort():
        calls["n"] += 1
        raise StopRetry("fatal 400")

    with pytest.raises(StopRetry):
        _retry.run_with_retry(
            abort,
            should_retry=lambda exc: True,
            sleep_fn=lambda _: None,
            label="no-retry",
        )
    assert calls["n"] == 1


def test_is_retryable_status_and_mock_httpx_safety(monkeypatch):
    from videotrans.recognition import _retry

    for code in (408, 429, 500, 502, 503, 504, "429", "503"):
        assert _retry.is_retryable_status(code) is True
    for code in (400, 401, 403, 404, 422, None, "abc", True):
        assert _retry.is_retryable_status(code) is False

    # Simulate tests/conftest.py MagicMock httpx missing TimeoutException/ReadTimeout
    mock_httpx = MagicMock()
    mock_httpx.ConnectTimeout = type("ConnectTimeout", (Exception,), {})
    mock_httpx.ReadError = type("ReadError", (Exception,), {})
    monkeypatch.setattr(_retry, "httpx", mock_httpx)

    retry_types = _retry.build_httpx_retry_types()
    assert TimeoutError in retry_types
    assert mock_httpx.ConnectTimeout in retry_types
    assert mock_httpx.ReadError in retry_types
    assert all(isinstance(t, type) and issubclass(t, BaseException) for t in retry_types)


# ============================================================================
# 2. DeepgramRecogn fixes
# ============================================================================

@pytest.mark.parametrize(
    "detect_lang,expected_lang_opt,expected_detect_opt,raw_text,expected_text",
    [
        (None, "zh", None, "你 好", "你好"),
        ("", "zh", None, "你 好", "你好"),
        ("auto", None, True, "Hello world", "Hello world"),
        ("AUTO", None, True, "Hello world", "Hello world"),
        ("a", "a", None, "Hello world", "Hello world"),
        ("ZH-CN", "zh", None, "繁體 测试", "繁体测试"),
    ],
)
def test_deepgram_detect_language_edge_cases(
    tmp_path, detect_lang, expected_lang_opt, expected_detect_opt, raw_text, expected_text
):
    from videotrans.recognition._deepgram import DeepgramRecogn

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")

    rec = DeepgramRecogn(
        audio_file=str(audio),
        cache_folder=str(tmp_path),
        model_name="nova-3",
        detect_language=detect_lang,
    )

    captured = {}

    def fake_transcribe(payload, options, timeout=600, **kwargs):
        captured["language"] = getattr(options, "language", None)
        captured["detect_language"] = getattr(options, "detect_language", None)
        res = MagicMock()
        utt = SimpleNamespace(start=0.0, end=1.0, transcript=raw_text, speaker=0)
        res.results.utterances = [utt]
        para = SimpleNamespace(
            start=0.0,
            end=1.0,
            transcript=raw_text,
            speaker=0,
            sentences=[SimpleNamespace(start=0.0, end=1.0, text=raw_text)],
        )
        alt = SimpleNamespace(
            paragraphs=SimpleNamespace(transcript=raw_text, paragraphs=[para])
        )
        res.results.channels = [SimpleNamespace(alternatives=[alt], detected_language=None)]
        return res

    with patch("videotrans.recognition._deepgram.params", {"deepgram_apikey": "k"}), \
         patch("videotrans.recognition._deepgram.DeepgramClient") as mock_cls:
        mock_cls.return_value.listen.rest.v.return_value.transcribe_file = fake_transcribe
        out = rec._exec()

    assert captured["language"] == expected_lang_opt
    assert captured["detect_language"] == expected_detect_opt
    assert len(out) == 1
    assert out[0]["text"] == expected_text


def test_deepgram_large_file_transcode_cleanup_and_none_cache(tmp_path, monkeypatch):
    from videotrans.recognition._deepgram import DeepgramRecogn

    audio = tmp_path / "big.wav"
    audio.write_bytes(b"original-wav")
    monkeypatch.setattr(config, "TEMP_DIR", str(tmp_path / "global_temp"))

    created_mp3s = []

    def fake_ffmpeg(cmd):
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"mp3-bytes")
        created_mp3s.append(out_path)
        return True

    rec = DeepgramRecogn(
        audio_file=str(audio),
        cache_folder=None,
        model_name="nova-3",
        detect_language=None,
    )

    with patch("videotrans.recognition._deepgram.params", {"deepgram_apikey": "k"}), \
         patch("videotrans.recognition._deepgram.os.path.getsize", return_value=60 * 1024 * 1024), \
         patch("videotrans.recognition._deepgram.runffmpeg", side_effect=fake_ffmpeg), \
         patch("videotrans.recognition._deepgram.DeepgramClient") as mock_cls:
        res = MagicMock()
        res.results.utterances = [SimpleNamespace(start=0.0, end=1.0, transcript="测试", speaker=0)]
        res.results.channels = []
        mock_cls.return_value.listen.rest.v.return_value.transcribe_file.return_value = res

        out = rec._exec()

    assert len(out) == 1
    assert rec.audio_file == str(audio)  # Never mutated
    assert len(created_mp3s) == 1
    assert created_mp3s[0].parent == Path(config.TEMP_DIR)
    assert not created_mp3s[0].exists()  # Cleaned up in finally


def test_deepgram_client_reuse_proxy_transport_and_retries(tmp_path, monkeypatch):
    from deepgram.clients.common.v1.errors import DeepgramApiError
    from videotrans.recognition import _retry
    from videotrans.recognition._deepgram import DeepgramRecogn

    monkeypatch.setattr(_retry.time, "sleep", lambda _s: None)
    monkeypatch.setitem(config.settings, "retry_nums", 2)

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")

    # 1. Transient 429 then 503 then success reuses single DeepgramClient and closes proxy transport
    rec = DeepgramRecogn(
        audio_file=str(audio),
        cache_folder=str(tmp_path),
        model_name="nova-3",
        detect_language="en",
    )
    rec.proxy_str = "http://127.0.0.1:8888"

    attempts = {"n": 0}

    def flaky_transcribe(payload, options, timeout=600, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _make_deepgram_api_error(DeepgramApiError, "rate limit", "429")
        if attempts["n"] == 2:
            raise _make_deepgram_api_error(DeepgramApiError, "unavailable", "503")
        res = MagicMock()
        res.results.utterances = [SimpleNamespace(start=0.0, end=1.0, transcript="ok", speaker=None)]
        res.results.channels = []
        return res

    fake_transport = MagicMock()
    with patch("videotrans.recognition._deepgram.params", {"deepgram_apikey": "k"}), \
         patch("videotrans.recognition._deepgram.httpx.HTTPTransport", return_value=fake_transport) as mock_tp, \
         patch("videotrans.recognition._deepgram.DeepgramClient") as mock_cls:
        mock_cls.return_value.listen.rest.v.return_value.transcribe_file = flaky_transcribe
        out = rec._exec()

    assert len(out) == 1
    assert attempts["n"] == 3
    assert mock_cls.call_count == 1
    mock_tp.assert_called_once_with(proxy="http://127.0.0.1:8888")
    fake_transport.close.assert_called_once()

    # 2. 401 fails fast without retry; no proxy -> no bare HTTPTransport()
    rec2 = DeepgramRecogn(
        audio_file=str(audio),
        cache_folder=str(tmp_path),
        model_name="nova-3",
        detect_language="en",
    )
    rec2.proxy_str = ""
    attempts["n"] = 0

    def unauthorized_transcribe(payload, options, timeout=600, **kwargs):
        assert "transport" not in kwargs
        attempts["n"] += 1
        raise _make_deepgram_api_error(DeepgramApiError, "unauthorized", "401")

    with patch("videotrans.recognition._deepgram.params", {"deepgram_apikey": "k"}), \
         patch("videotrans.recognition._deepgram.httpx.HTTPTransport") as mock_tp2, \
         patch("videotrans.recognition._deepgram.DeepgramClient") as mock_cls2:
        mock_cls2.return_value.listen.rest.v.return_value.transcribe_file = unauthorized_transcribe
        with pytest.raises(DeepgramApiError):
            rec2._exec()

    assert attempts["n"] == 1
    mock_tp2.assert_not_called()


# ============================================================================
# 3. RecognMixin asr_duration on speaker.json early return
# ============================================================================

def test_recogn_records_asr_duration_on_speaker_json_early_return(tmp_path, monkeypatch):
    from videotrans.task import _stage_recogn

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    target_dir = tmp_path / "out"
    target_dir.mkdir()
    source_wav = cache_dir / "source.wav"
    source_wav.write_bytes(b"wav")
    (cache_dir / "speaker.json").write_text('["[spk0]"]', encoding="utf-8")

    class DummyTask(_stage_recogn.RecognMixin):
        def __init__(self):
            self.uuid = "u1"
            self.precent = 0
            self.should_recogn = True
            self.should_trans = False
            self.max_speakers = 2
            self.event_sink = None
            self.asr_duration = None
            self.source_srt_list = []
            self.cfg = SimpleNamespace(
                subtitle_source="audio_asr",
                source_sub=str(target_dir / "source.srt"),
                force_recogn=False,
                source_wav=str(source_wav),
                remove_noise=False,
                recogn_type=1,
                model_name="nova-3",
                detect_language="en",
                cache_folder=str(cache_dir),
                target_dir=str(target_dir),
                noextname="test",
                app_mode="biaozhun",
                is_cuda=False,
                subtitle_type=0,
                rephrase=1,
                fix_punc=0,
                enable_diariz=True,
                basename="test.mp4",
            )

        def _exit(self):
            return False

        def signal(self, **_kwargs):
            pass

        def _save_srt_target(self, subs, path):
            Path(path).write_text(get_srt_from_list(subs), encoding="utf-8")

    task = DummyTask()
    monkeypatch.setattr(
        _stage_recogn,
        "run_recogn",
        lambda **_kw: [_make_srt("Hello world", 0, 1500, 1)],
    )

    task.recogn()
    assert isinstance(task.asr_duration, float)
    assert task.asr_duration >= 0.0


# ============================================================================
# 4. BaseRecogn._merge_sub & BaseRecogn.cut_audio
# ============================================================================

def test_merge_sub_handles_empty_items_and_syncs_timestamps():
    from videotrans.recognition._base import BaseRecogn

    rec = BaseRecogn(detect_language="en", recogn_type=0)

    # Leading blank item before short item must not IndexError in _phase1_merge_short
    subs1 = [
        _make_srt("   ", 0, 500, 1),
        _make_srt("Hi", 500, 700, 2),
        _make_srt("Second line long enough", 800, 3000, 3),
    ]
    out1 = rec._merge_sub(subs1)
    assert len(out1) >= 1
    assert [it["line"] for it in out1] == list(range(1, len(out1) + 1))

    # Punctuation-only item ",,," between neighbors must not IndexError in _phase4_redistribute_by_punct
    subs2 = [
        _make_srt("First sentence without punct", 0, 2000, 1),
        _make_srt(",,,", 2000, 4000, 2),
        _make_srt("ok, next sentence here", 4000, 7000, 3),
        _make_srt("Fourth sentence", 7500, 10000, 4),
    ]
    out2 = rec._merge_sub(subs2)
    assert all(it["text"].strip() for it in out2)
    assert [it["line"] for it in out2] == list(range(1, len(out2) + 1))

    # _phase2_merge_first and _phase3_merge_last must sync startraw/endraw/time
    p2 = rec._phase2_merge_first(
        [_make_srt("Hi", 100, 300, 1), _make_srt("World today is nice", 400, 2500, 2)],
        min_speech=1000,
    )
    assert len(p2) == 1
    assert p2[0]["start_time"] == 100
    assert p2[0]["startraw"] == ms_to_time_string(ms=100)
    assert p2[0]["endraw"] == ms_to_time_string(ms=2500)
    assert p2[0]["time"] == f"{p2[0]['startraw']} --> {p2[0]['endraw']}"

    p3 = rec._phase3_merge_last(
        [_make_srt("World today is nice", 400, 2500, 1), _make_srt("Yo", 2600, 2900, 2)],
        min_speech=1000,
    )
    assert len(p3) == 1
    assert p3[0]["end_time"] == 2900
    assert p3[0]["endraw"] == ms_to_time_string(ms=2900)
    assert p3[0]["time"] == f"{p3[0]['startraw']} --> {p3[0]['endraw']}"
    assert "00:00:02,900" in get_srt_from_list(p3)


def test_cut_audio_uses_cache_folder_no_pydub_and_overlong_cut(tmp_path, monkeypatch):
    from videotrans.recognition._base import BaseRecogn

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global_tmp = tmp_path / "global_tmp"
    global_tmp.mkdir()
    monkeypatch.setattr(config, "TEMP_DIR", str(global_tmp))

    sr = 16000
    # 34s signal: loud everywhere except quiet dip around 28s
    samples = np.full(sr * 34, 4000, dtype=np.int16)
    samples[sr * 27 : sr * 29] = 10
    wav_path = _write_pcm_wav(tmp_path / "long.wav", samples, sr=sr)

    rec = BaseRecogn(audio_file=str(wav_path), cache_folder=str(cache_dir), detect_language="en")
    rec.speech_timestamps = [[0, 2000], [2000, 34000]]

    # Poison pydub.AudioSegment if it were imported inside cut_audio
    monkeypatch.setitem(sys.modules, "pydub", None)

    clips = rec.cut_audio()
    assert len(clips) == 3
    clip_dir = Path(clips[0]["filename"]).parent
    assert clip_dir.parent == cache_dir
    assert clip_dir.name.startswith("clip_")
    assert list(global_tmp.glob("clip_*")) == []

    # Verify 500ms head+tail padding on first clip (2000ms + 1000ms = 3000ms)
    info = sf.info(clips[0]["filename"])
    assert info.frames == sr * 3
    # Verify overlong segment cut near the 27-29s quiet dip
    assert 26000 <= clips[1]["end_time"] <= 29500
    assert clips[2]["start_time"] == clips[1]["end_time"]


# ============================================================================
# 5. PrepareMixin._split_audio_byraw
# ============================================================================

def test_split_audio_byraw_dedup_and_fallbacks(tmp_path, monkeypatch):
    from videotrans.task import _stage_prepare

    def _make_prep(idx: int):
        cdir = tmp_path / f"cache_{idx}"
        tdir = tmp_path / f"target_{idx}"
        cdir.mkdir()
        tdir.mkdir()
        src = tmp_path / f"video_{idx}.mp4"
        src.write_bytes(b"mp4")

        class Prep(_stage_prepare.PrepareMixin):
            def __init__(self):
                self.uuid = f"u_{idx}"
                self.cfg = SimpleNamespace(
                    name=str(src),
                    cache_folder=str(cdir),
                    target_dir=str(tdir),
                    source_wav=str(cdir / "source.wav"),
                    vocal=str(cdir / "vocal.wav"),
                    instrument=str(cdir / "instrument.wav"),
                )

            def _process_callback(self, _data):
                pass

            def _new_process(self, callback=None, title="", is_cuda=False, kwargs=None):
                return True

        return Prep()

    # Case A: is_separate=False -> single ffmpeg extraction
    calls = []

    def fake_ffmpeg(cmd):
        calls.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"wav-data")
        return True

    monkeypatch.setattr(_stage_prepare, "runffmpeg", fake_ffmpeg)
    p1 = _make_prep(1)
    assert p1._split_audio_byraw(False) is True
    assert len(calls) == 1
    assert p1.cfg.name in calls[0]

    # Case B: is_separate=True success -> name decoded ONCE, source_wav derived from vocal, tmpfile cleaned
    calls.clear()
    p2 = _make_prep(2)

    def fake_proc_ok(callback=None, title="", is_cuda=False, kwargs=None):
        Path(kwargs["vocal_file"]).write_bytes(b"vocal")
        Path(kwargs["instr_file"]).write_bytes(b"instr")
        return True

    p2._new_process = fake_proc_ok
    monkeypatch.setitem(config.settings, "uvr_models", "UVR-MDX-NET-Voc_FT")
    with patch("videotrans.util.help_down.down_file_from_hf"):
        assert p2._split_audio_byraw(True) is True

    name_decodes = [c for c in calls if p2.cfg.name in c]
    assert len(name_decodes) == 1
    assert any(p2.cfg.vocal in c and "volume=1.5" in c for c in calls)
    assert not (Path(p2.cfg.cache_folder) / "441000_ac2_raw.wav").exists()

    # Case C: is_separate=True failure -> name still decoded ONCE, fallback derives from tmpfile, tmpfile cleaned
    calls.clear()
    p3 = _make_prep(3)
    p3._new_process = lambda **_kw: False
    with patch("videotrans.util.help_down.down_file_from_hf"):
        assert p3._split_audio_byraw(True) is False

    name_decodes = [c for c in calls if p3.cfg.name in c]
    assert len(name_decodes) == 1
    tmp_path_str = f"{p3.cfg.cache_folder}/441000_ac2_raw.wav"
    assert any(tmp_path_str in c and p3.cfg.source_wav in c for c in calls)
    assert Path(p3.cfg.source_wav).is_file()
    assert not Path(tmp_path_str).exists()

    # Case D: vocal & instrument pre-exist without source_wav -> 0 decodes of cfg.name, source_wav derived from vocal
    calls.clear()
    p4 = _make_prep(4)
    Path(p4.cfg.vocal).write_bytes(b"vocal")
    Path(p4.cfg.instrument).write_bytes(b"instr")
    assert p4._split_audio_byraw(True) is True
    assert len([c for c in calls if p4.cfg.name in c]) == 0
    assert Path(p4.cfg.source_wav).is_file()
    assert not (Path(p4.cfg.cache_folder) / "441000_ac2_raw.wav").exists()


# ============================================================================
# 6. FasterAll & GlobalProcessManager
# ============================================================================

def test_faster_all_header_duration_and_srt_cleanup(tmp_path, monkeypatch):
    from videotrans.recognition import _whisper

    pcm_wav = _write_pcm_wav(tmp_path / "pcm.wav", np.zeros(16000 * 3, dtype=np.int16), sr=16000)
    flt_wav = tmp_path / "flt.wav"
    sf.write(str(flt_wav), np.zeros(16000 * 2, dtype=np.float32), 16000, subtype="FLOAT")

    assert _whisper.get_audio_duration_ms(pcm_wav) == 3000
    assert _whisper.get_audio_duration_ms(flt_wav) == 2000
    assert _whisper.get_audio_duration_ms(None) == 0
    assert _whisper.get_audio_duration_ms(tmp_path / "missing.wav") == 0

    # Poison pydub so any AudioSegment.from_wav call would crash
    monkeypatch.setitem(sys.modules, "pydub", None)

    rec = _whisper.FasterAll(
        audio_file=str(pcm_wav),
        cache_folder=str(tmp_path),
        model_name="large-v3",
        detect_language="en",
    )
    assert rec.audio_duration == 3000

    # 1. Normal completion unlinks subtitle_srt
    created_srts = []

    def fake_new_process_ok(callback=None, title="", is_cuda=False, kwargs=None):
        srt_p = Path(kwargs["subtitle_srt"])
        created_srts.append(srt_p)
        srt_p.parent.mkdir(parents=True, exist_ok=True)
        srt_p.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
        return [_make_srt("Hi", 0, 1000, 1)]

    rec._new_process = fake_new_process_ok
    out = rec._faster()
    assert len(out) == 1
    assert not created_srts[-1].exists()

    # 2. SttTimeoutError recovers SRT AND unlinks subtitle_srt
    def fake_new_process_timeout(callback=None, title="", is_cuda=False, kwargs=None):
        srt_p = Path(kwargs["subtitle_srt"])
        created_srts.append(srt_p)
        srt_p.parent.mkdir(parents=True, exist_ok=True)
        srt_p.write_text("1\n00:00:00,000 --> 00:00:01,500\nRecovered\n", encoding="utf-8")
        raise SttTimeoutError("timeout")

    rec._new_process = fake_new_process_timeout
    out2 = rec._faster()
    assert len(out2) == 1
    assert out2[0]["text"] == "Recovered"
    assert not created_srts[-1].exists()

    # 3. RuntimeError still unlinks subtitle_srt
    def fake_new_process_err(callback=None, title="", is_cuda=False, kwargs=None):
        srt_p = Path(kwargs["subtitle_srt"])
        created_srts.append(srt_p)
        srt_p.parent.mkdir(parents=True, exist_ok=True)
        srt_p.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    rec._new_process = fake_new_process_err
    with pytest.raises(RuntimeError):
        rec._faster()
    assert not created_srts[-1].exists()


def test_async_result_future_wrapper_event_wait_and_cpu_guard(monkeypatch):
    from videotrans.process.process_manager import (
        AsyncResultFutureWrapper,
        GlobalProcessManager,
    )

    ev = threading.Event()

    class FakeAsyncResult:
        def ready(self):
            return ev.is_set()

        def wait(self, timeout=None):
            return ev.wait(timeout=timeout)

        def get(self, timeout=None):
            return ("done", None)

    fake_pool = SimpleNamespace(_pool=[SimpleNamespace(is_alive=lambda: True)])
    wrapper = AsyncResultFutureWrapper(FakeAsyncResult(), fake_pool)

    threading.Timer(0.03, ev.set).start()
    t0 = time.perf_counter()
    res = wrapper.result(timeout=2.0)
    elapsed = time.perf_counter() - t0
    assert res == ("done", None)
    assert elapsed < 0.15  # Wakes on event rather than sleeping 0.5s

    # os.cpu_count() returning None must not raise TypeError
    monkeypatch.setattr("os.cpu_count", lambda: None)
    assert GlobalProcessManager.get_cpu_process_nums() >= 1
    assert GlobalProcessManager.get_gpu_process_nums() >= 1


# ============================================================================
# 7. Cloud STT: ElevenLabs, Gemini, Google
# ============================================================================

def test_elevenlabs_client_closed_timeout_auto_scribe_v2_and_retry(tmp_path, monkeypatch):
    from elevenlabs.core import ApiError as ElevenLabsApiError
    from videotrans.recognition import _elevenlabs, _retry

    monkeypatch.setattr(_retry.time, "sleep", lambda _s: None)
    monkeypatch.setitem(config.settings, "retry_nums", 2)

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"wav")

    rec = _elevenlabs.ElevenLabsRecogn(
        audio_file=str(audio),
        detect_language="auto",
        model_name="scribe_v2",
    )

    closed = {"count": 0}
    captured_convert = []
    attempts = {"n": 0}

    class FakeClient:
        def __init__(self, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed["count"] += 1

    class FakeElevenLabs:
        def __init__(self, **_kw):
            self.speech_to_text = SimpleNamespace(convert=self._convert)

        def _convert(self, **kwargs):
            captured_convert.append(kwargs)
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _make_elevenlabs_api_error(ElevenLabsApiError, status_code=429, body={"detail": "rate limit"})
            return SimpleNamespace(
                words=[SimpleNamespace(type="word", text="Hello.", start=0.0, end=1.0, speaker_id=None)]
            )

    with patch("videotrans.recognition._elevenlabs.params", {"elevenlabstts_key": "k"}), \
         patch("videotrans.recognition._elevenlabs.httpx.Client", side_effect=FakeClient) as mock_http, \
         patch("videotrans.recognition._elevenlabs.ElevenLabs", side_effect=FakeElevenLabs):
        out = rec._exec()

    assert len(out) == 1
    assert closed["count"] == 1
    assert mock_http.call_args.kwargs["timeout"] is not None
    assert captured_convert[0]["model_id"] == "scribe_v2"
    assert "language_code" not in captured_convert[0]

    # 401 fails fast and still closes httpx.Client
    attempts["n"] = 0

    class FakeElevenLabs401:
        def __init__(self, **_kw):
            self.speech_to_text = SimpleNamespace(convert=self._fail)

        def _fail(self, **_kw):
            attempts["n"] += 1
            raise _make_elevenlabs_api_error(ElevenLabsApiError, status_code=401, body={"detail": "bad key"})

    with patch("videotrans.recognition._elevenlabs.params", {"elevenlabstts_key": "k"}), \
         patch("videotrans.recognition._elevenlabs.httpx.Client", side_effect=FakeClient), \
         patch("videotrans.recognition._elevenlabs.ElevenLabs", side_effect=FakeElevenLabs401):
        with pytest.raises(ElevenLabsApiError):
            rec._exec()

    assert attempts["n"] == 1
    assert closed["count"] == 2


def test_gemini_retries_429_500_503_rotates_keys_and_cleans_clips(tmp_path, monkeypatch):
    from google.genai import errors
    from videotrans.recognition import _gemini, _retry

    monkeypatch.setattr(_retry.time, "sleep", lambda _s: None)
    monkeypatch.setitem(config.settings, "retry_nums", 3)

    wav = _write_pcm_wav(tmp_path / "g.wav", np.zeros(16000, dtype=np.int16), sr=16000)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with patch("videotrans.recognition._gemini.params", {"gemini_key": "k1, k2"}):
        rec = _gemini.GeminiRecogn(audio_file=str(wav), cache_folder=str(cache_dir), detect_language="en")
    rec.speech_timestamps = [[0, 1000]]

    seen_keys = []
    seen_timeouts = []
    attempts = {"n": 0}

    class FakeGenaiClient:
        def __init__(self, api_key=None, http_options=None):
            seen_keys.append(api_key)
            seen_timeouts.append(getattr(http_options, "timeout", None))
            self.models = SimpleNamespace(generate_content_stream=self._stream)

        def _stream(self, **_kw):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise errors.APIError(429, {"error": {"message": "quota"}})
            if attempts["n"] == 2:
                raise errors.APIError(500, {"error": {"message": "internal"}})
            if attempts["n"] == 3:
                raise errors.APIError(503, {"error": {"message": "overloaded"}})
            return [SimpleNamespace(text="<audio_text>Hello Gemini</audio_text>")]

    with patch("videotrans.recognition._gemini.genai.Client", side_effect=FakeGenaiClient):
        out = rec._exec()

    assert len(out) == 1
    assert out[0]["text"] == "Hello Gemini"
    assert attempts["n"] == 4
    assert seen_timeouts == [180_000, 180_000]
    assert set(seen_keys) == {"k1", "k2"}
    assert list(cache_dir.glob("clip_*")) == []  # Clips cleaned up

    # 400 fails fast via StopRetry and still cleans up clips
    attempts["n"] = 0

    class FakeGenaiClient400:
        def __init__(self, **_kw):
            self.models = SimpleNamespace(generate_content_stream=self._fail)

        def _fail(self, **_kw):
            attempts["n"] += 1
            raise errors.APIError(400, {"error": {"message": "bad request"}})

    with patch("videotrans.recognition._gemini.genai.Client", side_effect=FakeGenaiClient400):
        with pytest.raises(StopRetry):
            rec._exec()

    assert attempts["n"] == 1
    assert list(cache_dir.glob("clip_*")) == []


def test_google_per_chunk_retry_no_duplicates_and_cleanup(tmp_path, monkeypatch):
    import speech_recognition as sr
    from videotrans.recognition import _google, _retry

    monkeypatch.setattr(_retry.time, "sleep", lambda _s: None)
    monkeypatch.setitem(config.settings, "retry_nums", 2)

    global_tmp = tmp_path / "global_tmp"
    global_tmp.mkdir()
    monkeypatch.setattr(config, "TEMP_DIR", str(global_tmp))

    wav = _write_pcm_wav(tmp_path / "google.wav", np.zeros(16000 * 3, dtype=np.int16), sr=16000)

    rec = _google.GoogleRecogn(audio_file=str(wav), cache_folder=None, detect_language=None)

    chunk_calls = {0: 0, 1: 0, 2: 0}
    seen_languages = []
    seen_timeouts = []

    class FakeRecognizer:
        def __init__(self):
            self.operation_timeout = None
            self._idx = 0

        def record(self, _source):
            idx = self._idx
            self._idx += 1
            return idx

        def recognize_google(self, audio_data, language=None):
            seen_languages.append(language)
            seen_timeouts.append(self.operation_timeout)
            chunk_calls[audio_data] += 1
            if audio_data == 1 and chunk_calls[1] < 3:
                raise sr.RequestError("transient 503")
            return f"chunk {audio_data}"

    class FakeAudioFile:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(rec, "_shorten_voice_old", lambda _s: [(0, 1000, False), (1000, 2000, False), (2000, 3000, False)])
    with patch("videotrans.recognition._google.sr.Recognizer", side_effect=FakeRecognizer), \
         patch("videotrans.recognition._google.sr.AudioFile", side_effect=FakeAudioFile):
        out = rec._exec()

    assert len(out) == 3
    assert [it["text"] for it in out] == ["Chunk 0.", "Chunk 1.", "Chunk 2."]
    assert chunk_calls == {0: 1, 1: 3, 2: 1}  # No duplicate chunk 0 re-runs
    assert all(lang == "en-US" for lang in seen_languages)
    assert all(t == 60 for t in seen_timeouts)
    assert list(global_tmp.iterdir()) == []  # Temp dir cleaned up when cache_folder=None


# ============================================================================
# 8. Ten-VAD & Qwen-ASR
# ============================================================================

def test_vad_single_pass_vectorized_energy_and_mock_safety(tmp_path, monkeypatch):
    from videotrans.process import vad

    sr = 16000
    rng = np.random.default_rng(42)

    def _speech_block(dur_s: float) -> np.ndarray:
        n = int(sr * dur_s)
        t = np.arange(n) / sr
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
        sig = (
            np.sin(2 * np.pi * 180 * t)
            + 0.6 * np.sin(2 * np.pi * 540 * t)
            + 0.3 * np.sin(2 * np.pi * 1100 * t)
        ) * env + rng.normal(0, 0.08, n)
        return np.clip(sig * 8000, -32000, 32000).astype(np.int16)

    short_speech = _speech_block(1.5)
    silence = np.zeros(sr, dtype=np.int16)
    long_speech = _speech_block(10.0)
    long_speech[sr * 4 : sr * 4 + 256 * 5] = 5  # low-energy dip for energy cut
    audio = np.concatenate([short_speech, silence, long_speech, silence, short_speech])
    wav_path = _write_pcm_wav(tmp_path / "vad.wav", audio, sr=sr)

    instantiations = {"count": 0}
    process_calls = {"count": 0}
    orig_tenvad = vad.TenVad

    class SpyTenVad:
        def __init__(self, hop_size, threshold):
            instantiations["count"] += 1
            self._inner = orig_tenvad(hop_size, threshold)

        def process(self, frame):
            process_calls["count"] += 1
            return self._inner.process(frame)

    monkeypatch.setattr(vad, "TenVad", SpyTenVad)
    segs, err = vad.get_speech_timestamp(
        input_wav=str(wav_path),
        threshold=None,  # Exercises None threshold guard
        min_speech_duration_ms=500,
        max_speech_duration_ms=6000,
        min_silent_duration_ms=200,
    )
    assert err is None
    assert segs and len(segs) >= 3
    expected_frames = len(audio) // 256
    assert instantiations["count"] == 1
    assert process_calls["count"] == expected_frames
    assert all((e - s) <= 6050 for s, e in segs)

    # Mock TenVad with inconsistent (prob=0.0, flag=1) falls back cleanly without failing
    class DummyMockVad:
        def __init__(self, _hop, _thresh):
            pass

        def process(self, _frame):
            return 0.0, 1

    monkeypatch.setattr(vad, "TenVad", DummyMockVad)
    segs_mock, err_mock = vad.get_speech_timestamp(
        input_wav=str(wav_path),
        threshold=0.5,
        min_speech_duration_ms=500,
        max_speech_duration_ms=6000,
        min_silent_duration_ms=200,
    )
    assert err_mock is None
    assert segs_mock and len(segs_mock) >= 2

    # Short audio < 256 samples returns (False, None)
    short_wav = _write_pcm_wav(tmp_path / "short.wav", np.zeros(100, dtype=np.int16), sr=sr)
    assert vad.get_speech_timestamp(input_wav=str(short_wav), threshold=0.5) == (False, None)


def test_qwenasrlocal_forwards_detect_language_and_cleans_up(tmp_path, monkeypatch):
    from videotrans.recognition._qwenasrlocal import QwenasrlocalRecogn

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    global_tmp = tmp_path / "global_tmp"
    global_tmp.mkdir()
    monkeypatch.setattr(config, "TEMP_DIR", str(global_tmp))

    wav = _write_pcm_wav(tmp_path / "qwen.wav", np.zeros(16000, dtype=np.int16), sr=16000)
    rec = QwenasrlocalRecogn(
        audio_file=str(wav),
        cache_folder=str(cache_dir),
        model_name="1.7B",
        detect_language="zh-cn",
    )
    rec.uuid = "qwen-u1"
    rec.speech_timestamps = [[0, 1000]]

    captured = {}

    def fake_new_process(callback=None, title="", is_cuda=False, kwargs=None):
        captured["kwargs"] = kwargs
        return [_make_srt("你好", 0, 1000, 1)]

    rec._new_process = fake_new_process
    out = rec._exec()
    assert len(out) == 1
    assert captured["kwargs"]["detect_language"] == "zh-cn"
    assert list(cache_dir.glob("clip_*")) == []
    assert list((global_tmp / "qwen-u1").glob("cut_audio_list_*.json")) == []


@pytest.mark.parametrize(
    "detect_language,expected_qwen_lang",
    [
        ("zh-cn", "Chinese"),
        ("zh-tw", "Chinese"),
        ("zh", "Chinese"),
        ("en", "English"),
        ("en-US", "English"),
        ("fa", "Persian"),
        ("yue", "Cantonese"),
        ("da", "Danish"),
        ("fi", "Finnish"),
        ("mk", "Macedonian"),
        ("ja", "Japanese"),
        ("auto", None),
        (None, None),
        ("", None),
        ("nb", None),
        ("uk", None),
        ("ur", None),
        ("xx-invalid", None),
    ],
)
def test_stt_qwen_language_resolution_and_transcribe_forwarding(
    tmp_path, monkeypatch, detect_language, expected_qwen_lang
):
    qwen_utils = _load_qwen_asr_utils()
    normalize_language_name = qwen_utils.normalize_language_name
    validate_language = qwen_utils.validate_language
    from videotrans.process import stt_qwen

    resolved = stt_qwen._resolve_qwen_asr_language(detect_language)
    assert resolved == expected_qwen_lang
    if resolved is not None:
        validate_language(normalize_language_name(resolved))

    cut_list_path = tmp_path / "cut.json"
    cut_list_path.write_text(
        json.dumps([dict(asdict(_make_srt("", 0, 1000, 1)), filename="seg0.wav")]),
        encoding="utf-8",
    )
    log_path = tmp_path / "qwen.log"
    captured_languages = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kw):
            return cls()

        def transcribe(self, audio=None, language=None, return_time_stamps=False):
            captured_languages.append(language)
            for lang in language:
                if lang is not None:
                    validate_language(normalize_language_name(lang))
            return [SimpleNamespace(text="transcribed") for _ in audio]

    monkeypatch.setattr(stt_qwen, "_load_qwen3_asr_model", lambda: FakeModel)

    res, err = stt_qwen.qwen3asr_fun(
        cut_audio_list=str(cut_list_path),
        logs_file=str(log_path),
        is_cuda=False,
        model_name="1.7B",
        detect_language=detect_language,
    )
    assert err is None
    assert len(res) == 1
    assert res[0]["text"] == "transcribed"
    assert captured_languages == [[expected_qwen_lang]]


def test_split_audio_byraw_vocal_to_source_wav_failure_falls_back_to_raw(tmp_path, monkeypatch):
    from videotrans.task import _stage_prepare

    cdir = tmp_path / "cache"
    tdir = tmp_path / "target"
    cdir.mkdir()
    tdir.mkdir()
    src = tmp_path / "video.mp4"
    src.write_bytes(b"mp4")

    class Prep(_stage_prepare.PrepareMixin):
        def __init__(self):
            self.uuid = "u_fallback"
            self.cfg = SimpleNamespace(
                name=str(src),
                cache_folder=str(cdir),
                target_dir=str(tdir),
                source_wav=str(cdir / "source.wav"),
                vocal=str(cdir / "vocal.wav"),
                instrument=str(cdir / "instrument.wav"),
            )

        def _process_callback(self, _data):
            pass

        def _new_process(self, callback=None, title="", is_cuda=False, kwargs=None):
            Path(kwargs["vocal_file"]).write_bytes(b"vocal")
            Path(kwargs["instr_file"]).write_bytes(b"instr")
            return True

    calls = []

    def fake_ffmpeg(cmd):
        calls.append(list(cmd))
        # Simulate failure when deriving source_wav from vocal.wav
        if str(cdir / "vocal.wav") in cmd:
            return False
        Path(cmd[-1]).write_bytes(b"wav-data")
        return True

    monkeypatch.setattr(_stage_prepare, "runffmpeg", fake_ffmpeg)
    monkeypatch.setitem(config.settings, "uvr_models", "UVR-MDX-NET-Voc_FT")
    prep = Prep()
    with patch("videotrans.util.help_down.down_file_from_hf"):
        ok = prep._split_audio_byraw(True)

    assert ok is False
    tmp_raw = cdir / "441000_ac2_raw.wav"
    assert not tmp_raw.exists()
    assert Path(prep.cfg.source_wav).is_file()
    tmp_path_str = f"{prep.cfg.cache_folder}/441000_ac2_raw.wav"
    assert any(tmp_path_str in c and prep.cfg.source_wav in c for c in calls)


def test_vad_float32_stereo_and_int32_scaling(tmp_path):
    import scipy.io.wavfile as wavfile
    from videotrans.process import vad

    sr = 16000
    t = np.arange(sr * 3, dtype=np.float32) / sr
    # Speech-like harmonic signal in [-0.5, 0.5]
    wave_flt = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    stereo_flt = np.column_stack([wave_flt, wave_flt]).astype(np.float32)
    stereo_i16 = np.round(stereo_flt * 32767.0).astype(np.int16)
    stereo_i32 = (stereo_i16.astype(np.int32) * 65536)

    flt_wav = tmp_path / "stereo_flt.wav"
    i16_wav = tmp_path / "stereo_i16.wav"
    i32_wav = tmp_path / "stereo_i32.wav"
    wavfile.write(str(flt_wav), sr, stereo_flt)
    wavfile.write(str(i16_wav), sr, stereo_i16)
    wavfile.write(str(i32_wav), sr, stereo_i32)

    frames_flt, pcm_flt = vad._prepare_pcm_frames(stereo_flt)
    frames_i16, pcm_i16 = vad._prepare_pcm_frames(stereo_i16)
    frames_i32, pcm_i32 = vad._prepare_pcm_frames(stereo_i32)
    assert frames_flt.shape == frames_i16.shape == frames_i32.shape
    assert np.max(np.abs(pcm_flt.astype(np.int32) - pcm_i16.astype(np.int32))) <= 1
    assert np.max(np.abs(pcm_i32.astype(np.int32) - pcm_i16.astype(np.int32))) <= 1

    segs_flt, err_flt = vad.get_speech_timestamp(input_wav=str(flt_wav), threshold=0.5)
    segs_i16, err_i16 = vad.get_speech_timestamp(input_wav=str(i16_wav), threshold=0.5)
    segs_i32, err_i32 = vad.get_speech_timestamp(input_wav=str(i32_wav), threshold=0.5)
    assert err_flt is None and err_i16 is None and err_i32 is None
    assert segs_flt and segs_flt == segs_i16 == segs_i32

    silent_flt_wav = tmp_path / "silent_flt.wav"
    wavfile.write(str(silent_flt_wav), sr, np.zeros((sr, 2), dtype=np.float32))
    assert vad.get_speech_timestamp(input_wav=str(silent_flt_wav), threshold=0.5) == (False, None)


def test_vad_max_speech_duration_invariant_and_balanced_cut(tmp_path, monkeypatch):
    from videotrans.process import vad

    sr = 16000
    rng = np.random.default_rng(7)
    frame_ms = (256 / sr) * 1000.0

    def _speech_block(sec: float) -> np.ndarray:
        n = int(sr * sec)
        t = np.arange(n) / sr
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
        sig = (
            0.5 * np.sin(2 * np.pi * 220 * t)
            + 0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.2 * np.sin(2 * np.pi * 880 * t)
        ) * env + rng.normal(0, 0.05, n)
        return np.clip(sig * 12000, -32000, 32000).astype(np.int16)

    # Case 1: 6.5s continuous speech with max=6000, min=1000
    wav_65 = _write_pcm_wav(tmp_path / "cont65.wav", _speech_block(6.5), sr=sr)
    segs_65, err_65 = vad.get_speech_timestamp(
        input_wav=str(wav_65),
        threshold=0.5,
        min_speech_duration_ms=1000,
        max_speech_duration_ms=6000,
        min_silent_duration_ms=200,
    )
    assert err_65 is None and segs_65
    assert all((e - s) <= 6000 + frame_ms for s, e in segs_65), segs_65

    # Case 2: 2.95s continuous speech with recogn2pass bounds max=2000, min=1000
    wav_295 = _write_pcm_wav(tmp_path / "cont295.wav", _speech_block(2.95), sr=sr)
    segs_295, err_295 = vad.get_speech_timestamp(
        input_wav=str(wav_295),
        threshold=0.5,
        min_speech_duration_ms=1000,
        max_speech_duration_ms=2000,
        min_silent_duration_ms=200,
    )
    assert err_295 is None and segs_295
    assert all((e - s) <= 2000 + frame_ms for s, e in segs_295), segs_295

    # Case 3: 60s mixed speech + silence with max=6000, min=1000
    blocks = []
    for i in range(6):
        blocks.append(_speech_block(8.5 if i % 2 == 0 else 7.2))
        blocks.append(np.zeros(int(sr * 1.5), dtype=np.int16))
    wav_60 = _write_pcm_wav(tmp_path / "mix60.wav", np.concatenate(blocks)[: sr * 60], sr=sr)
    segs_60, err_60 = vad.get_speech_timestamp(
        input_wav=str(wav_60),
        threshold=0.5,
        min_speech_duration_ms=1000,
        max_speech_duration_ms=6000,
        min_silent_duration_ms=200,
    )
    assert err_60 is None and segs_60
    assert all((e - s) <= 6000 + frame_ms for s, e in segs_60), segs_60

    # Case 4: Energy-cut fallback produces balanced halves >= min_speech_duration_ms when forced
    class AlwaysSpeechVad:
        def __init__(self, _hop, _thresh):
            pass

        def process(self, _frame):
            return 1.0, 1

    monkeypatch.setattr(vad, "TenVad", AlwaysSpeechVad)
    # Force energy-cut fallback by skipping _segments_from_flags multi-split once
    orig_seg_fn = vad._segments_from_flags

    def no_max_split_on_sub(flags_arg, min_sil_arg, max_speech_frames=None):
        if max_speech_frames is not None:
            return [[0, len(flags_arg)]]
        return orig_seg_fn(flags_arg, min_sil_arg, max_speech_frames=None)

    monkeypatch.setattr(vad, "_segments_from_flags", no_max_split_on_sub)
    segs_fb, err_fb = vad.get_speech_timestamp(
        input_wav=str(wav_65),
        threshold=0.5,
        min_speech_duration_ms=1000,
        max_speech_duration_ms=6000,
        min_silent_duration_ms=200,
    )
    assert err_fb is None and len(segs_fb) == 2
    assert all(950 <= (e - s) <= 6000 + frame_ms for s, e in segs_fb), segs_fb
