# -*- coding: utf-8 -*-
"""Opaque-box E2E tests for STT, AI Translation, TTS, Dubbing & CLI (F6, F7, F8, F9)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import uuid

import httpx
import numpy as np
import pytest
import soundfile as sf

from tests.e2e.conftest import make_mp4, make_wav


# ============================================================================
# F6: STT Correctness, Audio Prep & Temp Hygiene (Tier 1 & Tier 2)
# ============================================================================


def test_f6_prepare_mixin_extracts_16k_mono_wav_and_novoice_mp4(tmp_path: Path):
    from videotrans.configure import config as runtime_config
    from videotrans.task.taskcfg import TaskCfgVTT
    from videotrans.task.trans_create import TransCreate

    mp4 = make_mp4(tmp_path / "prep_demo.mp4", duration_s=1.5, with_audio=True, sr=44100)
    target_dir = tmp_path / "out_prep"
    cache_dir = Path(runtime_config.TEMP_DIR) / f"prep-{uuid.uuid4().hex[:8]}"
    cfg = TaskCfgVTT(
        name=mp4.as_posix(),
        target_dir=target_dir.as_posix(),
        cache_folder=cache_dir.as_posix(),
        source_language_code="en",
        target_language_code="vi",
        voice_role="No",
        only_out_dubbed_audio=False,
        is_separate=False,
        uuid=f"prep-{uuid.uuid4().hex[:8]}",
    )
    task = TransCreate(cfg=cfg, event_sink=lambda _e: None)
    try:
        task.prepare()
        assert Path(task.cfg.source_wav).is_file()
        info = sf.info(task.cfg.source_wav)
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.duration >= 1.0
    finally:
        task.set_end()


def test_f6_prepare_mixin_is_separate_single_raw_extract_and_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from videotrans.configure import config as runtime_config
    from videotrans.task import _stage_prepare
    from videotrans.task.taskcfg import TaskCfgVTT
    from videotrans.task.trans_create import TransCreate
    from videotrans.util import help_down

    monkeypatch.setattr(help_down, "down_file_from_hf", lambda *a, **kw: None)

    ffmpeg_calls: list[list[str]] = []
    real_runffmpeg = _stage_prepare.runffmpeg

    def spy_runffmpeg(cmd, *args, **kwargs):
        ffmpeg_calls.append(list(cmd))
        return real_runffmpeg(cmd, *args, **kwargs)

    monkeypatch.setattr(_stage_prepare, "runffmpeg", spy_runffmpeg)

    mp4 = make_mp4(tmp_path / "sep_demo.mp4", duration_s=1.5, with_audio=True, sr=44100)

    # Case A: vocal_bgm succeeds -> should NOT extract 16k source_wav directly from mp4 before vocal_bgm
    cfg_ok = TaskCfgVTT(
        name=mp4.as_posix(),
        target_dir=(tmp_path / "out_ok").as_posix(),
        cache_folder=(Path(runtime_config.TEMP_DIR) / f"sep-ok-{uuid.uuid4().hex[:6]}").as_posix(),
        source_language_code="en",
        target_language_code="vi",
        voice_role="No",
        only_out_dubbed_audio=True,
        is_separate=True,
        uuid=f"sep-ok-{uuid.uuid4().hex[:6]}",
    )
    task_ok = TransCreate(cfg=cfg_ok, event_sink=lambda _e: None)

    def fake_vocal_process_ok(callback=None, title="", is_cuda=False, kwargs=None):
        make_wav(kwargs["vocal_file"], duration_s=1.5, sr=44100, freq=440.0)
        make_wav(kwargs["instr_file"], duration_s=1.5, sr=44100, freq=220.0)
        return True

    monkeypatch.setattr(task_ok, "_new_process", fake_vocal_process_ok)
    try:
        task_ok.prepare()
        assert Path(task_ok.cfg.source_wav).is_file()
        assert sf.info(task_ok.cfg.source_wav).samplerate == 16000
        pre_vocal_16k = [
            c for c in ffmpeg_calls
            if task_ok.cfg.name in c and task_ok.cfg.source_wav in c and "16000" in c
        ]
        assert len(pre_vocal_16k) == 0, f"Redundant 16k extraction from raw video before vocal_bgm: {pre_vocal_16k}"
    finally:
        task_ok.set_end()

    # Case B: vocal_bgm fails -> fallback must still produce 16k source_wav
    ffmpeg_calls.clear()
    cfg_fail = TaskCfgVTT(
        name=mp4.as_posix(),
        target_dir=(tmp_path / "out_fail").as_posix(),
        cache_folder=(Path(runtime_config.TEMP_DIR) / f"sep-fail-{uuid.uuid4().hex[:6]}").as_posix(),
        source_language_code="en",
        target_language_code="vi",
        voice_role="No",
        only_out_dubbed_audio=True,
        is_separate=True,
        uuid=f"sep-fail-{uuid.uuid4().hex[:6]}",
    )
    task_fail = TransCreate(cfg=cfg_fail, event_sink=lambda _e: None)
    monkeypatch.setattr(task_fail, "_new_process", lambda **kw: False)
    try:
        task_fail.prepare()
        assert Path(task_fail.cfg.source_wav).is_file()
        assert sf.info(task_fail.cfg.source_wav).samplerate == 16000
    finally:
        task_fail.set_end()


def test_f6_ten_vad_and_cut_audio_keep_clips_inside_cache_folder(tmp_path: Path):
    from videotrans.configure import config as runtime_config
    from videotrans.process.vad import get_speech_timestamp
    from videotrans.recognition._base import BaseRecogn

    wav = make_wav(tmp_path / "speech_vad.wav", duration_s=3.0, sr=16000, speech_like=True, amplitude=0.5)
    segs, err = get_speech_timestamp(
        input_wav=str(wav),
        threshold=0.3,
        min_speech_duration_ms=500,
        max_speech_duration_ms=10000,
        min_silent_duration_ms=200,
    )
    assert err is None
    assert segs and len(segs) >= 1
    assert all(0 <= s < e <= 3100 for s, e in segs)

    cache_folder = Path(runtime_config.TEMP_DIR) / f"cut-{uuid.uuid4().hex[:8]}"
    cache_folder.mkdir(parents=True, exist_ok=True)
    rec = BaseRecogn(
        detect_language="en",
        audio_file=str(wav),
        cache_folder=cache_folder.as_posix(),
        uuid=f"cut-{uuid.uuid4().hex[:8]}",
    )
    rec.speech_timestamps = segs
    clips = rec.cut_audio()
    assert len(clips) >= 1
    for item in clips:
        clip_path = Path(item["filename"]).resolve()
        assert clip_path.is_file()
        assert cache_folder.resolve() in clip_path.parents, f"Clip leaked outside cache_folder: {clip_path}"

    leaked = list(Path(runtime_config.TEMP_DIR).glob("clip_*"))
    assert leaked == []


def test_f6_deepgram_recogn_handles_none_language_and_records_asr_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from videotrans import recognition
    from videotrans.configure import config as runtime_config
    from videotrans.recognition import _deepgram
    from videotrans.task.taskcfg import TaskCfgVTT
    from videotrans.task.trans_create import TransCreate

    class FakeUtterance:
        def __init__(self, start, end, transcript, speaker):
            self.start = start
            self.end = end
            self.transcript = transcript
            self.speaker = speaker

    class FakeResponse:
        def __init__(self):
            alt = SimpleNamespace(
                paragraphs=SimpleNamespace(
                    transcript="Hello world.",
                    paragraphs=[
                        SimpleNamespace(
                            start=0.0,
                            end=1.2,
                            speaker=0,
                            sentences=[SimpleNamespace(start=0.0, end=1.2, text="Hello world.")],
                        )
                    ],
                )
            )
            self.results = SimpleNamespace(
                utterances=[FakeUtterance(0.0, 1.2, "Hello world.", 0)],
                channels=[SimpleNamespace(alternatives=[alt])],
            )

    class FakeRestClient:
        def transcribe_file(self, payload, options, **kwargs):
            return FakeResponse()

    class FakeDeepgramClient:
        def __init__(self, api_key):
            self.listen = SimpleNamespace(rest=SimpleNamespace(v=lambda _v: FakeRestClient()))

    monkeypatch.setattr(_deepgram, "DeepgramClient", FakeDeepgramClient)
    runtime_config.params["deepgram_apikey"] = "dg-test-key"

    wav = make_wav(tmp_path / "dg_input.wav", duration_s=1.5, sr=16000, speech_like=True)
    cache_dir = Path(runtime_config.TEMP_DIR) / f"dg-{uuid.uuid4().hex[:8]}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    items = recognition.run(
        recogn_type=recognition.Deepgram,
        detect_language=None,
        audio_file=str(wav),
        cache_folder=cache_dir.as_posix(),
        model_name="nova-3",
        uuid=f"dg-{uuid.uuid4().hex[:8]}",
        max_speakers=0,
    )
    assert items and len(items) == 1
    assert "Hello" in items[0]["text"]
    assert (cache_dir / "speaker.json").is_file()
    assert (cache_dir / "transcript_options.json").is_file()

    # Verify RecognMixin.recogn() sets asr_duration even when speaker.json exists!
    mp4 = make_mp4(tmp_path / "dg_task.mp4", duration_s=1.5)
    t_cache = Path(runtime_config.TEMP_DIR) / f"dgtask-{uuid.uuid4().hex[:8]}"
    cfg = TaskCfgVTT(
        name=mp4.as_posix(),
        target_dir=(tmp_path / "dg_out").as_posix(),
        cache_folder=t_cache.as_posix(),
        source_language_code="en",
        target_language_code="vi",
        recogn_type=recognition.Deepgram,
        model_name="nova-3",
        voice_role="No",
        enable_diariz=True,
        only_out_dubbed_audio=True,
        uuid=f"dgtask-{uuid.uuid4().hex[:8]}",
    )
    task = TransCreate(cfg=cfg, event_sink=lambda _e: None)
    try:
        task.prepare()
        task.recogn()
        assert task.asr_duration is not None and task.asr_duration >= 0.0
    finally:
        task.set_end()


def test_f6_qwen_stt_forwards_detect_language_and_faster_cleans_srt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from videotrans.configure import config as runtime_config
    from videotrans.process import stt_qwen
    from videotrans.recognition._qwenasrlocal import QwenasrlocalRecogn
    from videotrans.recognition._whisper import FasterAll

    # 1. Qwen-ASR language forwarding
    captured_languages = []

    class FakeQwenModel:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def transcribe(self, *, audio, language=None, **kw):
            captured_languages.append(language)
            return [SimpleNamespace(text="Ni hao") for _ in audio]

    monkeypatch.setattr(stt_qwen, "_load_qwen3_asr_model", lambda: FakeQwenModel)

    wav = make_wav(tmp_path / "qwen.wav", duration_s=1.2, sr=16000, speech_like=True)
    q_cache = Path(runtime_config.TEMP_DIR) / f"qwen-{uuid.uuid4().hex[:8]}"
    q_cache.mkdir(parents=True, exist_ok=True)
    qrec = QwenasrlocalRecogn(
        detect_language="zh-cn",
        audio_file=str(wav),
        cache_folder=q_cache.as_posix(),
        model_name="1.7B",
        uuid=f"qwen-{uuid.uuid4().hex[:8]}",
    )
    qrec.speech_timestamps = [[0, 1000]]
    monkeypatch.setattr(qrec, "_download", lambda: None)
    monkeypatch.setattr(qrec, "_new_process", lambda callback, title="", is_cuda=False, kwargs=None: callback(**kwargs)[0])
    res_q = qrec.run()
    assert res_q and res_q[0]["text"] == "Ni hao"
    assert captured_languages and any(lang is not None for lang in captured_languages[0]), (
        f"Expected detect_language to be forwarded instead of all-None: {captured_languages}"
    )

    # 2. FasterAll cleans up TEMP_ROOT/faster-*.srt after normal completion
    f_cache = Path(runtime_config.TEMP_DIR) / f"faster-{uuid.uuid4().hex[:8]}"
    f_cache.mkdir(parents=True, exist_ok=True)
    frec = FasterAll(
        detect_language="en",
        audio_file=str(wav),
        cache_folder=f_cache.as_posix(),
        model_name="tiny",
        uuid=f"faster-{uuid.uuid4().hex[:8]}",
    )
    created_srt_paths: list[Path] = []

    def fake_faster_proc(callback, title="", is_cuda=False, kwargs=None):
        srt_p = Path(kwargs["subtitle_srt"])
        srt_p.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        created_srt_paths.append(srt_p)
        return [{"line": 1, "start_time": 0, "end_time": 1000, "startraw": "00:00:00,000", "endraw": "00:00:01,000", "time": "00:00:00,000 --> 00:00:01,000", "text": "Hello"}]

    monkeypatch.setattr(frec, "_download", lambda: None)
    monkeypatch.setattr(frec, "_new_process", fake_faster_proc)
    res_f = frec.run()
    assert res_f and res_f[0]["text"] == "Hello"
    assert created_srt_paths and not created_srt_paths[0].exists(), f"Leaked faster-*.srt: {created_srt_paths[0]}"


def test_f6_base_recogn_merge_sub_handles_empty_text_and_syncs_timestamps(tmp_path: Path):
    from videotrans.recognition._base import BaseRecogn

    wav = make_wav(tmp_path / "merge.wav", duration_s=1.0, sr=16000)
    rec = BaseRecogn(detect_language="en", audio_file=str(wav), cache_folder=str(tmp_path), uuid="merge-test")
    items = [
        {"line": 1, "start_time": 0, "end_time": 200, "startraw": "00:00:00,000", "endraw": "00:00:00,200", "time": "00:00:00,000 --> 00:00:00,200", "text": "Hi"},
        {"line": 2, "start_time": 200, "end_time": 1500, "startraw": "00:00:00,200", "endraw": "00:00:01,500", "time": "00:00:00,200 --> 00:00:01,500", "text": ""},
        {"line": 3, "start_time": 200, "end_time": 1800, "startraw": "00:00:00,200", "endraw": "00:00:01,800", "time": "00:00:00,200 --> 00:00:01,800", "text": "ok, really nice"},
        {"line": 4, "start_time": 1800, "end_time": 2000, "startraw": "00:00:01,800", "endraw": "00:00:02,000", "time": "00:00:01,800 --> 00:00:02,000", "text": "a"},
    ]
    merged = rec._merge_sub(items)
    assert merged
    for it in merged:
        assert it["time"] == f"{it['startraw']} --> {it['endraw']}"


def test_f6_ten_vad_silent_wav_returns_false_no_crash(tmp_path: Path):
    from videotrans.process.vad import get_speech_timestamp

    silent_wav = make_wav(tmp_path / "silence.wav", duration_s=1.5, sr=16000, silent=True)
    segs, err = get_speech_timestamp(
        input_wav=str(silent_wav),
        threshold=0.5,
        min_speech_duration_ms=500,
        max_speech_duration_ms=10000,
        min_silent_duration_ms=200,
    )
    assert not segs
    assert err is None


def test_f6_elevenlabs_recogn_closes_httpx_client_and_sets_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from videotrans.configure import config as runtime_config
    from videotrans.recognition import _elevenlabs

    created_clients = []
    real_client_cls = httpx.Client

    class SpyHttpxClient(real_client_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.was_closed = False
            self.init_kwargs = kwargs
            created_clients.append(self)

        def close(self):
            self.was_closed = True
            super().close()

    class FakeElevenLabs:
        def __init__(self, api_key="", httpx_client=None, **kwargs):
            self.httpx_client = httpx_client
            self.speech_to_text = SimpleNamespace(
                convert=lambda **kw: SimpleNamespace(
                    words=[SimpleNamespace(type="word", text="Hello.", speaker_id="speaker_0", start=0.0, end=0.8)]
                )
            )

    monkeypatch.setattr(_elevenlabs.httpx, "Client", SpyHttpxClient)
    monkeypatch.setattr(_elevenlabs, "ElevenLabs", FakeElevenLabs)
    runtime_config.params["elevenlabstts_key"] = "el-test-key"

    wav = make_wav(tmp_path / "el.wav", duration_s=1.0, sr=16000)
    rec = _elevenlabs.ElevenLabsRecogn(
        detect_language="en",
        audio_file=str(wav),
        cache_folder=str(tmp_path),
        uuid="el-test",
    )
    res = rec.run()
    assert res and res[0]["text"].rstrip(".") == "Hello"
    assert created_clients
    assert created_clients[0].is_closed is True
    assert created_clients[0].init_kwargs.get("timeout") is not None


def test_f6_gemini_recogn_retries_on_429_without_stop_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from google.genai import errors
    from videotrans.configure import config as runtime_config
    from videotrans.recognition import _gemini

    attempts = {"count": 0}

    class FakeGeminiClient:
        def __init__(self, **kwargs):
            self.models = self

        def generate_content_stream(self, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise errors.APIError(429, {"error": {"message": "rate limited"}})
            return [SimpleNamespace(text="<audio_text>Hello Gemini</audio_text>")]

    monkeypatch.setattr(_gemini.genai, "Client", FakeGeminiClient)
    runtime_config.params["gemini_key"] = "gem-key"
    monkeypatch.setattr(runtime_config.settings, "retry_nums", 3, raising=False)

    wav = make_wav(tmp_path / "gem_stt.wav", duration_s=1.0, sr=16000)
    rec = _gemini.GeminiRecogn(
        detect_language="en",
        audio_file=str(wav),
        cache_folder=str(tmp_path),
        uuid="gem-stt-retry",
    )
    rec.speech_timestamps = [[0, 900]]
    res = rec.run()
    assert attempts["count"] >= 2
    assert res and res[0]["text"] == "Hello Gemini"


def test_f6_faster_all_avoids_full_wav_pydub_decode_for_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from pydub import AudioSegment
    from videotrans.recognition._whisper import FasterAll

    wav = make_wav(tmp_path / "dur_probe.wav", duration_s=1.25, sr=16000)
    monkeypatch.setattr(
        AudioSegment,
        "from_wav",
        classmethod(lambda cls, *a, **kw: (_ for _ in ()).throw(AssertionError("Should not decode full WAV via AudioSegment.from_wav"))),
    )
    rec = FasterAll(
        detect_language="en",
        audio_file=str(wav),
        cache_folder=str(tmp_path),
        model_name="tiny",
        uuid="dur-probe",
    )
    assert rec.audio_duration == pytest.approx(1250, abs=20)


# ============================================================================
# F7: AI Translation Cache, Parsing & Sliding Context (Tier 1 & Tier 2)
# ============================================================================


def _sample_srts(count: int = 3) -> list[dict]:
    items = []
    for i in range(1, count + 1):
        start_ms = (i - 1) * 1500
        end_ms = start_ms + 1200
        s_str = f"00:00:{start_ms // 1000:02d},{start_ms % 1000:03d}"
        e_str = f"00:00:{end_ms // 1000:02d},{end_ms % 1000:03d}"
        items.append({
            "line": i,
            "start_time": start_ms,
            "end_time": end_ms,
            "startraw": s_str,
            "endraw": e_str,
            "time": f"{s_str} --> {e_str}",
            "text": f"Sample line number {i}",
        })
    return items


@pytest.mark.parametrize(
    ("channel_name", "aisendsrt"),
    [
        ("chatgpt", False),
        ("chatgpt", True),
        ("deepseek", False),
        ("deepseek", True),
    ],
)
def test_f7_translator_run_chatgpt_and_deepseek_via_local_http(local_llm_server, channel_name: str, aisendsrt: bool):
    from videotrans import translator
    from videotrans.configure import config as runtime_config

    if channel_name == "chatgpt":
        t_type = translator.CHATGPT_INDEX
        runtime_config.params.chatgpt_api = local_llm_server.openai_url
        runtime_config.params["chatgpt_key"] = "test-openai-key"
        runtime_config.params.chatgpt_model = "gpt-5-mini"
    else:
        t_type = translator.DEEPSEEK_INDEX
        runtime_config.params.deepseek_api = local_llm_server.openai_url
        runtime_config.params["deepseek_key"] = "test-deepseek-key"
        runtime_config.params.deepseek_model = "deepseek-v4-flash"

    res = translator.run(
        translate_type=t_type,
        text_list=_sample_srts(3),
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=aisendsrt,
        uuid=f"f7-{channel_name}-{aisendsrt}",
    )
    assert len(res) == 3
    assert [it["text"] for it in res] == [
        "[TR] Sample line number 1",
        "[TR] Sample line number 2",
        "[TR] Sample line number 3",
    ]
    assert len(local_llm_server.requests) >= 1


def test_f7_translator_run_gemini_via_local_http_sse(local_llm_server):
    from videotrans import translator
    from videotrans.configure import config as runtime_config

    runtime_config.params.gemini_api = local_llm_server.gemini_url
    runtime_config.params["gemini_key"] = "test-gemini-key"
    runtime_config.params.gemini_model = "gemini-2.5-flash"

    res = translator.run(
        translate_type=translator.GEMINI_INDEX,
        text_list=_sample_srts(2),
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f7-gemini-sse",
    )
    assert len(res) == 2
    assert res[0]["text"] == "[TR] Sample line number 1"
    assert res[1]["text"] == "[TR] Sample line number 2"
    assert any("streamGenerateContent" in r["path"] for r in local_llm_server.requests)


def test_f7_run_srt_cache_hit_on_second_identical_run(local_llm_server):
    from videotrans import translator
    from videotrans.configure import config as runtime_config

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"

    srts = _sample_srts(3)
    res1 = translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=srts,
        is_test=False,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f7-cache-1",
    )
    first_count = len(local_llm_server.requests)
    assert first_count >= 1

    res2 = translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=_sample_srts(3),
        is_test=False,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f7-cache-2",
    )
    assert len(local_llm_server.requests) == first_count, "Expected 100% cache hit on identical 2nd SRT run"
    assert [r["text"] for r in res2] == [r["text"] for r in res1]


def test_f7_strips_multiline_think_blocks_and_unclosed_translate_text(local_llm_server):
    from videotrans import translator
    from videotrans.configure import config as runtime_config

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"
    local_llm_server.think_prefix = "<think>\nstep 1 reasoning\nstep 2 reasoning\n</think>\n"
    local_llm_server.unclosed_tag = True

    res = translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=_sample_srts(2),
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f7-think-strip",
    )
    assert len(res) == 2
    assert all("<think>" not in r["text"] and "<TRANSLATE_TEXT>" not in r["text"] for r in res)
    assert res[0]["text"] == "[TR] Sample line number 1"


def test_f7_aitrans_context_sliding_window_and_google_url_encoding(local_llm_server, monkeypatch: pytest.MonkeyPatch):
    from videotrans import translator
    from videotrans.configure import config as runtime_config
    from videotrans.translator import _google, _runner

    # 1. Sliding context window when aitrans_context=True (should NOT cram all 6 into 1 batch)
    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"
    monkeypatch.setattr(runtime_config.settings, "aitrans_context", True, raising=False)
    monkeypatch.setattr(runtime_config.settings, "aitrans_thread", 2, raising=False)

    res = translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=_sample_srts(6),
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f7-sliding-ctx",
    )
    assert len(res) == 6
    assert len(local_llm_server.requests) >= 3, "aitrans_context=True should still batch with a sliding window instead of 1 giant prompt"

    # 2. Google Translate URL encoding, verify=True, timeout, and html.unescape
    captured = []

    class FakeResp:
        status_code = 200
        text = '<div class="result-container">Tom &amp; Jerry&#39;s &quot;Show&quot;</div>'

        def raise_for_status(self):
            return None

    def fake_get(url, *args, **kwargs):
        captured.append({"url": url, "kwargs": kwargs})
        return FakeResp()

    monkeypatch.setattr(_runner, "_check_google", lambda: True)
    monkeypatch.setattr(_google.requests, "get", fake_get)
    if hasattr(_google.requests, "Session"):
        monkeypatch.setattr(_google.requests.Session, "get", lambda self, url, *a, **kw: fake_get(url, *a, **kw))

    g_res = translator.run(
        translate_type=translator.GOOGLE_INDEX,
        text_list=[{
            "line": 1, "start_time": 0, "end_time": 1000,
            "startraw": "00:00:00,000", "endraw": "00:00:01,000",
            "time": "00:00:00,000 --> 00:00:01,000",
            "text": "A&B=100%?",
        }],
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=False,
        uuid="f7-google-encode",
    )
    assert g_res[0]["text"] == 'Tom & Jerry\'s "Show"'
    assert captured
    call = captured[0]
    assert call["kwargs"].get("verify") is True
    assert call["kwargs"].get("timeout") is not None
    assert "A&B=100%?" not in call["url"], f"Raw unescaped query parameter leaked into URL: {call['url']}"


def test_f7_openaicompat_instantiation_with_aitrans_context_and_text_list_none(monkeypatch: pytest.MonkeyPatch):
    from videotrans.configure import config as runtime_config
    from videotrans.translator._openaicompat import OpenAICampat

    monkeypatch.setattr(runtime_config.settings, "aitrans_context", True, raising=False)
    ob = OpenAICampat(ainame="chatgpt", text_list=None, uuid="ctx-none-test")
    assert ob is not None


def test_f7_cache_isolated_between_modes_and_target_languages(local_llm_server):
    from videotrans import translator
    from videotrans.configure import config as runtime_config

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"

    translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=_sample_srts(2),
        is_test=False,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f7-iso-1",
    )
    c1 = len(local_llm_server.requests)

    translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=_sample_srts(2),
        is_test=False,
        source_code="en",
        target_code="ja",
        aisendsrt=True,
        uuid="f7-iso-2",
    )
    assert len(local_llm_server.requests) > c1


def test_f7_is_test_true_bypasses_translation_cache(local_llm_server):
    from videotrans import translator
    from videotrans.configure import config as runtime_config

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"

    translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=_sample_srts(2),
        is_test=False,
        source_code="en",
        target_code="vi",
        aisendsrt=False,
        uuid="f7-istest-1",
    )
    c1 = len(local_llm_server.requests)
    translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=_sample_srts(2),
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=False,
        uuid="f7-istest-2",
    )
    assert len(local_llm_server.requests) > c1


def test_f7_run_text_preserves_1_to_1_line_count_with_whitespace(local_llm_server):
    from videotrans import translator
    from videotrans.configure import config as runtime_config

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"

    srts = _sample_srts(3)
    srts[1]["text"] = "(Music)"
    res = translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=srts,
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=False,
        uuid="f7-line-preserve",
    )
    assert len(res) == 3
    assert all(r["text"].strip() for r in res)


def test_f7_empty_translation_response_raises_translate_srt_error(local_llm_server):
    from videotrans import translator
    from videotrans.configure import config as runtime_config
    from videotrans.configure.excepts import TranslateSrtError

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"
    local_llm_server.empty_response = True

    with pytest.raises(TranslateSrtError):
        translator.run(
            translate_type=translator.CHATGPT_INDEX,
            text_list=_sample_srts(1),
            is_test=True,
            source_code="en",
            target_code="vi",
            aisendsrt=True,
            uuid="f7-empty-err",
        )


# ============================================================================
# F8: AI Translation Concurrency, Pooling, Retries & Prompts (Tier 1 & Tier 2)
# ============================================================================


def test_f8_concurrent_batch_execution_preserves_order_and_speedup(local_llm_server, monkeypatch: pytest.MonkeyPatch):
    from videotrans import translator
    from videotrans.configure import config as runtime_config

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"
    monkeypatch.setattr(runtime_config.settings, "aitrans_thread", 1, raising=False)
    monkeypatch.setattr(runtime_config.settings, "translation_wait", 0, raising=False)

    local_llm_server.delay_sec = 0.25
    t0 = time.perf_counter()
    res = translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=_sample_srts(4),
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f8-concurrency",
    )
    elapsed = time.perf_counter() - t0
    assert [it["text"] for it in res] == [f"[TR] Sample line number {i}" for i in range(1, 5)]
    assert local_llm_server.max_in_flight >= 2, "Expected concurrent batch execution (max_in_flight >= 2)"
    assert elapsed < 0.75, f"Concurrent execution took {elapsed:.2f}s (expected < 0.75s vs 1.0s sequential)"


def test_f8_retries_on_429_and_500_even_when_retry_nums_is_1(local_llm_server, monkeypatch: pytest.MonkeyPatch):
    from videotrans import translator
    from videotrans.configure import config as runtime_config

    monkeypatch.setattr(runtime_config.settings, "retry_nums", 1, raising=False)

    # 1. OpenAICampat retry on 429
    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"
    local_llm_server.failures_before_ok = 2
    local_llm_server.failure_status = 429
    res_oai = translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=_sample_srts(1),
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f8-retry-oai",
    )
    assert res_oai[0]["text"] == "[TR] Sample line number 1"
    assert len(local_llm_server.requests) >= 3

    # 2. Gemini retry on 500
    local_llm_server.reset()
    runtime_config.params.gemini_api = local_llm_server.gemini_url
    runtime_config.params["gemini_key"] = "test-gemini-key"
    runtime_config.params.gemini_model = "gemini-2.5-flash"
    local_llm_server.failures_before_ok = 1
    local_llm_server.failure_status = 500
    res_gem = translator.run(
        translate_type=translator.GEMINI_INDEX,
        text_list=_sample_srts(1),
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f8-retry-gem",
    )
    assert res_gem[0]["text"] == "[TR] Sample line number 1"
    assert len(local_llm_server.requests) >= 2


def test_f8_gemini_thread_safe_key_rotation_and_client_pooling(local_llm_server, monkeypatch: pytest.MonkeyPatch):
    from videotrans import translator
    from videotrans.configure import config as runtime_config
    from videotrans.translator import _gemini

    created_keys: list[str] = []
    real_client = _gemini.genai.Client

    def spy_client(*, api_key=None, **kwargs):
        created_keys.append(api_key)
        return real_client(api_key=api_key, **kwargs)

    monkeypatch.setattr(_gemini.genai, "Client", spy_client)
    runtime_config.params.gemini_api = local_llm_server.gemini_url
    runtime_config.params["gemini_key"] = "key-alpha,key-beta"
    runtime_config.params.gemini_model = "gemini-2.5-flash"
    monkeypatch.setattr(runtime_config.settings, "aitrans_thread", 1, raising=False)

    res = translator.run(
        translate_type=translator.GEMINI_INDEX,
        text_list=_sample_srts(6),
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f8-gem-pool",
    )
    assert len(res) == 6
    used_tokens = {r["headers"].get("x-goog-api-key") for r in local_llm_server.requests}
    assert {"key-alpha", "key-beta"}.issubset(used_tokens)
    # Pooled client reuse: at most 1 Client created per unique key (2 keys -> <= 2 clients for 6 batches)
    assert len(created_keys) <= 2, f"Expected pooled genai.Client reuse per key, got {len(created_keys)} constructions"


def test_f8_static_system_prompt_prefix_and_batch_filtered_glossary(local_llm_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from videotrans import translator
    from videotrans.configure import config as runtime_config
    from videotrans.util import prompts as prompts_mod

    # Copy prompts into isolated tmp_path and add a temporary glossary.txt without touching repo files
    fake_vt = tmp_path / "videotrans"
    shutil.copytree(Path(runtime_config.ROOT_DIR) / "videotrans" / "prompts", fake_vt / "prompts")
    (fake_vt / "glossary.txt").write_text("Quantum=Lượng tử\nBanana=Chuối\n", encoding="utf-8")
    monkeypatch.setattr(prompts_mod, "ROOT_DIR", str(tmp_path))

    assert prompts_mod.qwenmt_glossary() == [
        {"source": "Quantum", "target": "Lượng tử"},
        {"source": "Banana", "target": "Chuối"},
    ]
    assert "Quantum" in prompts_mod.get_prompt("chatgpt", aisendsrt=True)

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"
    monkeypatch.setattr(runtime_config.settings, "aitrans_thread", 1, raising=False)

    srts = _sample_srts(2)
    srts[0]["text"] = "Quantum physics lecture"
    srts[1]["text"] = "Regular everyday greeting"

    translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=srts,
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f8-prefix-glossary",
    )

    assert len(local_llm_server.requests) == 2
    sys_prompts = [r["body"]["messages"][0]["content"] for r in local_llm_server.requests]
    assert sys_prompts[0] == sys_prompts[1]
    assert len(sys_prompts[0]) > 200, "Static translation instructions should live in system message for prefix caching"

    user_prompts = [r["body"]["messages"][-1]["content"] for r in local_llm_server.requests]
    batch0_user = next(u for u in user_prompts if "Quantum physics" in u)
    batch1_user = next(u for u in user_prompts if "Regular everyday" in u)
    assert "Quantum" in batch0_user
    assert "Banana" not in batch0_user
    assert "Quantum" not in batch1_user


def test_f8_api_translate_cleans_temp_dirs_and_emits_batch_progress(e2e_client, local_llm_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from videotrans import translator
    from videotrans.configure import config as runtime_config
    from videotrans.task.orchestrator import EventKind, TaskRequest, run_staged_translation

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"
    monkeypatch.setattr(runtime_config.settings, "aitrans_thread", 2, raising=False)

    res = e2e_client.post(
        "/api/translate",
        json={
            "sourceLanguage": "en",
            "targetLanguage": "vi",
            "translateType": translator.CHATGPT_INDEX,
            "translationMode": "srt",
            "segments": [
                {"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "One", "targetText": ""},
                {"id": 2, "startSec": 1.0, "endSec": 2.0, "sourceText": "Two", "targetText": ""},
                {"id": 3, "startSec": 2.0, "endSec": 3.0, "sourceText": "Three", "targetText": ""},
            ],
        },
    )
    assert res.status_code == 200, res.text
    segs = res.json()["segments"]
    assert [s["targetText"] for s in segs] == ["[TR] One", "[TR] Two", "[TR] Three"]

    temp_dir = Path(runtime_config.TEMP_DIR)
    assert list(temp_dir.glob("cache-*")) == []
    assert list(temp_dir.glob("target-*")) == []
    assert list(temp_dir.glob("webui-trans-*.mp4")) == []

    # Progress events during multi-batch run_staged_translation
    dummy_mp4 = tmp_path / "prog.mp4"
    dummy_mp4.touch()
    c_dir = temp_dir / f"prog-{uuid.uuid4().hex[:6]}"
    c_dir.mkdir(parents=True, exist_ok=True)
    events = []
    run_staged_translation(
        TaskRequest({
            "name": dummy_mp4.as_posix(),
            "target_dir": (tmp_path / "prog_out").as_posix(),
            "cache_folder": c_dir.as_posix(),
            "source_language_code": "en",
            "target_language_code": "vi",
            "translate_type": translator.CHATGPT_INDEX,
            "aisendsrt": True,
            "segments": [
                {"id": i, "startSec": float(i - 1), "endSec": float(i) - 0.1, "sourceText": f"Line {i}", "targetText": ""}
                for i in range(1, 5)
            ],
            "uuid": "prog-trans",
        }),
        event_sink=events.append,
    )
    prog_values = [e.progress for e in events if e.kind == EventKind.PROGRESS and e.progress is not None]
    assert len(prog_values) >= 2


def test_f8_non_retryable_401_auth_error_fails_fast(local_llm_server):
    from videotrans import translator
    from videotrans.configure import config as runtime_config
    from videotrans.configure.excepts import StopTask

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "bad-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"
    local_llm_server.always_fail_status = 401

    with pytest.raises(StopTask):
        translator.run(
            translate_type=translator.CHATGPT_INDEX,
            text_list=_sample_srts(1),
            is_test=True,
            source_code="en",
            target_code="vi",
            aisendsrt=True,
            uuid="f8-auth-401",
        )
    assert len(local_llm_server.requests) == 1


def test_f8_openai_client_reused_across_batches(local_llm_server, monkeypatch: pytest.MonkeyPatch):
    from videotrans import translator
    from videotrans.configure import config as runtime_config
    from videotrans.translator import _openaicompat

    constructions = []
    real_openai = _openaicompat.OpenAI

    def spy_openai(*args, **kwargs):
        constructions.append(kwargs)
        return real_openai(*args, **kwargs)

    monkeypatch.setattr(_openaicompat, "OpenAI", spy_openai)
    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"
    monkeypatch.setattr(runtime_config.settings, "aitrans_thread", 1, raising=False)

    translator.run(
        translate_type=translator.CHATGPT_INDEX,
        text_list=_sample_srts(4),
        is_test=True,
        source_code="en",
        target_code="vi",
        aisendsrt=True,
        uuid="f8-oai-reuse",
    )
    assert len(local_llm_server.requests) == 4
    assert len(constructions) == 1, f"Expected single pooled OpenAI client per run, got {len(constructions)}"


def test_f8_gemini_400_bad_request_not_retried(local_llm_server):
    from videotrans import translator
    from videotrans.configure import config as runtime_config
    from videotrans.configure.excepts import StopTask

    runtime_config.params.gemini_api = local_llm_server.gemini_url
    runtime_config.params["gemini_key"] = "test-gemini-key"
    runtime_config.params.gemini_model = "gemini-2.5-flash"
    local_llm_server.always_fail_status = 400

    with pytest.raises(StopTask):
        translator.run(
            translate_type=translator.GEMINI_INDEX,
            text_list=_sample_srts(1),
            is_test=True,
            source_code="en",
            target_code="vi",
            aisendsrt=True,
            uuid="f8-gem-400",
        )
    assert len(local_llm_server.requests) == 1


def test_f8_prompt_loader_cached_across_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from videotrans.configure import config as runtime_config
    from videotrans.util import prompts as prompts_mod

    fake_vt = tmp_path / "videotrans"
    shutil.copytree(Path(runtime_config.ROOT_DIR) / "videotrans" / "prompts", fake_vt / "prompts")
    glossary_file = fake_vt / "glossary.txt"
    glossary_file.write_text("AlphaTerm=BetaTrans\n", encoding="utf-8")
    monkeypatch.setattr(prompts_mod, "ROOT_DIR", str(tmp_path))

    read_paths: list[Path] = []
    orig_read_text = Path.read_text

    def spy_read_text(self: Path, *args, **kwargs) -> str:
        if tmp_path in self.parents:
            read_paths.append(self)
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read_text)

    # First public call populates prompt & glossary caches from disk
    p1 = prompts_mod.get_prompt("chatgpt", aisendsrt=True)
    g1 = prompts_mod.qwenmt_glossary()
    assert "AlphaTerm" in p1
    assert g1 == [{"source": "AlphaTerm", "target": "BetaTrans"}]
    initial_reads = len(read_paths)
    assert initial_reads >= 2

    # Repeated calls hit caches without re-reading disk files
    t0 = time.perf_counter()
    for _ in range(50):
        assert prompts_mod.get_prompt("chatgpt", aisendsrt=True) == p1
        assert prompts_mod.qwenmt_glossary() == g1
    elapsed = time.perf_counter() - t0
    assert len(read_paths) == initial_reads, f"Expected cache hits on repeated get_prompt/qwenmt_glossary calls, saw extra reads: {read_paths[initial_reads:]}"
    assert elapsed < 0.25

    # Updating glossary.txt invalidates cache transparently via public API
    glossary_file.write_text("AlphaTerm=BetaTrans\nGammaTerm=DeltaTrans\n", encoding="utf-8")
    p_updated = prompts_mod.get_prompt("chatgpt", aisendsrt=True)
    g_updated = prompts_mod.qwenmt_glossary()
    assert "GammaTerm" in p_updated and "DeltaTrans" in p_updated
    assert g_updated == [
        {"source": "AlphaTerm", "target": "BetaTrans"},
        {"source": "GammaTerm", "target": "DeltaTrans"},
    ]


def test_f8_api_translate_empty_and_identical_lang_bypass(e2e_client, local_llm_server):
    res_empty = e2e_client.post(
        "/api/translate",
        json={"sourceLanguage": "en", "targetLanguage": "vi", "translateType": 1, "segments": []},
    )
    assert res_empty.status_code == 200
    assert res_empty.json() == {"ok": True, "segments": []}

    res_same = e2e_client.post(
        "/api/translate",
        json={
            "sourceLanguage": "en",
            "targetLanguage": "en",
            "translateType": 1,
            "segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Unchanged", "targetText": ""}],
        },
    )
    assert res_same.status_code == 200
    assert res_same.json()["segments"][0]["targetText"] == "Unchanged"
    assert len(local_llm_server.requests) == 0


# ============================================================================
# F9: TTS Preview Concurrency, Batch Dubbing, NumPy Assembly & CLI (Tier 1 & 2)
# ============================================================================


def test_f9_tts_preview_cloud_not_blocked_by_slow_local_lock(e2e_app, monkeypatch: pytest.MonkeyPatch):
    from videotrans import tts
    from videotrans.configure import config as runtime_config
    from videotrans.services import voice_preview

    def slow_local_vieneu(ref_path, out_path: Path, text: str, use_cuda: bool, **kwargs):
        time.sleep(0.7)
        make_wav(out_path, duration_s=1.0, sr=24000, freq=300.0)

    class FakeElevenLabsCloud:
        def __init__(self, api_key="", **kwargs):
            self.text_to_speech = SimpleNamespace(
                convert=lambda **kw: [make_wav_bytes(duration_s=0.6, sr=24000, freq=500.0)]
            )

    def make_wav_bytes(duration_s=0.6, sr=24000, freq=500.0) -> bytes:
        import io
        buf = io.BytesIO()
        t = np.linspace(0.0, duration_s, int(duration_s * sr), endpoint=False)
        sf.write(buf, (0.3 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    import elevenlabs
    monkeypatch.setattr(voice_preview, "_run_vieneu_synthesis", slow_local_vieneu)
    monkeypatch.setattr(elevenlabs, "ElevenLabs", FakeElevenLabsCloud)
    runtime_config.params["elevenlabstts_key"] = "el-key"

    async def _run_concurrent():
        transport = httpx.ASGITransport(app=e2e_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            local_task = asyncio.create_task(
                client.post(
                    "/api/tts/preview",
                    json={"provider": tts.VIENEU_TTS, "voice": "Mai Anh", "text": "Slow local", "forceRefresh": True},
                )
            )
            await asyncio.sleep(0.05)
            t0 = time.perf_counter()
            cloud_res = await client.post(
                "/api/tts/preview",
                json={"provider": tts.ELEVENLABS_TTS, "voice": "Rachel", "text": "Fast cloud", "forceRefresh": True},
            )
            cloud_elapsed = time.perf_counter() - t0
            local_res = await local_task
            return local_res, cloud_res, cloud_elapsed

    local_res, cloud_res, cloud_elapsed = asyncio.run(_run_concurrent())
    assert local_res.status_code == 200, local_res.text
    assert cloud_res.status_code == 200, cloud_res.text
    assert cloud_elapsed < 0.45, f"Cloud TTS preview blocked behind local GPU lock ({cloud_elapsed:.2f}s)"


def test_f9_stage3_batch_dubbing_endpoint_reports_per_segment_status(e2e_client, monkeypatch: pytest.MonkeyPatch):
    from videotrans.services import voice_preview

    pid = e2e_client.post("/api/projects", json={"name": "Batch Dubbing", "duration": 4.0, "stage": 3}).json()["id"]

    def _synth(voice_dict, out_path: Path, sample_text: str):
        if "FAIL_ME" in sample_text:
            raise RuntimeError("Synthetic TTS segment failure")
        make_wav(out_path, duration_s=0.8, sr=24000, freq=440.0)

    voice_preview.set_preview_synthesizer(_synth)

    payload = {
        "provider": 2,
        "voice": "Mai Anh",
        "language": "vi",
        "segments": [
            {"id": 1, "segmentId": 1, "startSec": 0.0, "endSec": 1.0, "text": "Segment one"},
            {"id": 2, "segmentId": 2, "startSec": 1.2, "endSec": 2.2, "text": "FAIL_ME segment"},
            {"id": 3, "segmentId": 3, "startSec": 2.4, "endSec": 3.4, "text": "Segment three"},
        ],
    }

    res = e2e_client.post(f"/api/projects/{pid}/dubbing/generate", json=payload)
    if res.status_code == 404:
        res = e2e_client.post("/api/tts/batch", json={"projectId": pid, **payload})
    assert res.status_code == 200, f"M5 batch dubbing endpoint missing or failed: {res.status_code} {res.text}"

    body = res.json()
    results = body.get("results") or body.get("segments") or []
    assert len(results) == 3
    by_id = {r.get("segmentId", r.get("id")): r for r in results}
    assert by_id[1]["ok"] is True and by_id[1].get("previewUrl")
    assert by_id[2]["ok"] is False and by_id[2].get("error")
    assert by_id[3]["ok"] is True and by_id[3].get("previewUrl")


def test_f9_speed_rate_and_smart_fit_produce_valid_wav_and_clean_temps(tmp_path: Path):
    from videotrans.services.audio_fit import smart_fit_audio_file
    from videotrans.task._rate import SpeedRate

    # 1. smart_fit_audio_file fits a 2.0s clip into 1.4s slot
    raw_clip = make_wav(tmp_path / "fit_in.wav", duration_s=2.0, sr=24000, freq=440.0)
    fitted_path, speed, status = smart_fit_audio_file(raw_clip, slot_duration_s=1.4, max_speed_rate=1.6)
    assert status in {"audio_stretched", "overflow_trimmed"}
    assert speed > 1.0
    assert sf.info(str(fitted_path)).duration == pytest.approx(1.4, abs=0.2)

    # 2. SpeedRate stitches 2 dubbed clips cleanly and cleans temp intermediates
    cache_dir = tmp_path / "rate_cache"
    cache_dir.mkdir()
    c1 = make_wav(cache_dir / "c1.wav", duration_s=1.2, sr=24000, freq=440.0)
    c2 = make_wav(cache_dir / "c2.wav", duration_s=0.8, sr=24000, freq=660.0)
    target_audio = tmp_path / "merged_rate.wav"

    queue_tts = [
        {"line": 1, "start_time": 0, "end_time": 1000, "start_time_source": 0, "end_time_source": 1000, "filename": str(c1), "text": "One"},
        {"line": 2, "start_time": 1500, "end_time": 2500, "start_time_source": 1500, "end_time_source": 2500, "filename": str(c2), "text": "Two"},
    ]
    sr_inst = SpeedRate(
        queue_tts=queue_tts,
        should_videorate=False,
        should_audiorate=True,
        uuid="rate-e2e",
        novoice_mp4=None,
        raw_total_time=3000,
        target_audio=str(target_audio),
        cache_folder=str(cache_dir),
    )
    sr_inst.run()
    assert target_audio.is_file()
    info = sf.info(str(target_audio))
    assert info.duration >= 2.3
    assert list(tmp_path.glob("*-after.wav")) == []
    assert list(cache_dir.glob("silence_*.wav")) == []


def test_f9_cli_list_commands_return_zero(monkeypatch: pytest.MonkeyPatch, capsys):
    import cli

    for mode in ("providers", "languages", "models"):
        monkeypatch.setattr(sys, "argv", ["cli.py", "--list", mode])
        rc = cli.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert len(out.strip()) > 20


def test_f9_cli_stt_sts_tts_vtv_end_to_end(tmp_path: Path, local_llm_server, monkeypatch: pytest.MonkeyPatch):
    import cli
    from videotrans import recognition, translator, tts
    from videotrans.configure import config as runtime_config
    from videotrans.process import vad as vad_mod
    from videotrans.util._ffprobe import get_video_info

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"

    def fake_recogn_run(**kwargs):
        return [
            {
                "line": 1,
                "start_time": 0,
                "end_time": 1200,
                "startraw": "00:00:00,000",
                "endraw": "00:00:01,200",
                "time": "00:00:00,000 --> 00:00:01,200",
                "text": "Hello from CLI",
            }
        ]

    def fake_tts_run(*, queue_tts=None, **kwargs):
        for item in queue_tts or []:
            make_wav(item["filename"], duration_s=0.9, sr=24000, freq=440.0)

    monkeypatch.setattr(recognition, "run", fake_recogn_run)
    monkeypatch.setattr(tts, "run", fake_tts_run)
    from videotrans.task import _stage_dubbing, _stage_recogn
    monkeypatch.setattr(_stage_recogn, "run_recogn", fake_recogn_run)
    monkeypatch.setattr(_stage_dubbing, "run_tts", fake_tts_run)
    monkeypatch.setattr(vad_mod, "get_speech_timestamp", lambda **kw: ([[0, 1200]], None))

    mp4 = make_mp4(tmp_path / "cli_input.mp4", duration_s=1.5, with_audio=True)
    srt_in = tmp_path / "cli_input.srt"
    srt_in.write_text("1\n00:00:00,000 --> 00:00:01,200\nHello from CLI\n", encoding="utf-8")

    # 1. --task stt
    stt_out = tmp_path / "cli_stt_out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["cli.py", "--task", "stt", "--name", str(mp4), "--recogn_type", "1", "--detect_language", "en", "--output-dir", str(stt_out)],
    )
    assert cli.main() == 0
    assert list(stt_out.glob("*.srt"))

    # 2. --task sts
    sts_out = tmp_path / "cli_sts_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py", "--task", "sts", "--name", str(srt_in),
            "--translate_type", str(translator.CHATGPT_INDEX),
            "--source_language_code", "en", "--target_language_code", "vi",
            "--output-dir", str(sts_out),
        ],
    )
    assert cli.main() == 0
    sts_srts = list(sts_out.glob("*.srt"))
    assert sts_srts and "[TR]" in sts_srts[0].read_text(encoding="utf-8")

    # 3. --task tts
    tts_out = tmp_path / "cli_tts_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py", "--task", "tts", "--name", str(srt_in),
            "--tts_type", str(tts.VIENEU_TTS), "--voice_role", "Mai Anh",
            "--output-dir", str(tts_out),
        ],
    )
    assert cli.main() == 0
    assert list(tts_out.glob("*.wav"))

    # 4. --task vtv
    vtv_out = tmp_path / "cli_vtv_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py", "--task", "vtv", "--name", str(mp4),
            "--source_language_code", "en", "--target_language_code", "vi",
            "--recogn_type", "1",
            "--translate_type", str(translator.CHATGPT_INDEX),
            "--tts_type", str(tts.VIENEU_TTS), "--voice_role", "Mai Anh",
            "--output-dir", str(vtv_out),
        ],
    )
    assert cli.main() == 0
    out_mp4s = list(vtv_out.glob("*.mp4"))
    assert out_mp4s and get_video_info(out_mp4s[0]).get("video_streams", 0) >= 1


def test_f9_tts_run_raises_when_no_output_wav_produced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from videotrans import tts
    from videotrans.configure.excepts import DubbingSrtError
    from videotrans.tts import _vieneutts

    monkeypatch.setattr(_vieneutts.VieNeuTTS, "_exec", lambda self: None)
    with pytest.raises(DubbingSrtError):
        tts.run(
            tts_type=tts.VIENEU_TTS,
            queue_tts=[{"text": "Hello", "filename": str(tmp_path / "missing.wav"), "role": "Mai Anh", "rate": "+0%", "volume": "+0%", "pitch": "+0Hz"}],
            language="en",
            uuid="tts-empty-fail",
        )


def test_f9_smart_fit_audio_missing_and_nonpositive_slot(tmp_path: Path):
    from videotrans.services.audio_fit import smart_fit_audio_file

    _, spd1, st1 = smart_fit_audio_file(tmp_path / "nope.wav", slot_duration_s=1.0)
    assert st1 == "not_found" and spd1 == 1.0

    wav = make_wav(tmp_path / "zero_slot.wav", duration_s=1.0, sr=24000)
    _, spd2, st2 = smart_fit_audio_file(wav, slot_duration_s=0.0)
    assert st2 == "fits" and spd2 == 1.0


def test_f9_preview_force_refresh_and_cache_valid_wav(e2e_client):
    from videotrans.services import voice_preview

    calls = {"n": 0}

    def _synth(voice_dict, out_path: Path, sample_text: str):
        calls["n"] += 1
        make_wav(out_path, duration_s=0.7, sr=24000, freq=440.0 + calls["n"] * 20)

    voice_preview.set_preview_synthesizer(_synth)
    r1 = e2e_client.post("/api/tts/preview", json={"provider": 2, "voice": "Mai Anh", "text": "Cache check"})
    assert r1.status_code == 200
    r2 = e2e_client.post("/api/tts/preview", json={"provider": 2, "voice": "Mai Anh", "text": "Cache check", "forceRefresh": True})
    assert r2.status_code == 200
    assert calls["n"] == 2
    dl = e2e_client.get(r2.json()["audio_url"])
    assert dl.status_code == 200 and len(dl.content) > 44


def test_f9_cli_validation_errors_return_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import cli

    valid_mp4 = make_mp4(tmp_path / "cli_val.mp4", duration_s=0.6)
    bad_argvs = [
        ["cli.py", "--task", "stt"],  # missing --name
        ["cli.py", "--task", "stt", "--name", str(tmp_path / "missing.mp4")],  # nonexistent file
        ["cli.py", "--task", "tts", "--name", str(valid_mp4)],  # missing --voice_role
        ["cli.py", "--task", "sts", "--name", str(valid_mp4)],  # missing --target_language_code
        ["cli.py", "--task", "vtv", "--name", str(valid_mp4), "--target_language_code", "vi"],  # missing --source_language_code
    ]
    for argv in bad_argvs:
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code != 0


def test_f9_elevenlabs_and_gemini_tts_client_reuse_and_temp_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from videotrans import tts
    from videotrans.configure import config as runtime_config
    from videotrans.tts import _elevenlabs, _geminitts

    # 1. ElevenLabsC client reuse & .mp3 temp cleanup
    el_clients = []

    class SpyElevenLabs:
        def __init__(self, api_key="", **kwargs):
            el_clients.append(api_key)
            self.text_to_speech = SimpleNamespace(
                convert=lambda **kw: [b"\xff\xfb\x90\x00" + b"\x00" * 128]
            )

    monkeypatch.setattr(_elevenlabs, "ElevenLabs", SpyElevenLabs)
    monkeypatch.setattr(
        _elevenlabs.ElevenLabsC,
        "convert_to_wav",
        lambda self, mp3_file, wav_file: make_wav(wav_file, duration_s=0.5, sr=24000),
    )
    runtime_config.params["elevenlabstts_key"] = "el-reuse-key"

    el_items = [
        {"text": f"Line {i}", "filename": str(tmp_path / f"el_{i}.wav"), "role": "21m00Tcm4TlvDq8ikWAM", "rate": "+0%", "volume": "+0%", "pitch": "+0Hz"}
        for i in range(2)
    ]
    tts.run(tts_type=tts.ELEVENLABS_TTS, queue_tts=el_items, language="en", uuid="f9-el-reuse")
    assert len(el_clients) == 1, f"Expected reused ElevenLabs client, got {len(el_clients)}"
    assert not any((tmp_path / f"el_{i}.wav.mp3").exists() for i in range(2))

    # 2. GEMINITTS client reuse & .wav sidecar temp cleanup
    gem_clients = []

    class SpyGeminiClient:
        def __init__(self, api_key="", **kwargs):
            gem_clients.append(api_key)
            self.models = SimpleNamespace(
                generate_content=lambda **kw: SimpleNamespace(
                    candidates=[
                        SimpleNamespace(
                            content=SimpleNamespace(
                                parts=[
                                    SimpleNamespace(
                                        inline_data=SimpleNamespace(
                                            data=b"\x00\x01" * 2400,
                                            mime_type="audio/L16;rate=24000",
                                        )
                                    )
                                ]
                            )
                        )
                    ]
                )
            )

    monkeypatch.setattr(_geminitts.genai, "Client", SpyGeminiClient)
    monkeypatch.setattr(
        _geminitts.GEMINITTS,
        "convert_to_wav",
        lambda self, src_wav, dst_wav: make_wav(dst_wav, duration_s=0.5, sr=24000),
    )
    runtime_config.params["gemini_key"] = "gem-reuse-key"

    gem_items = [
        {"text": f"Line {i}", "filename": str(tmp_path / f"gem_{i}.wav"), "role": "Puck", "rate": "+0%", "volume": "+0%", "pitch": "+0Hz"}
        for i in range(2)
    ]
    tts.run(tts_type=tts.GEMINI_TTS, queue_tts=gem_items, language="en", uuid="f9-gem-reuse")
    assert len(gem_clients) == 1, f"Expected reused Gemini client, got {len(gem_clients)}"
    assert not any((tmp_path / f"gem_{i}.wav.wav").exists() for i in range(2))


def test_f9_previews_dir_and_dubbing_cache_evict_stale_entries(
    e2e_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from videotrans import tts
    from videotrans.configure import config as runtime_config
    from videotrans.core import voice_store
    from videotrans.services import voice_preview
    from videotrans.task import _stage_dubbing
    from videotrans.task.taskcfg import TaskCfgVTT
    from videotrans.task.trans_create import TransCreate

    stale_ts = time.time() - 365 * 86400

    # 1. PREVIEWS_DIR bounded LRU/TTL eviction during preview synthesis
    previews_dir = Path(voice_store.PREVIEWS_DIR)
    previews_dir.mkdir(parents=True, exist_ok=True)
    stale_prev_wav = previews_dir / "prev_stale_expired.wav"
    stale_prev_json = previews_dir / "prev_stale_expired.json"
    make_wav(stale_prev_wav, duration_s=0.5, sr=24000, freq=330.0)
    stale_prev_json.write_text('{"applied_speed": 1.0}', encoding="utf-8")
    os.utime(stale_prev_wav, (stale_ts, stale_ts))
    os.utime(stale_prev_json, (stale_ts, stale_ts))

    fresh_prev_wav = previews_dir / "prev_fresh_recent.wav"
    fresh_prev_json = previews_dir / "prev_fresh_recent.json"
    make_wav(fresh_prev_wav, duration_s=0.5, sr=24000, freq=440.0)
    fresh_prev_json.write_text('{"applied_speed": 1.0}', encoding="utf-8")

    def _synth(voice_dict, out_path: Path, sample_text: str):
        make_wav(out_path, duration_s=0.5, sr=24000, freq=520.0)

    voice_preview.set_preview_synthesizer(_synth)
    res = e2e_client.post(
        "/api/tts/preview",
        json={"provider": tts.VIENEU_TTS, "voice": "Mai Anh", "text": "Eviction check"},
    )
    assert res.status_code == 200, res.text
    assert not stale_prev_wav.exists(), "Stale preview WAV in PREVIEWS_DIR should be evicted"
    assert not stale_prev_json.exists(), "Stale preview metadata JSON in PREVIEWS_DIR should be evicted"
    assert fresh_prev_wav.exists(), "Recent preview WAV in PREVIEWS_DIR should be retained"

    # 2. DUBBING_CACHE bounded LRU/TTL eviction during stage dubbing
    dub_cache_dir = Path(runtime_config.DUBBING_CACHE)
    dub_cache_dir.mkdir(parents=True, exist_ok=True)
    stale_dub_wav = dub_cache_dir / "stale_deadbeef0001.wav"
    fresh_dub_wav = dub_cache_dir / "fresh_cafebabe0002.wav"
    make_wav(stale_dub_wav, duration_s=0.5, sr=24000, freq=350.0)
    os.utime(stale_dub_wav, (stale_ts, stale_ts))
    make_wav(fresh_dub_wav, duration_s=0.5, sr=24000, freq=460.0)

    def _fake_run_tts(*, queue_tts=None, **kwargs):
        for item in queue_tts or []:
            make_wav(item["filename"], duration_s=0.6, sr=24000, freq=480.0)

    monkeypatch.setattr(_stage_dubbing, "run_tts", _fake_run_tts)

    target_dir = tmp_path / "dub_cache_out"
    cache_dir = Path(runtime_config.TEMP_DIR) / f"dub-evict-{uuid.uuid4().hex[:6]}"
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello world\n", encoding="utf-8")
    (target_dir / "vi.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")

    mp4 = make_mp4(tmp_path / "dub_evict.mp4", duration_s=1.2, with_audio=True)
    cfg = TaskCfgVTT(
        name=mp4.as_posix(),
        target_dir=target_dir.as_posix(),
        cache_folder=cache_dir.as_posix(),
        source_language_code="en",
        target_language_code="vi",
        tts_type=tts.VIENEU_TTS,
        voice_role="Mai Anh",
        clear_cache=False,
        uuid=f"dub-evict-{uuid.uuid4().hex[:6]}",
    )
    task = TransCreate(cfg=cfg, event_sink=lambda _e: None)
    try:
        task.dubbing()
    finally:
        task.set_end()

    assert not stale_dub_wav.exists(), "Stale WAV in DUBBING_CACHE should be evicted"
    assert fresh_dub_wav.exists(), "Recent WAV in DUBBING_CACHE should be retained"
    assert len(list(dub_cache_dir.glob("*.wav"))) >= 2
