# -*- coding: utf-8 -*-
"""Unit, integration, and reproducible benchmark tests for M5 TTS & dubbing optimizations."""
from __future__ import annotations

import asyncio
import gc
import json
import os
import sys
import threading
import time
import types
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


def _make_tone_wav(path: Path, duration_s: float = 0.5, sr: int = 24000, freq: float = 440.0, channels: int = 1) -> Path:
    n = max(1, int(round(duration_s * sr)))
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float32)
    mono = 0.35 * np.sin(2.0 * np.pi * freq * t)
    if channels == 2:
        data = np.column_stack((mono, mono * 0.8))
    else:
        data = mono
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16", format="WAV")
    return path


class TestPreviewLockDecouplingAndNonBlocking:
    def test_cloud_preview_does_not_serialize_behind_gpu_preview(self, tmp_path: Path, monkeypatch):
        from videotrans import tts
        from videotrans.configure import config as runtime_config
        from videotrans.core import voice_store
        from videotrans.services import voice_preview

        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", tmp_path / "previews")
        voice_preview.set_preview_synthesizer(None)
        monkeypatch.setattr("videotrans.util.help_role.get_vieneu_custom_voice_path", lambda v: None)
        prev_gem_key = runtime_config.params.get("gemini_key", "")
        runtime_config.params["gemini_key"] = "fake-gem-key"

        gpu_entered = threading.Event()
        gpu_release = threading.Event()
        cloud_finished_while_gpu_blocked = {"value": False}

        def slow_vieneu(*args, **kwargs):
            out = kwargs.get("output_file") or args[1]
            gpu_entered.set()
            assert gpu_release.wait(timeout=3.0)
            _make_tone_wav(Path(out), duration_s=0.4, sr=24000, freq=440.0)

        def fast_gemini(preview_path, sample_text, target_voice, language=None):
            _make_tone_wav(Path(preview_path), duration_s=0.4, sr=24000, freq=520.0)
            if gpu_entered.is_set() and not gpu_release.is_set():
                cloud_finished_while_gpu_blocked["value"] = True

        monkeypatch.setattr(voice_preview, "_run_vieneu_synthesis", slow_vieneu)
        monkeypatch.setattr(voice_preview, "_run_gemini_preview", fast_gemini)

        async def _run():
            gpu_task = asyncio.create_task(
                voice_preview.synthesize_unified_tts_preview(
                    provider=tts.VIENEU_TTS,
                    voice="Mai Anh",
                    text="Local GPU blocked",
                    force_refresh=True,
                )
            )
            # Wait for GPU task to enter _run_vieneu_synthesis inside _gpu_lock
            for _ in range(100):
                if gpu_entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert gpu_entered.is_set()

            cloud_res = await asyncio.wait_for(
                voice_preview.synthesize_unified_tts_preview(
                    provider=tts.GEMINI_TTS,
                    voice="Puck",
                    text="Fast cloud call",
                    force_refresh=True,
                ),
                timeout=1.5,
            )
            assert cloud_finished_while_gpu_blocked["value"] is True
            assert Path(cloud_res[1]).is_file()
            gpu_release.set()
            gpu_res = await gpu_task
            assert Path(gpu_res[1]).is_file()

        try:
            asyncio.run(_run())
        finally:
            runtime_config.params["gemini_key"] = prev_gem_key

    def test_cloud_preview_offloads_off_event_loop(self, tmp_path: Path, monkeypatch):
        from videotrans import tts
        from videotrans.configure import config as runtime_config
        from videotrans.core import voice_store
        from videotrans.services import voice_preview

        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", tmp_path / "previews")
        voice_preview.set_preview_synthesizer(None)
        prev_el_key = runtime_config.params.get("elevenlabstts_key", "")
        runtime_config.params["elevenlabstts_key"] = "fake-el-key"

        # Build a valid RIFF WAV buffer returned by mocked ElevenLabs SDK
        wav_buf_path = _make_tone_wav(tmp_path / "el_src.wav", duration_s=0.3, sr=44100)
        wav_bytes = wav_buf_path.read_bytes()

        class SlowElevenLabs:
            def __init__(self, api_key=""):
                self.text_to_speech = SimpleNamespace(
                    convert=self._convert,
                )

            def _convert(self, **kwargs):
                time.sleep(0.25)
                return [wav_bytes]

        import elevenlabs
        monkeypatch.setattr(elevenlabs, "ElevenLabs", SlowElevenLabs)

        async def _run():
            t0 = time.perf_counter()
            tts_task = asyncio.create_task(
                voice_preview.synthesize_unified_tts_preview(
                    provider=tts.ELEVENLABS_TTS,
                    voice="21m00Tcm4TlvDq8ikWAM",
                    text="Non blocking check",
                    force_refresh=True,
                )
            )
            # Event loop heartbeat ticks should complete well under 0.1s while thread sleeps 0.25s
            await asyncio.sleep(0.02)
            tick_elapsed = time.perf_counter() - t0
            assert tick_elapsed < 0.15, f"Event loop blocked for {tick_elapsed:.3f}s"
            _, p_path, _ = await tts_task
            assert p_path.is_file()

        try:
            asyncio.run(_run())
        finally:
            runtime_config.params["elevenlabstts_key"] = prev_el_key


class TestModelCacheAndBatchEndpoint:
    def test_vieneu_cache_shares_single_instance_with_max_batch_size_4(self):
        from videotrans.services import model_cache

        model_cache.clear_model_cache()
        fake_engine = MagicMock()
        fake_engine.max_batch_size = 1
        fake_engine.close = MagicMock()

        created = []

        def fake_ctor(**kwargs):
            created.append(kwargs)
            return fake_engine

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("vieneu.Vieneu", fake_ctor)
            e1 = model_cache.get_cached_vieneu_engine(device="cpu", backend="onnx", max_batch_size=1)
            e2 = model_cache.get_cached_vieneu_engine(device="cpu", backend="onnx", max_batch_size=4)
            assert e1 is e2
            assert len(created) == 1
            assert e2.max_batch_size == 4
            model_cache.evict_vieneu_engine(device="cpu")

    def test_batch_endpoint_groups_by_voice_partial_error_and_cancel(self, tmp_path: Path, monkeypatch):
        from videotrans.api.app import app
        from videotrans.core import voice_store
        from videotrans.services import model_cache, voice_preview

        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", tmp_path / "previews")
        voice_preview.set_preview_synthesizer(None)
        monkeypatch.setattr("videotrans.util.help_role.get_vieneu_custom_voice_path", lambda v: None)

        batch_calls = []

        class BatchSpyEngine:
            sample_rate = 24000
            max_batch_size = 4

            def get_preset_voice(self, name):
                return {"preset": name}

            def infer_batch(self, texts, **kwargs):
                batch_calls.append((list(texts), dict(kwargs)))
                t = np.linspace(0.0, 0.3, int(0.3 * 24000), endpoint=False, dtype=np.float32)
                tone = 0.4 * np.sin(2 * np.pi * 440.0 * t)
                # Return empty array for "FAIL_ME" to trigger partial failure on sub-batch item
                out = []
                for txt in texts:
                    if txt == "FAIL_ME":
                        out.append(np.array([], dtype=np.float32))
                    else:
                        out.append(tone.copy())
                return out

        monkeypatch.setattr(model_cache, "get_cached_vieneu_engine", lambda **kw: BatchSpyEngine())

        client = TestClient(app)
        resp = client.post("/api/tts/batch", json={
            "provider": 2,
            "language": "vi",
            "segments": [
                {"segmentId": 1, "text": "Câu một", "voice": "Mai Anh"},
                {"segmentId": 2, "text": "FAIL_ME", "voice": "Mai Anh"},
                {"segmentId": 3, "text": "Câu ba", "voice": "Mai Anh"},
                {"segmentId": 4, "text": "Câu bốn", "voice": "Mai Anh"},
                {"segmentId": 5, "text": "Câu năm", "voice": "Mai Anh"},
                {"segmentId": 6, "text": "Khác giọng", "voice": "Thanh Bình"},
            ],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert body["succeeded"] == 5
        assert body["failed"] == 1
        by_id = {r["segmentId"]: r for r in body["results"]}
        assert by_id[1]["ok"] is True and by_id[1]["previewUrl"]
        assert by_id[2]["ok"] is False and by_id[2]["error"]
        assert by_id[3]["ok"] is True
        assert by_id[4]["ok"] is True
        assert by_id[5]["ok"] is True
        assert by_id[6]["ok"] is True
        # First voice had 5 items -> sub-batched into 4 + 1, second voice had 1
        assert [len(c[0]) for c in batch_calls] == [4, 1, 1]
        assert batch_calls[0][1].get("batch_size") == 4

        # Cancellation test
        cancel_ev = threading.Event()
        cancel_ev.set()
        cancelled_res = asyncio.run(
            voice_preview.synthesize_tts_batch(
                [{"segment_id": 99, "text": "Should cancel", "voice": "Mai Anh", "provider": 2, "force_refresh": True}],
                cancel_event=cancel_ev,
            )
        )
        assert len(cancelled_res) == 1
        assert cancelled_res[0]["ok"] is False
        assert "Cancel" in (cancelled_res[0]["error"] or "")


class TestNumpyAssemblyAndBenchmark:
    def test_numpy_assembly_mixed_rates_collision_and_speed_vs_pydub(self, tmp_path: Path, monkeypatch):
        from fastapi.testclient import TestClient
        from pydub import AudioSegment
        from videotrans.api.app import app
        from videotrans.core import project_store, voice_store

        projects_dir = tmp_path / "projects"
        previews_dir = tmp_path / "previews"
        monkeypatch.setattr(project_store, "DEFAULT_PROJECTS_DIR", projects_dir, raising=False)
        monkeypatch.setattr(project_store, "PROJECTS_OUTPUT_DIR", projects_dir, raising=False)
        monkeypatch.setattr(project_store, "get_projects_root", lambda: projects_dir)
        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", previews_dir, raising=False)
        voice_store.init_voice_dirs()

        proj = project_store.create_project(name="Bench Proj", project_id="bench-proj")
        project_store.update_project("bench-proj", duration=16.0)

        # Create 16 mixed-rate clips (24kHz mono and 44.1kHz stereo) across 16s
        segments = []
        pydub_inputs = []
        for i in range(16):
            pid = f"prev_{i:012x}"
            sr = 24000 if i % 2 == 0 else 44100
            ch = 1 if i % 2 == 0 else 2
            wav_p = _make_tone_wav(previews_dir / f"{pid}.wav", duration_s=0.8, sr=sr, freq=300 + i * 25, channels=ch)
            start_s = float(i)
            segments.append({
                "id": i + 1,
                "startSec": start_s,
                "endSec": start_s + 0.8,
                "targetText": f"Line {i}",
                "previewAudioId": pid,
            })
            pydub_inputs.append((int(start_s * 1000), wav_p))

        # Measure legacy pydub overlay baseline
        t_pydub_0 = time.perf_counter()
        merged_pydub = AudioSegment.silent(duration=16000, frame_rate=48000).set_channels(1)
        for start_ms, wav_p in pydub_inputs:
            clip = AudioSegment.from_wav(wav_p).set_frame_rate(48000).set_channels(1)
            merged_pydub = merged_pydub.overlay(clip, position=start_ms)
        pydub_out = tmp_path / "pydub_baseline.wav"
        merged_pydub.export(pydub_out, format="wav")
        pydub_elapsed = time.perf_counter() - t_pydub_0

        # Measure NumPy vectorized handler
        client = TestClient(app)
        t_np_0 = time.perf_counter()
        resp = client.post("/api/projects/bench-proj/dubbing/assemble", json={"segments": segments})
        np_elapsed = time.perf_counter() - t_np_0
        assert resp.status_code == 200, resp.text

        out_wav = project_store.get_project_dir("bench-proj") / "dubbing" / "voiceover_merged.wav"
        assert out_wav.is_file()
        assert not (out_wav.parent / "voiceover_merged.tmp.wav").exists()
        info = sf.info(str(out_wav))
        assert info.samplerate == 48000
        assert info.channels == 1
        assert info.subtype == "PCM_16"
        assert info.duration == pytest.approx(16.0, abs=0.05)

        # Verify silent gap [0.85s .. 0.95s] is exact zero
        data, sr = sf.read(str(out_wav), dtype="float32")
        assert np.max(np.abs(data[int(0.85 * sr):int(0.95 * sr)])) < 1e-4
        assert np.max(np.abs(data[int(0.2 * sr):int(0.6 * sr)])) > 0.1
        assert np.max(np.abs(data[int(1.2 * sr):int(1.6 * sr)])) > 0.1
        assert np_elapsed <= pydub_elapsed


class TestBackendTTSProvidersAndRate:
    def test_base_tts_cloud_autoscale_error_retention_and_cleantts_guards(self):
        from videotrans import tts
        from videotrans.tts._base import BaseTTS

        # Empty queue & language=None must not crash _cleantts
        b_empty = BaseTTS(tts_type=tts.VIENEU_TTS, queue_tts=[], language=None)
        assert b_empty.len == 0
        assert b_empty.dub_nums >= 1

        # Cloud provider auto-scales concurrency and removes artificial 1s wait
        q = [{"text": f"hi {i}", "filename": f"f{i}.wav"} for i in range(6)]
        b_cloud = BaseTTS(tts_type=tts.GEMINI_TTS, queue_tts=q, language=None)
        assert b_cloud.dub_nums == 4
        assert b_cloud.wait_sec == 0.0

        # _local_mul_thread preserves earlier error when later tasks succeed
        class PartialFailTTS(BaseTTS):
            def _run(self, data_item, idx=-1):
                if idx == 0:
                    raise RuntimeError("boom-0")
                return None

        p = PartialFailTTS(tts_type=tts.GEMINI_TTS, queue_tts=q[:3], language="en", dub_nums=2, wait_sec=0.0)
        # Ensure filenames don't exist so tasks run
        p._local_mul_thread()
        assert isinstance(p.error, RuntimeError) and "boom-0" in str(p.error)

    def test_elevenlabs_pooling_proxy_voice_cache_and_mp3_cleanup(self, tmp_path: Path, monkeypatch):
        from videotrans import tts
        from videotrans.configure import config as runtime_config
        from videotrans.tts import _elevenlabs

        created_proxies = []
        created_clients = []

        class SpyHttpClient:
            def __init__(self, proxy=None, timeout=None):
                created_proxies.append(proxy)
                self.closed = False

            def close(self):
                self.closed = True

        class SpyElevenLabs:
            def __init__(self, api_key="", timeout=120.0, httpx_client=None):
                created_clients.append((api_key, httpx_client))
                self.text_to_speech = SimpleNamespace(
                    convert=lambda **kw: [b"ID3" + b"\x00" * 64],
                )

        monkeypatch.setattr(_elevenlabs.httpx, "Client", SpyHttpClient)
        monkeypatch.setattr(_elevenlabs, "ElevenLabs", SpyElevenLabs)
        monkeypatch.setattr(
            _elevenlabs.ElevenLabsC,
            "convert_to_wav",
            lambda self, src, dst: _make_tone_wav(Path(dst), duration_s=0.3, sr=48000),
        )
        runtime_config.params["elevenlabstts_key"] = "key-el"
        monkeypatch.setattr(runtime_config.settings, "get", lambda k, d=None: 1 if k == "retry_nums" else d)

        queue = [
            {"text": "Hello 1", "filename": str(tmp_path / "e1.wav"), "role": "Rachel"},
            {"text": "Hello 2", "filename": str(tmp_path / "e2.wav"), "role": "Rachel"},
        ]
        inst = _elevenlabs.ElevenLabsC(queue_tts=queue, language="en", tts_type=tts.ELEVENLABS_TTS, uuid="el-opt")
        inst.proxy_str = "http://127.0.0.1:7890"
        inst.run()

        assert len(created_clients) == 1
        assert created_proxies == ["http://127.0.0.1:7890"]
        assert created_clients[0][1].closed is True
        assert not (tmp_path / "e1.wav.mp3").exists()
        assert not (tmp_path / "e2.wav.mp3").exists()

    def test_gemini_key_rotation_api_error_handling_and_temp_cleanup(self, tmp_path: Path, monkeypatch):
        from google.genai import errors
        from videotrans import tts
        from videotrans.configure import config as runtime_config
        from videotrans.configure.excepts import StopTask
        from videotrans.tts import _geminitts

        seen_keys = []
        created_clients = []

        class SpyClient:
            def __init__(self, api_key="", http_options=None):
                created_clients.append(api_key)
                self.closed = False
                self.models = SimpleNamespace(
                    generate_content=lambda **kw: SimpleNamespace(
                        candidates=[
                            SimpleNamespace(
                                content=SimpleNamespace(
                                    parts=[
                                        SimpleNamespace(
                                            inline_data=SimpleNamespace(
                                                data=b"\x01\x00" * 1200,
                                                mime_type="audio/L16;rate=24000",
                                            )
                                        )
                                    ]
                                )
                            )
                        ]
                    )
                )

            def close(self):
                self.closed = True

        monkeypatch.setattr(_geminitts.genai, "Client", SpyClient)
        monkeypatch.setattr(
            _geminitts.GEMINITTS,
            "convert_to_wav",
            lambda self, src, dst: _make_tone_wav(Path(dst), duration_s=0.3, sr=48000),
        )
        runtime_config.params["gemini_key"] = "k1, k2"

        inst = _geminitts.GEMINITTS(
            queue_tts=[
                {"text": "One", "filename": str(tmp_path / "g1.wav"), "role": "Puck"},
                {"text": "Two", "filename": str(tmp_path / "g2.wav"), "role": "Puck"},
            ],
            language="en",
            tts_type=tts.GEMINI_TTS,
            uuid="gem-rot",
        )
        seen_keys.extend([inst._next_api_key(), inst._next_api_key(), inst._next_api_key()])
        assert seen_keys == ["k1", "k2", "k1"]
        inst._exec()
        assert not (tmp_path / "g1.wav.wav").exists()
        assert not (tmp_path / "g2.wav.wav").exists()

        # Non-retryable APIError 403 raises StopTask
        class Fail403Client:
            def __init__(self, api_key="", http_options=None):
                self.models = SimpleNamespace(
                    generate_content=lambda **kw: (_ for _ in ()).throw(
                        errors.APIError(403, {"error": {"message": "forbidden", "status": "PERMISSION_DENIED"}})
                    )
                )

        monkeypatch.setattr(_geminitts.genai, "Client", Fail403Client)
        inst_err = _geminitts.GEMINITTS(
            queue_tts=[{"text": "Fail", "filename": str(tmp_path / "gfail.wav"), "role": "Puck"}],
            language="en",
            tts_type=tts.GEMINI_TTS,
            uuid="gem-403",
        )
        with pytest.raises(StopTask):
            inst_err._run(inst_err.queue_tts[0], 0)

    def test_vieneu_tts_uses_model_cache_without_closing_and_memoizes_options(self, tmp_path: Path, monkeypatch):
        import sys
        from videotrans.services import model_cache
        from videotrans.tts import _vieneutts

        class CachedFakeEngine:
            sample_rate = 24000
            max_batch_size = 4

            def __init__(self):
                self.closed = False
                self.calls = []

            def get_preset_voice(self, role):
                return {"preset": role}

            def infer_batch(self, texts, **opts):
                self.calls.append((list(texts), opts))
                return [np.ones(480, dtype=np.float32) * 0.2 for _ in texts]

            def close(self):
                self.closed = True

        fake_engine = CachedFakeEngine()
        # Ensure sys.modules["vieneu"] is a real ModuleType so VieNeuTTS uses model_cache
        if not isinstance(sys.modules.get("vieneu"), types.ModuleType):
            fake_mod = types.ModuleType("vieneu")
            fake_mod.Vieneu = lambda **kw: fake_engine
            monkeypatch.setitem(sys.modules, "vieneu", fake_mod)

        monkeypatch.setattr(model_cache, "get_cached_vieneu_engine", lambda **kw: fake_engine)
        resolve_calls = {"n": 0}
        from videotrans.core import voice_store
        orig_resolve = voice_store.resolve_voice_params

        def spy_resolve(role, *a, **kw):
            resolve_calls["n"] += 1
            return orig_resolve(role, *a, **kw)

        monkeypatch.setattr(voice_store, "resolve_voice_params", spy_resolve)

        queue = [
            {"text": f"Line {i}", "role": "Mai Anh", "filename": str(tmp_path / f"v_{i}.wav")}
            for i in range(5)
        ]
        vt = _vieneutts.VieNeuTTS(queue_tts=queue, language="vi", is_cuda=False, uuid="vn-cache")
        vt._exec()
        assert fake_engine.closed is False, "Cached engine must NOT be closed by VieNeuTTS._exec()"
        assert resolve_calls["n"] == 1, f"Expected memoized _inference_options across 5 items, got {resolve_calls['n']}"
        assert [len(c[0]) for c in fake_engine.calls] == [4, 1]

    def test_omnivoice_caches_roles_and_cleans_temp_files(self, tmp_path: Path, monkeypatch):
        from videotrans import tts
        from videotrans.process import omnivoice_tts
        from videotrans.tts import _omnivoice
        from videotrans.util import help_role

        ref_wav = _make_tone_wav(tmp_path / "ref.wav", duration_s=0.4, sr=24000)
        out1 = tmp_path / "o1.wav"
        out2 = tmp_path / "o2.wav"
        q_file = tmp_path / "q.json"
        q_file.write_text(
            json.dumps([
                {"text": "A", "role": "SpeakerA", "filename": str(out1)},
                {"text": "B", "role": "SpeakerA", "filename": str(out2)},
            ]),
            encoding="utf-8",
        )

        role_info_calls = {"n": 0}

        def spy_voice_info(role):
            role_info_calls["n"] += 1
            return str(ref_wav), "ref"

        monkeypatch.setattr(help_role, "get_omnivoice_voice_info", spy_voice_info)
        monkeypatch.setattr(
            help_role,
            "get_f5tts_role",
            lambda: (_ for _ in ()).throw(AssertionError("get_f5tts_role must not be called in loop")),
        )

        class FakeOmniModel:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def generate(self, **kw):
                return [np.ones(2400, dtype=np.float32) * 0.2]

        fake_omni_mod = types.ModuleType("omnivoice")
        fake_omni_mod.OmniVoice = FakeOmniModel
        monkeypatch.setitem(sys.modules, "omnivoice", fake_omni_mod)

        ok, err = omnivoice_tts.omnivoice_fun(
            queue_tts_file=str(q_file),
            logs_file=str(tmp_path / "omni.log"),
            is_cuda=False,
            is_redubb=False,
        )
        assert ok is True and err is None
        assert role_info_calls["n"] == 1
        assert (Path(str(out1) + "-24k.wav")).is_file()

        # OmniVoice._exec cleans -24k.wav and omnivoice-*.json
        from videotrans.configure import config as runtime_config
        omni_temp_root = tmp_path / "omni_temp"
        omni_temp_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(runtime_config, "TEMP_DIR", str(omni_temp_root))
        monkeypatch.setattr(_omnivoice, "TEMP_DIR", str(omni_temp_root))
        omni_uuid = f"omni-clean-{uuid.uuid4().hex[:8]}"

        inst = _omnivoice.OmniVoice(
            queue_tts=[
                {"text": "A", "role": "SpeakerA", "filename": str(out1)},
                {"text": "B", "role": "SpeakerA", "filename": str(out2)},
            ],
            language="en",
            tts_type=tts.OMNIVOICE_TTS,
            uuid=omni_uuid,
        )
        monkeypatch.setattr(inst, "_new_process", lambda **kw: None)
        monkeypatch.setattr(inst, "convert_to_wav", lambda src, dst: _make_tone_wav(Path(dst), duration_s=0.3, sr=48000))
        inst._exec()
        assert not Path(str(out1) + "-24k.wav").exists()
        assert not Path(str(out2) + "-24k.wav").exists()
        assert list((omni_temp_root / omni_uuid).glob("omnivoice-*.json")) == []

    def test_speed_rate_header_duration_mixed_rates_and_zero_temp_leaks(self, tmp_path: Path, monkeypatch):
        from pydub import AudioSegment
        from videotrans.task import _rate

        cache_dir = tmp_path / "rate_cache"
        cache_dir.mkdir()
        c1 = _make_tone_wav(cache_dir / "c1.wav", duration_s=1.0, sr=24000, freq=440.0, channels=1)
        c2 = _make_tone_wav(cache_dir / "c2.wav", duration_s=0.8, sr=44100, freq=550.0, channels=2)
        target_audio = tmp_path / "rate_out.wav"

        # Ensure AudioSegment.from_file is NEVER called for duration reads
        monkeypatch.setattr(
            AudioSegment,
            "from_file",
            staticmethod(lambda *a, **kw: (_ for _ in ()).throw(AssertionError("AudioSegment.from_file should not be called"))),
        )

        assert _rate.get_audio_duration_ms(c1) == pytest.approx(1000, abs=5)

        sr_inst = _rate.SpeedRate(
            queue_tts=[
                {"line": 1, "start_time": 0, "end_time": 1200, "filename": str(c1), "text": "One"},
                {"line": 2, "start_time": 1800, "end_time": 2600, "filename": str(c2), "text": "Two"},
                {"line": 3, "start_time": 2700, "end_time": 3200, "filename": "", "text": "Missing"},
            ],
            should_videorate=False,
            should_audiorate=True,
            uuid="rate-no-pydub",
            raw_total_time=3500,
            target_audio=str(target_audio),
            cache_folder=str(cache_dir),
        )
        sr_inst.run()
        assert target_audio.is_file()
        info = sf.info(str(target_audio))
        assert info.samplerate == 48000
        assert info.channels == 2
        assert info.subtype == "PCM_16"
        assert list(cache_dir.glob("silence_*.wav")) == []
        assert list(cache_dir.glob("silent_place_*.wav")) == []
        assert list(cache_dir.glob("*-after.wav")) == []

    def test_previews_dir_and_dubbing_cache_lru_ttl_eviction(self, tmp_path: Path, monkeypatch):
        from videotrans.core import voice_store
        from videotrans.services import voice_preview
        from videotrans.tts._cache import prune_cache_dir

        # 1. PREVIEWS_DIR eviction preserves voice_*_preview.wav & removes stale + LRU + paired .json
        prev_dir = tmp_path / "previews"
        prev_dir.mkdir()
        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", prev_dir, raising=False)

        saved_voice_prev = _make_tone_wav(prev_dir / "voice_1234abcd_preview.wav", duration_s=0.3)
        stale = _make_tone_wav(prev_dir / "prev_000000000001.wav", duration_s=0.3)
        stale_json = prev_dir / "prev_000000000001.json"
        stale_json.write_text('{"applied_speed": 1.0}', encoding="utf-8")
        old_ts = time.time() - 10 * 86400
        os.utime(saved_voice_prev, (old_ts, old_ts))
        os.utime(stale, (old_ts, old_ts))
        os.utime(stale_json, (old_ts, old_ts))

        recent1 = _make_tone_wav(prev_dir / "prev_000000000002.wav", duration_s=0.3)
        recent1_json = prev_dir / "prev_000000000002.json"
        recent1_json.write_text('{"applied_speed": 1.0}', encoding="utf-8")
        recent2 = _make_tone_wav(prev_dir / "prev_000000000003.wav", duration_s=0.3)
        recent2_json = prev_dir / "prev_000000000003.json"
        recent2_json.write_text('{"applied_speed": 1.0}', encoding="utf-8")
        now = time.time()
        os.utime(recent1, (now - 100, now - 100))
        os.utime(recent2, (now - 10, now - 10))

        removed = prune_cache_dir(
            prev_dir,
            max_age_seconds=7 * 86400,
            max_bytes=10 * 1024 * 1024,
            max_files=1,
            primary_suffixes=(".wav",),
            paired_suffixes=(".json",),
            preserve_prefixes=("voice_",),
            preserve_suffixes=("_preview.wav",),
        )
        assert removed == 2
        assert saved_voice_prev.exists()
        assert not stale.exists() and not stale_json.exists()
        assert not recent1.exists() and not recent1_json.exists()
        assert recent2.exists() and recent2_json.exists()

        # 2. cleanup_previews_dir dynamically resolves monkeypatched PREVIEWS_DIR
        extra_stale = _make_tone_wav(prev_dir / "prev_aaaaaaaaaaaa.wav", duration_s=0.3)
        os.utime(extra_stale, (old_ts, old_ts))
        assert voice_preview.cleanup_previews_dir() == 1
        assert not extra_stale.exists()


class TestM5Iteration2Regressions:
    def test_sidecar_atomic_vs_cleanup_200_iterations_and_temp_skip(self, tmp_path: Path, monkeypatch):
        from videotrans.core import voice_store
        from videotrans.services import voice_preview
        from videotrans.tts._cache import prune_cache_dir

        prev_dir = tmp_path / "previews"
        prev_dir.mkdir()
        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", prev_dir, raising=False)

        # In-flight .tmp files and recent orphaned .json sidecars (<60s) must be untouched
        inflight_tmp = prev_dir / ".prev_deadbeef1234.abc123.tmp"
        inflight_tmp.write_text('{"applied_speed": 1.0}', encoding="utf-8")
        recent_orphan = prev_dir / "prev_999999999999.json"
        recent_orphan.write_text('{"applied_speed": 1.0}', encoding="utf-8")
        old_orphan = prev_dir / "prev_888888888888.json"
        old_orphan.write_text('{"applied_speed": 1.0}', encoding="utf-8")
        old_ts = time.time() - 120.0
        os.utime(old_orphan, (old_ts, old_ts))

        prune_cache_dir(
            prev_dir,
            max_age_seconds=7 * 86400,
            max_bytes=256 * 1024 * 1024,
            max_files=500,
            primary_suffixes=(".wav",),
            paired_suffixes=(".json",),
        )
        assert inflight_tmp.exists()
        assert recent_orphan.exists()
        assert not old_orphan.exists()

        # 200-iteration concurrent stress test: _write_sidecar_atomic vs cleanup_previews_dir
        errors: list[tuple[str, int, str]] = []
        wav_paths = [_make_tone_wav(prev_dir / f"prev_{i:012x}.wav", duration_s=0.15) for i in range(8)]

        def writer():
            for i in range(200):
                try:
                    voice_preview._write_sidecar_atomic(wav_paths[i % len(wav_paths)], 1.0 + (i % 3) * 0.1)
                except Exception as exc:
                    errors.append(("writer", i, repr(exc)))

        def cleaner():
            for i in range(200):
                try:
                    voice_preview.cleanup_previews_dir()
                except Exception as exc:
                    errors.append(("cleaner", i, repr(exc)))

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=cleaner)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        assert errors == [], f"Concurrent sidecar/cleanup errors: {errors[:5]}"

    def test_concurrent_api_tts_preview_no_winerror5(self, tmp_path: Path, monkeypatch):
        from concurrent.futures import ThreadPoolExecutor
        from videotrans.api.app import app
        from videotrans.configure import config as runtime_config
        from videotrans.core import voice_store
        from videotrans.services import voice_preview

        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", tmp_path / "previews", raising=False)
        voice_preview.set_preview_synthesizer(None)
        prev_el_key = runtime_config.params.get("elevenlabstts_key", "")
        runtime_config.params["elevenlabstts_key"] = "fake-el-key"

        def fake_el(preview_path: Path, sample_text: str, target_voice: str, speed=None):
            time.sleep(0.02)
            _make_tone_wav(Path(preview_path), duration_s=0.25, sr=24000, freq=440.0)

        monkeypatch.setattr(voice_preview, "_run_elevenlabs_preview", fake_el)

        try:
            with TestClient(app) as client:
                def call_one(i: int):
                    return client.post(
                        "/api/tts/preview",
                        json={
                            "provider": "elevenlabs",
                            "voice": "Rachel",
                            "text": f"Concurrent preview test {i}",
                            "language": "en",
                            "forceRefresh": True,
                        },
                    )

                with ThreadPoolExecutor(max_workers=6) as pool:
                    responses = list(pool.map(call_one, range(24)))
                assert all(r.status_code == 200 and r.json().get("ok") is True for r in responses), [
                    (r.status_code, r.text) for r in responses if r.status_code != 200
                ]
        finally:
            runtime_config.params["elevenlabstts_key"] = prev_el_key

    def test_multigroup_vieneu_batch_cancel_and_duplicate_fast_path(self, tmp_path: Path, monkeypatch):
        from videotrans.core import voice_store
        from videotrans.services import model_cache, voice_preview

        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", tmp_path / "previews", raising=False)
        voice_preview.set_preview_synthesizer(None)
        monkeypatch.setattr("videotrans.util.help_role.get_vieneu_custom_voice_path", lambda v: None)

        cancel_ev = threading.Event()

        class CancelMidEngine:
            sample_rate = 24000
            max_batch_size = 4

            def get_preset_voice(self, name):
                return {"preset": name}

            def infer_batch(self, texts, **kwargs):
                cancel_ev.set()
                t = np.linspace(0.0, 0.25, int(0.25 * 24000), endpoint=False, dtype=np.float32)
                tone = 0.4 * np.sin(2 * np.pi * 440.0 * t)
                return [tone.copy() for _ in texts]

        monkeypatch.setattr(model_cache, "get_cached_vieneu_engine", lambda **kw: CancelMidEngine())

        items = [
            *[{"segment_id": f"v1_{i}", "text": f"G1 {i}", "voice": "Mai Anh", "provider": 2, "force_refresh": True} for i in range(5)],
            *[{"segment_id": f"v2_{i}", "text": f"G2 {i}", "voice": "Thanh Bình", "provider": 2, "force_refresh": True} for i in range(3)],
        ]
        res = asyncio.run(voice_preview.synthesize_tts_batch(items, cancel_event=cancel_ev))
        assert len(res) == 8
        by_id = {r["segmentId"]: r for r in res}
        assert set(by_id.keys()) == {f"v1_{i}" for i in range(5)} | {f"v2_{i}" for i in range(3)}
        # Group 1 item 4 and all of Group 2 (v2_0..v2_2) must be Cancelled with real segmentId/voice/provider
        assert by_id["v1_4"]["ok"] is False and by_id["v1_4"]["error"] == "Cancelled"
        for i in range(3):
            item_res = by_id[f"v2_{i}"]
            assert item_res["ok"] is False
            assert item_res["error"] == "Cancelled"
            assert item_res["voice"] == "Thanh Bình"
            assert item_res["provider"] == 2

        # Duplicate preview_path items under _preview_synthesizer fast path must both succeed
        def fake_synth(voice_dict, preview_path: Path, sample_text: str):
            _make_tone_wav(Path(preview_path), duration_s=0.25, sr=24000)

        voice_preview.set_preview_synthesizer(fake_synth)
        try:
            dup_res = asyncio.run(
                voice_preview.synthesize_tts_batch([
                    {"segment_id": "dupA", "text": "Same line", "voice": "Mai Anh", "provider": 2},
                    {"segment_id": "dupB", "text": "Same line", "voice": "Mai Anh", "provider": 2},
                ])
            )
            assert [r["segmentId"] for r in dup_res] == ["dupA", "dupB"]
            assert all(r["ok"] is True and r["previewUrl"] for r in dup_res)
        finally:
            voice_preview.set_preview_synthesizer(None)

    def test_gemini_tts_empty_candidates_and_parts_guard(self, tmp_path: Path, monkeypatch):
        from videotrans import tts
        from videotrans.configure import config as runtime_config
        from videotrans.tts import _geminitts

        chunks = [
            SimpleNamespace(candidates=[]),
            SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))]),
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    inline_data=SimpleNamespace(
                                        data=b"\x01\x00" * 600,
                                        mime_type="audio/L16;rate=24000",
                                    )
                                )
                            ]
                        )
                    )
                ]
            ),
        ]

        class EmptyCandClient:
            def __init__(self, api_key="", http_options=None):
                self.models = SimpleNamespace(generate_content_stream=lambda **kw: iter(chunks))

            def close(self):
                pass

        monkeypatch.setattr(_geminitts.genai, "Client", EmptyCandClient)
        runtime_config.params["gemini_key"] = "k-empty-cand"
        inst = _geminitts.GEMINITTS(
            queue_tts=[{"text": "Hi", "filename": str(tmp_path / "g.wav"), "role": "Puck"}],
            language="en",
            tts_type=tts.GEMINI_TTS,
            uuid="gem-empty-cand",
        )
        out = tmp_path / "out.wav"
        inst.generate_tts_segment("Hi", "Puck", "gemini-2.5-flash-preview-tts", str(out))
        assert out.is_file() and out.stat().st_size > 44

    def test_concurrent_dubbing_assemble_same_project_no_winerror32(self, tmp_path: Path, monkeypatch):
        from concurrent.futures import ThreadPoolExecutor
        from videotrans.api.app import app
        from videotrans.core import project_store, voice_store

        projects_dir = tmp_path / "projects"
        previews_dir = tmp_path / "previews"
        monkeypatch.setattr(project_store, "DEFAULT_PROJECTS_DIR", projects_dir, raising=False)
        monkeypatch.setattr(project_store, "PROJECTS_OUTPUT_DIR", projects_dir, raising=False)
        monkeypatch.setattr(project_store, "get_projects_root", lambda: projects_dir)
        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", previews_dir, raising=False)
        voice_store.init_voice_dirs()

        project_store.create_project(name="Conc Assemble", project_id="conc-asm")
        project_store.update_project("conc-asm", duration=2.0)
        pid = "prev_0000000000ab"
        _make_tone_wav(previews_dir / f"{pid}.wav", duration_s=0.4, sr=24000)
        payload = {
            "segments": [
                {"id": 1, "startSec": 0.0, "endSec": 0.5, "targetText": "One", "previewAudioId": pid},
                {"id": 2, "startSec": 0.8, "endSec": 1.3, "targetText": "Two", "previewAudioId": pid},
            ]
        }

        with TestClient(app) as client:
            def run_asm(_):
                return client.post("/api/projects/conc-asm/dubbing/assemble", json=payload)

            with ThreadPoolExecutor(max_workers=6) as pool:
                responses = list(pool.map(run_asm, range(12)))
        assert all(r.status_code == 200 and r.json().get("ok") is True for r in responses), [
            (r.status_code, r.text) for r in responses if r.status_code != 200
        ]
        dubbing_dir = project_store.get_project_dir("conc-asm") / "dubbing"
        assert (dubbing_dir / "voiceover_merged.wav").is_file()
        assert list(dubbing_dir.glob("*.tmp.wav")) == []

    def test_batch_route_propagates_auto_speed_and_rejects_invalid_provider(self, monkeypatch):
        from videotrans.api.app import app
        from videotrans.api.routes import voices as voices_route

        captured: list[list[dict]] = []

        async def fake_batch(items, **kwargs):
            captured.append(items)
            return [
                {
                    "segmentId": it["segment_id"],
                    "ok": True,
                    "id": f"prev_{idx:012x}",
                    "previewUrl": f"/api/tts/preview/prev_{idx:012x}/audio",
                    "appliedSpeed": 1.0,
                }
                for idx, it in enumerate(items)
            ]

        monkeypatch.setattr(voices_route, "synthesize_tts_batch", fake_batch)

        with TestClient(app) as client:
            # 1. Batch-level autoSpeed / maxSpeedRate propagate when omitted on items
            r_ok = client.post("/api/tts/batch", json={
                "provider": 2,
                "autoSpeed": True,
                "maxSpeedRate": 1.85,
                "segments": [
                    {"segmentId": 1, "text": "Inherit", "voice": "Mai Anh"},
                    {"segmentId": 2, "text": "Override", "voice": "Mai Anh", "autoSpeed": False, "maxSpeedRate": 1.15},
                ],
            })
            assert r_ok.status_code == 200, r_ok.text
            assert captured[0][0]["auto_speed"] is True
            assert captured[0][0]["max_speed_rate"] == pytest.approx(1.85)
            assert captured[0][1]["auto_speed"] is False
            assert captured[0][1]["max_speed_rate"] == pytest.approx(1.15)

            # 2. Invalid explicit providers are rejected with 400 (never silently coerced to VieNeu)
            for bad_prov in (999, -1, "nonexistent-tts", True):
                r_bad = client.post("/api/tts/batch", json={
                    "provider": bad_prov,
                    "segments": [{"segmentId": 1, "text": "Hi", "voice": "Mai Anh"}],
                })
                assert r_bad.status_code == 400, f"Expected 400 for batch provider={bad_prov!r}, got {r_bad.status_code}"

            r_bad_item = client.post("/api/tts/batch", json={
                "segments": [{"segmentId": 1, "text": "Hi", "voice": "Mai Anh", "provider": 999}],
            })
            assert r_bad_item.status_code == 400

            r_bad_prev = client.post("/api/tts/preview", json={
                "provider": 999,
                "voice": "Mai Anh",
                "text": "Hi",
            })
            assert r_bad_prev.status_code == 400


class TestM5Iteration3Regressions:
    def test_concurrent_vieneu_batch_and_preview_auto_speed_shared_path(self, tmp_path: Path, monkeypatch):
        from videotrans import tts
        from videotrans.core import voice_store
        from videotrans.services import model_cache, voice_preview

        prev_dir = tmp_path / "previews"
        prev_dir.mkdir()
        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", prev_dir, raising=False)
        voice_preview.set_preview_synthesizer(None)
        monkeypatch.setattr("videotrans.util.help_role.get_vieneu_custom_voice_path", lambda v: None)

        class SharedPathEngine:
            sample_rate = 24000
            max_batch_size = 4

            def get_preset_voice(self, name):
                return {"preset": name}

            def infer_batch(self, texts, **kwargs):
                # 0.9s tone so slot_duration_s=0.6 with max_speed_rate=1.5 produces appliedSpeed=1.5
                n = int(0.9 * self.sample_rate)
                t = np.linspace(0.0, 0.9, n, endpoint=False, dtype=np.float32)
                tone = 0.35 * np.sin(2.0 * np.pi * 440.0 * t)
                return [tone.copy() for _ in texts]

        monkeypatch.setattr(model_cache, "get_cached_vieneu_engine", lambda **kw: SharedPathEngine())

        async def _run():
            batch_coros = [
                voice_preview.synthesize_tts_batch(
                    [
                        {
                            "segmentId": f"b{i}",
                            "provider": tts.VIENEU_TTS,
                            "voice": "Bình (nam miền Bắc)",
                            "text": "Shared concurrent text",
                            "force_refresh": True,
                            "auto_speed": True,
                            "slot_duration_s": 0.6,
                            "max_speed_rate": 1.5,
                        }
                    ],
                    use_cuda=False,
                )
                for i in range(6)
            ]
            preview_coros = [
                voice_preview.synthesize_unified_tts_preview(
                    provider=tts.VIENEU_TTS,
                    voice="Bình (nam miền Bắc)",
                    text="Shared concurrent text",
                    force_refresh=True,
                    use_cuda=False,
                    auto_speed=True,
                    slot_duration_s=0.6,
                    max_speed_rate=1.5,
                )
                for _ in range(3)
            ]
            all_results = await asyncio.gather(*batch_coros, *preview_coros)
            return all_results[:6], all_results[6:]

        batch_results, preview_results = asyncio.run(_run())
        assert len(batch_results) == 6
        for idx, res_list in enumerate(batch_results):
            assert len(res_list) == 1
            item = res_list[0]
            assert item["ok"] is True, f"batch[{idx}] failed: {item}"
            assert item["appliedSpeed"] == pytest.approx(1.5, abs=0.05)
            assert item["durationSec"] == pytest.approx(0.6, abs=0.08)

        for prev_id, prev_path, spd in preview_results:
            assert prev_id.startswith("prev_")
            assert Path(prev_path).is_file()
            assert spd == pytest.approx(1.5, abs=0.05)

        leftover_tmps = [
            p.name for p in prev_dir.iterdir()
            if p.name.startswith(".") or ".tmp" in p.name.lower() or p.name.lower().startswith("tmp")
        ]
        assert leftover_tmps == [], f"Leftover temp files found: {leftover_tmps}"

    def test_gemini_tts_extracts_audio_after_non_audio_leading_part(self, tmp_path: Path, monkeypatch):
        from videotrans import tts
        from videotrans.configure import config as runtime_config
        from videotrans.tts import _geminitts

        audio_part = SimpleNamespace(
            inline_data=SimpleNamespace(
                data=b"\x01\x00" * 600,
                mime_type="audio/L16;rate=24000",
            )
        )
        chunks = [
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(inline_data=None, text="metadata"), audio_part]
                        )
                    )
                ]
            )
        ]

        class MultiPartClient:
            def __init__(self, api_key="", http_options=None):
                self.models = SimpleNamespace(generate_content_stream=lambda **kw: iter(chunks))

            def close(self):
                pass

        monkeypatch.setattr(_geminitts.genai, "Client", MultiPartClient)
        runtime_config.params["gemini_key"] = "k-multipart"
        inst = _geminitts.GEMINITTS(
            queue_tts=[{"text": "Hi", "filename": str(tmp_path / "g.wav"), "role": "Puck"}],
            language="en",
            tts_type=tts.GEMINI_TTS,
            uuid="gem-multipart",
        )
        out = tmp_path / "out_multipart.wav"
        inst.generate_tts_segment("Hi", "Puck", "gemini-2.5-flash-preview-tts", str(out))
        assert out.is_file() and out.stat().st_size > 44


class TestM5Iteration4Regressions:
    def test_audio_fit_concurrent_unlocked_reader_and_prune_stress(self, tmp_path: Path, monkeypatch):
        from videotrans import tts
        from videotrans.core import voice_store
        from videotrans.services import audio_fit, model_cache, voice_preview
        from videotrans.tts._cache import prune_cache_dir

        # Part A: direct apply_pitch_preserving_stretch vs concurrent Windows reader + prune_cache_dir
        prev_dir = tmp_path / "fit_stress"
        prev_dir.mkdir(parents=True, exist_ok=True)
        src_wav = _make_tone_wav(prev_dir / "src.wav", duration_s=0.9, sr=24000)
        target_wav = _make_tone_wav(prev_dir / "target.wav", duration_s=0.9, sr=24000)
        for i in range(3):
            filler = _make_tone_wav(prev_dir / f"filler_{i}.wav", duration_s=0.25, sr=24000)
            old_ts = time.time() - 3600 - i
            os.utime(filler, (old_ts, old_ts))

        stop_ev = threading.Event()
        reader_opens = {"n": 0}
        errors: list[str] = []

        def reader_worker():
            while not stop_ev.is_set():
                try:
                    with open(target_wav, "rb") as fh:
                        fh.read(64)
                        reader_opens["n"] += 1
                        time.sleep(0.008)
                except OSError:
                    pass
                time.sleep(0.015)

        def pruner_worker():
            while not stop_ev.is_set():
                try:
                    prune_cache_dir(
                        prev_dir,
                        max_age_seconds=86400,
                        max_bytes=10 * 1024 * 1024,
                        max_files=2,
                        primary_suffixes=(".wav",),
                    )
                except Exception as exc:
                    errors.append(f"pruner: {exc!r}")
                time.sleep(0.01)

        rt = threading.Thread(target=reader_worker, daemon=True)
        pt = threading.Thread(target=pruner_worker, daemon=True)
        rt.start()
        pt.start()
        try:
            for _ in range(6):
                assert audio_fit.apply_pitch_preserving_stretch(src_wav, target_wav, speed_factor=1.5) is True
        finally:
            stop_ev.set()
            rt.join(timeout=2.0)
            pt.join(timeout=2.0)

        assert errors == []
        assert reader_opens["n"] > 0
        assert audio_fit.get_audio_duration_seconds(target_wav) == pytest.approx(0.6, abs=0.08)
        assert [p.name for p in prev_dir.iterdir() if p.name.startswith(".") or ".tmp" in p.name.lower()] == []

        # Part B: mixed force_refresh=False + force_refresh=True + PREVIEWS_MAX_FILES=2
        e2e_prev = tmp_path / "e2e_previews"
        e2e_prev.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", e2e_prev, raising=False)
        monkeypatch.setattr(voice_preview, "PREVIEWS_MAX_FILES", 2)
        voice_preview.set_preview_synthesizer(None)
        monkeypatch.setattr("videotrans.util.help_role.get_vieneu_custom_voice_path", lambda v: None)

        class FitStressEngine:
            sample_rate = 24000
            max_batch_size = 4

            def get_preset_voice(self, name):
                return {"preset": name}

            def infer_batch(self, texts, **kwargs):
                n = int(0.9 * self.sample_rate)
                t = np.linspace(0.0, 0.9, n, endpoint=False, dtype=np.float32)
                tone = 0.35 * np.sin(2.0 * np.pi * 440.0 * t)
                return [tone.copy() for _ in texts]

        monkeypatch.setattr(model_cache, "get_cached_vieneu_engine", lambda **kw: FitStressEngine())

        async def _run_mixed():
            # Pre-warm shared entry
            warm = await voice_preview.synthesize_tts_batch(
                [{
                    "segmentId": "warm",
                    "provider": tts.VIENEU_TTS,
                    "voice": "Bình (nam miền Bắc)",
                    "text": "Mixed reader writer text",
                    "force_refresh": False,
                    "auto_speed": True,
                    "slot_duration_s": 0.6,
                    "max_speed_rate": 1.5,
                }],
                use_cuda=False,
            )
            assert warm[0]["ok"] is True

            coros = []
            for i in range(6):
                coros.append(voice_preview.synthesize_tts_batch(
                    [{
                        "segmentId": f"r{i}",
                        "provider": tts.VIENEU_TTS,
                        "voice": "Bình (nam miền Bắc)",
                        "text": "Mixed reader writer text",
                        "force_refresh": False,
                        "auto_speed": True,
                        "slot_duration_s": 0.6,
                        "max_speed_rate": 1.5,
                    }],
                    use_cuda=False,
                ))
            for i in range(4):
                coros.append(voice_preview.synthesize_tts_batch(
                    [{
                        "segmentId": f"w{i}",
                        "provider": tts.VIENEU_TTS,
                        "voice": "Bình (nam miền Bắc)",
                        "text": "Mixed reader writer text",
                        "force_refresh": True,
                        "auto_speed": True,
                        "slot_duration_s": 0.6,
                        "max_speed_rate": 1.5,
                    }],
                    use_cuda=False,
                ))
            for _ in range(4):
                coros.append(voice_preview.synthesize_unified_tts_preview(
                    provider=tts.VIENEU_TTS,
                    voice="Bình (nam miền Bắc)",
                    text="Mixed reader writer text",
                    force_refresh=False,
                    use_cuda=False,
                    auto_speed=True,
                    slot_duration_s=0.6,
                    max_speed_rate=1.5,
                ))
            return await asyncio.gather(*coros)

        mixed_res = asyncio.run(_run_mixed())
        for res in mixed_res[:10]:
            assert len(res) == 1 and res[0]["ok"] is True
            assert res[0]["appliedSpeed"] == pytest.approx(1.5, abs=0.05)
            assert res[0]["durationSec"] == pytest.approx(0.6, abs=0.08)
        for pid, ppath, spd in mixed_res[10:]:
            assert pid.startswith("prev_") and Path(ppath).is_file()
            assert spd == pytest.approx(1.5, abs=0.05)
        leftover_tmps = [
            p.name for p in e2e_prev.iterdir()
            if p.name.startswith(".") or ".tmp" in p.name.lower() or p.name.lower().startswith("tmp")
        ]
        assert leftover_tmps == []

    def test_multi_loop_threaded_batch_and_audio_get_no_winerror(self, tmp_path: Path, monkeypatch):
        from concurrent.futures import ThreadPoolExecutor
        from videotrans import tts
        from videotrans.api.app import app
        from videotrans.core import voice_store
        from videotrans.services import model_cache, voice_preview

        prev_dir = tmp_path / "tc_previews"
        prev_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", prev_dir, raising=False)
        voice_preview.set_preview_synthesizer(None)
        monkeypatch.setattr("videotrans.util.help_role.get_vieneu_custom_voice_path", lambda v: None)

        class MultiLoopEngine:
            sample_rate = 24000
            max_batch_size = 4

            def get_preset_voice(self, name):
                return {"preset": name}

            def infer_batch(self, texts, **kwargs):
                n = int(0.9 * self.sample_rate)
                t = np.linspace(0.0, 0.9, n, endpoint=False, dtype=np.float32)
                tone = 0.35 * np.sin(2.0 * np.pi * 440.0 * t)
                return [tone.copy() for _ in texts]

        monkeypatch.setattr(model_cache, "get_cached_vieneu_engine", lambda **kw: MultiLoopEngine())

        # Pre-resolve preview_id by priming once
        with TestClient(app) as prime_client:
            prime_res = prime_client.post("/api/tts/batch", json={
                "segments": [{
                    "segmentId": "prime",
                    "provider": tts.VIENEU_TTS,
                    "voice": "Bình (nam miền Bắc)",
                    "text": "Probe 1B-TC shared text",
                    "forceRefresh": True,
                    "autoSpeed": True,
                    "slotDuration": 0.6,
                    "maxSpeedRate": 1.5,
                }],
            })
            assert prime_res.status_code == 200
            preview_id = prime_res.json()["results"][0]["id"]

        def post_worker(i: int):
            with TestClient(app) as client:
                r = client.post("/api/tts/batch", json={
                    "segments": [{
                        "segmentId": f"ml_{i}",
                        "provider": tts.VIENEU_TTS,
                        "voice": "Bình (nam miền Bắc)",
                        "text": "Probe 1B-TC shared text",
                        "forceRefresh": True,
                        "autoSpeed": True,
                        "slotDuration": 0.6,
                        "maxSpeedRate": 1.5,
                    }],
                })
                return r.status_code, r.json()

        def get_worker(_: int):
            with TestClient(app) as client:
                r = client.get(f"/api/tts/preview/{preview_id}/audio")
                return r.status_code, r.content[:4], len(r.content)

        with ThreadPoolExecutor(max_workers=16) as pool:
            post_futs = [pool.submit(post_worker, i) for i in range(8)]
            get_futs = [pool.submit(get_worker, i) for i in range(8)]
            post_results = [f.result() for f in post_futs]
            get_results = [f.result() for f in get_futs]

        for status_code, body in post_results:
            assert status_code == 200
            assert len(body["results"]) == 1
            item = body["results"][0]
            assert item["ok"] is True, f"Multi-loop POST failed: {item}"
            assert item["appliedSpeed"] == pytest.approx(1.5, abs=0.05)
            assert item["durationSec"] == pytest.approx(0.6, abs=0.08)

        for status_code, magic, size in get_results:
            assert status_code == 200
            assert magic == b"RIFF"
            assert size > 44

        leftover_tmps = [
            p.name for p in prev_dir.iterdir()
            if p.name.startswith(".") or ".tmp" in p.name.lower() or p.name.lower().startswith("tmp")
        ]
        assert leftover_tmps == []

    def test_cross_loop_path_lock_serializes_and_gcs(self, tmp_path: Path):
        from concurrent.futures import ThreadPoolExecutor
        from videotrans.services import voice_preview

        shared_path = tmp_path / "SharedLockFile.wav"
        guard = threading.Lock()
        state = {"active": 0, "max_active": 0, "count": 0}

        def worker(_: int):
            async def _inner():
                lock = voice_preview.get_path_lock(shared_path)
                async with lock:
                    with guard:
                        state["active"] += 1
                        state["max_active"] = max(state["max_active"], state["active"])
                        state["count"] += 1
                    await asyncio.sleep(0.01)
                    with guard:
                        state["active"] -= 1

            asyncio.run(_inner())

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(8)))

        assert state["count"] == 8
        assert state["max_active"] == 1
        gc.collect()
        resolved_key = os.path.normcase(str(shared_path.resolve()))
        assert resolved_key not in voice_preview._path_locks

    def test_task_cancel_mid_infer_batch_no_tmp_orphan_and_no_deadlock(self, tmp_path: Path, monkeypatch):
        from videotrans import tts
        from videotrans.core import voice_store
        from videotrans.services import model_cache, voice_preview

        prev_dir = tmp_path / "cancel_previews"
        prev_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", prev_dir, raising=False)
        voice_preview.set_preview_synthesizer(None)
        monkeypatch.setattr("videotrans.util.help_role.get_vieneu_custom_voice_path", lambda v: None)

        entered = threading.Event()
        release = threading.Event()
        worker_done = threading.Event()

        class BlockingEngine:
            sample_rate = 24000
            max_batch_size = 4

            def get_preset_voice(self, name):
                return {"preset": name}

            def infer_batch(self, texts, **kwargs):
                entered.set()
                release.wait(timeout=5.0)
                try:
                    n = int(0.3 * self.sample_rate)
                    t = np.linspace(0.0, 0.3, n, endpoint=False, dtype=np.float32)
                    tone = 0.35 * np.sin(2.0 * np.pi * 440.0 * t)
                    return [tone.copy() for _ in texts]
                finally:
                    worker_done.set()

        monkeypatch.setattr(model_cache, "get_cached_vieneu_engine", lambda **kw: BlockingEngine())

        async def _run():
            task = asyncio.create_task(
                voice_preview.synthesize_tts_batch(
                    [
                        {
                            "segmentId": f"c{i}",
                            "provider": tts.VIENEU_TTS,
                            "voice": "Mai Anh",
                            "text": f"Cancel mid infer {i}",
                            "force_refresh": True,
                        }
                        for i in range(4)
                    ],
                    use_cuda=False,
                )
            )
            while not entered.is_set():
                await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            release.set()
            while not worker_done.is_set():
                await asyncio.sleep(0.005)
            # Give the background thread a moment to finish post-infer abort check & finally cleanup
            await asyncio.sleep(0.05)

            # Follow-up batch must not deadlock on _gpu_async_lock or _gpu_thread_lock
            followup = await voice_preview.synthesize_tts_batch(
                [{
                    "segmentId": "after_cancel",
                    "provider": tts.VIENEU_TTS,
                    "voice": "Mai Anh",
                    "text": "Post cancel health check",
                    "force_refresh": True,
                }],
                use_cuda=False,
            )
            return followup

        followup = asyncio.run(_run())
        assert followup[0]["ok"] is True
        assert list(prev_dir.glob("*.tmp")) == []
        assert [p.name for p in prev_dir.iterdir() if p.name.startswith(".")] == []

    def test_wait_for_timeout_and_gather_fit_cancel_no_tmp_orphan(self, tmp_path: Path, monkeypatch):
        from videotrans import tts
        from videotrans.core import voice_store
        from videotrans.services import audio_fit, model_cache, voice_preview

        prev_dir = tmp_path / "timeout_fit_previews"
        prev_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(voice_store, "PREVIEWS_DIR", prev_dir, raising=False)
        voice_preview.set_preview_synthesizer(None)
        monkeypatch.setattr("videotrans.util.help_role.get_vieneu_custom_voice_path", lambda v: None)

        # (a) wait_for timeout while infer_batch takes longer than PREVIEW_TIMEOUT_SECONDS
        cond = threading.Condition()
        call_counts = {"entered": 0, "finished": 0}

        class SlowEngine:
            sample_rate = 24000
            max_batch_size = 4

            def get_preset_voice(self, name):
                return {"preset": name}

            def infer_batch(self, texts, **kwargs):
                with cond:
                    call_counts["entered"] += 1
                try:
                    time.sleep(0.18)
                    n = int(0.3 * self.sample_rate)
                    t = np.linspace(0.0, 0.3, n, endpoint=False, dtype=np.float32)
                    tone = 0.35 * np.sin(2.0 * np.pi * 440.0 * t)
                    return [tone.copy() for _ in texts]
                finally:
                    with cond:
                        call_counts["finished"] += 1
                        cond.notify_all()

        monkeypatch.setattr(model_cache, "get_cached_vieneu_engine", lambda **kw: SlowEngine())
        monkeypatch.setattr(voice_preview, "PREVIEW_TIMEOUT_SECONDS", 0.05)

        res_timeout = asyncio.run(
            voice_preview.synthesize_tts_batch(
                [
                    {
                        "segmentId": f"to_{i}",
                        "provider": tts.VIENEU_TTS,
                        "voice": "Mai Anh",
                        "text": f"Timeout item {i}",
                        "force_refresh": True,
                    }
                    for i in range(2)
                ],
                use_cuda=False,
            )
        )
        assert all(r["ok"] is False for r in res_timeout)
        with cond:
            cond.wait_for(lambda: call_counts["entered"] > 0 and call_counts["finished"] >= call_counts["entered"], timeout=5.0)
        time.sleep(0.05)
        assert list(prev_dir.glob("*.tmp")) == []
        assert [p.name for p in prev_dir.iterdir() if p.name.startswith(".")] == []

        # (b) Task.cancel() while gather(*fit_tasks) is suspended inside smart_fit_audio_file
        monkeypatch.setattr(voice_preview, "PREVIEW_TIMEOUT_SECONDS", 30.0)
        fit_entered = threading.Event()
        fit_release = threading.Event()

        class FastEngine:
            sample_rate = 24000
            max_batch_size = 4

            def get_preset_voice(self, name):
                return {"preset": name}

            def infer_batch(self, texts, **kwargs):
                n = int(0.6 * self.sample_rate)
                t = np.linspace(0.0, 0.6, n, endpoint=False, dtype=np.float32)
                tone = 0.35 * np.sin(2.0 * np.pi * 440.0 * t)
                return [tone.copy() for _ in texts]

        def slow_fit(in_p, slot_s, max_r=1.35, *args, **kwargs):
            fit_entered.set()
            fit_release.wait(timeout=5.0)
            return Path(in_p), 1.0, "fits"

        monkeypatch.setattr(model_cache, "get_cached_vieneu_engine", lambda **kw: FastEngine())
        monkeypatch.setattr(audio_fit, "smart_fit_audio_file", slow_fit)

        async def _run_fit_cancel():
            task = asyncio.create_task(
                voice_preview.synthesize_tts_batch(
                    [
                        {
                            "segmentId": f"fc_{i}",
                            "provider": tts.VIENEU_TTS,
                            "voice": "Mai Anh",
                            "text": f"Fit cancel {i}",
                            "force_refresh": True,
                            "auto_speed": True,
                            "slot_duration_s": 0.4,
                        }
                        for i in range(4)
                    ],
                    use_cuda=False,
                )
            )
            while not fit_entered.is_set():
                await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            fit_release.set()
            await asyncio.sleep(0.05)

        asyncio.run(_run_fit_cancel())
        assert list(prev_dir.glob("*.tmp")) == []
        assert [p.name for p in prev_dir.iterdir() if p.name.startswith(".")] == []
