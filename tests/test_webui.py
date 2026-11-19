import asyncio
import io
import threading
import time
from pathlib import Path

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from videotrans import recognition, translator, tts
from videotrans.api.app import create_app, main
from videotrans.api.provider_helpers import (
    ensure_asr_configured,
    ensure_translation_configured,
    test_asr_provider as run_asr_provider_test,
)
from videotrans.api.task_params import build_task_params
from videotrans.configure import config as runtime_config
from videotrans.configure.config import ROOT_DIR
from videotrans.core.job_manager import JobManager, JobRecord, run_prepare_review
from videotrans.task.orchestrator import (
    CancellationToken,
    EventKind,
    TaskEvent,
    TaskRequest,
    TaskResult,
    TaskStatus,
)


@pytest.fixture(autouse=True)
def isolate_test_db_and_env(tmp_path):
    from videotrans.core.db import get_db_path, init_db, set_db_path
    from videotrans.core import secret_store
    test_db = tmp_path / "test_webui.db"
    orig_db = get_db_path()
    set_db_path(test_db)
    init_db(test_db)
    secret_store.invalidate()
    yield
    set_db_path(orig_db)
    secret_store.invalidate()


def test_new_frontend_is_the_only_webui():
    app = create_app()
    routes = {route.resource.canonical for route in app.router.routes()}

    assert "/" in routes
    assert "/api/options" in routes
    assert "/api/media" in routes
    assert "/api/jobs" in routes
    assert "/api/jobs/{job_id}" in routes
    assert "/api/asr-settings/{provider_id}" in routes
    assert "/api/asr-settings/{provider_id}/test" in routes
    assert "/api/translation-settings/{provider_id}" in routes
    assert "/api/translation-settings/{provider_id}/test" in routes
    assert not any(route.startswith(("/js", "/css")) for route in routes)
    assert "gradio" not in (Path(ROOT_DIR) / "webui.py").read_text(encoding="utf-8").lower()


def test_frontend_static_and_html_have_no_cache_headers():
    app = create_app()

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            res_html = await client.get("/")
            assert res_html.status == 200
            assert "no-cache" in res_html.headers.get("Cache-Control", "")
            assert "no-store" in res_html.headers.get("Cache-Control", "")

            asset = next((Path(ROOT_DIR) / "frontend" / "dist" / "assets").glob("*.js"))
            res_js = await client.get(f"/assets/{asset.name}")
            assert res_js.status == 200
            assert "no-cache" in res_js.headers.get("Cache-Control", "")
            assert "no-store" in res_js.headers.get("Cache-Control", "")
        finally:
            await client.close()

    asyncio.run(scenario())


def test_main_accepts_reload_flag(monkeypatch):
    captured = {}
    monkeypatch.setattr("sys.argv", ["py", "--reload", "--port", "8765"])
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(app=app, **kwargs))

    main()

    assert captured["port"] == 8765
    assert captured["reload"] is True


def test_build_task_params_maps_supported_frontend_fields(tmp_path, monkeypatch):
    source = tmp_path / "sample video.mp4"
    source.write_bytes(b"video")
    temp_dir = tmp_path / "temp"
    output_dir = tmp_path / "output"
    temp_dir.mkdir()
    params = build_task_params(source, {
        "sourceLanguage": "en",
        "targetLanguage": "fr",
        "recognType": recognition.Deepgram,
        "translateType": translator.CHATGPT_INDEX,
        "ttsType": 3,
        "modelName": "nova-3",
        "timingMode": "video",
        "translationMode": "line",
        "speakerDiarization": True,
        "speakerCount": 2,
        "voiceRate": "+10%",
    }, job_type="asr", temp_dir=str(temp_dir), output_dir=output_dir)

    assert params["name"] == source.resolve().as_posix()
    assert params["source_language_code"] == "en"
    assert params["target_language_code"] == "fr"
    assert params["recogn_type"] == recognition.Deepgram
    assert params["model_name"] == "nova-3"
    assert params["translate_type"] == translator.CHATGPT_INDEX
    assert params["aisendsrt"] is False
    assert params["tts_type"] == 3
    assert params["voice_role"] == "No"
    assert "remove_noise" not in params
    assert params["enable_diariz"] is True
    assert params["nums_diariz"] == 2
    assert params["voice_rate"] == "+10%"
    assert params["voice_autorate"] is False
    assert params["video_autorate"] is False
    assert params["align_sub_audio"] is False
    assert params["subtitle_type"] == 0
    assert params["only_out_dubbed_audio"] is True
    assert params["embed_bgm"] is False
    assert Path(params["cache_folder"]).is_relative_to(temp_dir)


def test_build_task_params_rejects_unknown_backend_choices(tmp_path):
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video")

    with pytest.raises(ValueError, match="ASR engine"):
        build_task_params(source, {
            "sourceLanguage": next(iter(translator.LANGNAME_DICT)),
            "targetLanguage": next(iter(translator.LANGNAME_DICT)),
            "recognType": 9999,
            "translateType": 0,
            "ttsType": 0,
        })

    with pytest.raises(ValueError, match="translation mode"):
        build_task_params(source, {
            "sourceLanguage": "zh-cn",
            "targetLanguage": "vi",
            "recognType": recognition.Deepgram,
            "modelName": "nova-3",
            "translateType": translator.CHATGPT_INDEX,
            "translationMode": "invented",
            "ttsType": 0,
        })


def test_build_task_params_rejects_removed_provider_and_mismatched_model(tmp_path):
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video")

    with pytest.raises(ValueError, match="Unknown ASR engine"):
        build_task_params(source, {
            "sourceLanguage": "zh-cn",
            "targetLanguage": "vi",
            "recognType": 99,
            "translateType": 0,
            "ttsType": 0,
        })

    with pytest.raises(ValueError, match="not supported by Deepgram"):
        build_task_params(source, {
            "sourceLanguage": "zh-cn",
            "targetLanguage": "vi",
            "recognType": recognition.Deepgram,
            "modelName": "large-v3",
            "translateType": 0,
            "ttsType": 0,
        })

    with pytest.raises(ValueError, match="Unknown translation engine"):
        build_task_params(source, {
            "sourceLanguage": "zh-cn",
            "targetLanguage": "vi",
            "recognType": recognition.Deepgram,
            "modelName": "nova-3",
            "translateType": 99,
            "ttsType": 0,
        })


def test_build_task_params_accepts_loaded_deepgram_model(tmp_path, monkeypatch):
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(runtime_config.settings, "deepgram_model", "nova-3-general", raising=False)

    params = build_task_params(source, {
        "sourceLanguage": "zh-cn",
        "targetLanguage": "vi",
        "recognType": recognition.Deepgram,
        "modelName": "nova-3-general",
        "translateType": 0,
        "ttsType": 0,
    }, job_type="asr")

    assert params["model_name"] == "nova-3-general"


def test_build_task_params_maps_each_timing_mode(tmp_path):
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video")
    common = {
        "sourceLanguage": "zh-cn",
        "targetLanguage": "vi",
        "recognType": recognition.Deepgram,
        "modelName": "nova-3",
        "translateType": translator.CHATGPT_INDEX,
        "translationMode": "srt",
        "ttsType": 0,
    }

    voice = build_task_params(source, {**common, "timingMode": "voice"})
    video = build_task_params(source, {**common, "timingMode": "video"})
    align = build_task_params(source, {**common, "timingMode": "align"})

    assert (voice["voice_autorate"], voice["video_autorate"], voice["align_sub_audio"]) == (True, False, False)
    assert voice["aisendsrt"] is True
    assert (video["voice_autorate"], video["video_autorate"], video["align_sub_audio"]) == (False, True, False)
    assert (align["voice_autorate"], align["video_autorate"], align["align_sub_audio"]) == (False, False, True)


def test_api_provider_configuration_is_required_before_start():
    class EmptySettings:
        def get(self, _key, default=None):
            return default

    with pytest.raises(ValueError, match="Configure Deepgram API settings"):
        ensure_asr_configured(recognition.Deepgram, EmptySettings())

    ensure_asr_configured(recognition.FASTER_WHISPER, EmptySettings())
    with pytest.raises(ValueError, match="Configure OpenAI ChatGPT settings"):
        ensure_translation_configured(translator.CHATGPT_INDEX, EmptySettings())
    ensure_translation_configured(translator.GOOGLE_INDEX, EmptySettings())


def test_options_expose_only_supported_asr_providers_and_safe_configuration_state(tmp_path):
    class FakeSettings:
        def __init__(self):
            self.values = {
                "source_language": "zh-cn",
                "target_language": "vi",
                "deepgram_apikey": "configured-secret",
                "elevenlabstts_key": "",
                "chatgpt_api": "https://api.openai.com/v1",
                "chatgpt_key": "translation-secret",
                "chatgpt_model": "gpt-5-mini",
                "gemini_key": "",
                "gemini_model": "gemini-2.5-flash",
                "deepseek_key": "",
                "deepseek_model": "deepseek-v4-flash",
            }

        def get(self, key, default=None):
            return self.values.get(key, default)

        def __setitem__(self, key, value):
            self.values[key] = value

        def save(self):
            pass

    app = create_app(upload_dir=tmp_path / "uploads", settings_store=FakeSettings())

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/api/options")
            assert response.status == 200
            data = await response.json()
            providers = data["asrProviders"]
            assert [item["label"] for item in providers] == [
                "Qwen-ASR",
                "Deepgram",
                "Gemini STT",
                "Google STT API",
                "ElevenLabs",
                "Whisper Large-v3",
            ]
            assert data["defaults"] == {
                "sourceLanguage": "zh-cn",
                "targetLanguage": "vi",
                "recognType": recognition.Deepgram,
                "modelName": "nova-3",
                "timingMode": "voice",
                "translateType": translator.GOOGLE_INDEX,
                "translationMode": "srt" if runtime_config.settings.get("aisendsrt", True) else "line",
                "ttsType": tts.VIENEU_TTS,
            }
            assert data["voices"] == [
                [tts.ELEVENLABS_TTS, "ElevenLabs"],
                [tts.OMNIVOICE_TTS, "OmniVoice(Built-in)"],
                [tts.VIENEU_TTS, "VieNeu-TTS"],
                [tts.GEMINI_TTS, "Gemini TTS"],
            ]
            assert data["translationModes"] == [
                {"id": "line", "label": "Line-by-line", "description": "Send plain subtitle text in batches."},
                {"id": "srt", "label": "Send SRT", "description": "Send subtitle blocks with timestamps and structure."},
            ]
            assert next(item for item in providers if item["id"] == "qwen-asr")["models"] == ["1.7B", "0.6B"]
            assert next(item for item in providers if item["id"] == "whisper-large-v3")["models"] == ["large-v3"]
            deepgram = next(item for item in providers if item["id"] == "deepgram")
            assert deepgram["requiresSettings"] is True
            assert deepgram["configured"] is True
            assert deepgram["testable"] is True
            assert next(item for item in providers if item["id"] == "google-stt")["testable"] is True
            assert next(item for item in providers if item["id"] == "qwen-asr")["testable"] is False
            translation_providers = data["translationProviders"]
            assert [item["label"] for item in translation_providers] == [
                "Google Translate",
                "OpenAI ChatGPT",
                "Gemini",
                "DeepSeek",
            ]
            openai = next(item for item in translation_providers if item["id"] == "openai")
            assert openai["configured"] is True
            assert openai["baseUrl"] == "https://api.openai.com/v1"
            assert openai["model"] == "gpt-5-mini"
            assert "gpt-5-mini" in openai["models"]
            assert "configured-secret" not in await response.text()
            assert "translation-secret" not in await response.text()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_options_clamps_negative_gpu_count_to_zero(tmp_path):
    app = create_app(upload_dir=tmp_path / "uploads")
    app.state.gpu_initializer = lambda: -1

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/api/options")
            assert response.status == 200
            data = await response.json()
            assert data["hardware"] == {"cudaAvailable": False, "gpuCount": 0}
        finally:
            await client.close()

    asyncio.run(scenario())


def test_translation_settings_save_and_test_supported_providers(tmp_path):
    class FakeSettings:
        def __init__(self):
            self.values = {}
            self.save_count = 0

        def get(self, key, default=None):
            return self.values.get(key, default)

        def __setitem__(self, key, value):
            self.values[key] = value

        def save(self):
            self.save_count += 1

    tested = []

    def fake_test(translate_type, aisendsrt):
        tested.append((translate_type, aisendsrt))
        return "Hello, my friend"

    settings = FakeSettings()
    app = create_app(
        upload_dir=tmp_path / "uploads",
        settings_store=settings,
        translation_tester=fake_test,
    )

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            cases = [
                ("openai", "chatgpt", "https://api.openai.com/v1", "gpt-5-mini"),
                ("gemini", "gemini", "https://gemini.example.test", "gemini-2.5-flash"),
                ("deepseek", "deepseek", "https://deepseek.example.test/v1", "deepseek-v4-flash"),
            ]
            for provider_id, prefix, base_url, model in cases:
                payload = {"baseUrl": base_url, "apiKey": f"{provider_id}-secret", "model": model}
                response = await client.post(f"/api/translation-settings/{provider_id}", json=payload)
                assert response.status == 200
                body = await response.json()
                assert body["configured"] is True
                assert "secret" not in str(body)
                assert settings.values[f"{prefix}_api"] == base_url
                assert settings.values[f"{prefix}_key"] == f"{provider_id}-secret"
                assert settings.values[f"{prefix}_model"] == model

            response = await client.post("/api/translation-settings/openai/test", json={
                "baseUrl": "https://api.openai.com/v1",
                "model": "gpt-5-mini",
                "translationMode": "srt",
            })
            assert response.status == 200
            assert await response.json() == {"ok": True, "message": "Connection successful", "result": "Hello, my friend"}
            assert tested == [(translator.CHATGPT_INDEX, True)]

            response = await client.post("/api/translation-settings/google", json={})
            assert response.status == 404
            response = await client.post("/api/translation-settings/unsupported/test", json={})
            assert response.status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_translation_backends_use_configured_base_urls(monkeypatch):
    from videotrans.translator import _chatgpt, _deepseek, _gemini

    class FakeParams:
        values = {
            "chatgpt_api": "https://openai.example.test/v1",
            "chatgpt_key": "key",
            "chatgpt_model": "gpt-test",
            "chatgpt_max_token": 100,
            "gemini_api": "https://gemini.example.test",
            "gemini_key": "key",
            "gemini_model": "gemini-test",
            "deepseek_api": "https://deepseek.example.test/v1",
            "deepseek_key": "key",
            "deepseek_model": "deepseek-test",
            "deepseek_max_token": 100,
        }

        def get(self, key, default=None):
            return self.values.get(key, default)

    fake = FakeParams()
    monkeypatch.setattr(_chatgpt, "params", fake)
    monkeypatch.setattr(_deepseek, "params", fake)
    monkeypatch.setattr(_gemini, "params", fake)
    common = {
        "text_list": [],
        "source_code": "zh-cn",
        "target_code": "en",
        "target_language_name": "English",
    }

    assert _chatgpt.ChatGPT(translate_type=translator.CHATGPT_INDEX, **common).api_url == "https://openai.example.test/v1"
    assert _gemini.Gemini(translate_type=translator.GEMINI_INDEX, **common).api_url == "https://gemini.example.test"
    assert _deepseek.DeepSeek(translate_type=translator.DEEPSEEK_INDEX, **common).api_url == "https://deepseek.example.test/v1"


def test_asr_settings_saves_only_supported_provider_keys(tmp_path):
    class FakeSettings:
        def __init__(self):
            self.values = {"deepgram_apikey": "", "gemini_key": "", "elevenlabstts_key": ""}
            self.save_count = 0

        def get(self, key, default=None):
            return self.values.get(key, default)

        def __setitem__(self, key, value):
            self.values[key] = value

        def save(self):
            self.save_count += 1

    settings = FakeSettings()
    app = create_app(upload_dir=tmp_path / "uploads", settings_store=settings)

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post("/api/asr-settings/deepgram", json={"apiKey": "new-secret"})
            assert response.status == 200
            assert await response.json() == {"configured": True}
            assert settings.values["deepgram_apikey"] == "new-secret"
            assert settings.save_count == 1

            response = await client.post("/api/asr-settings/google-stt", json={"apiKey": "not-allowed"})
            assert response.status == 404
            response = await client.post("/api/asr-settings/gemini-stt", json={"apiKey": "  "})
            assert response.status == 400
        finally:
            await client.close()

    asyncio.run(scenario())


def test_asr_settings_tests_selected_third_party_model(tmp_path):
    class FakeSettings:
        def __init__(self):
            self.values = {"deepgram_apikey": "", "gemini_key": "", "elevenlabstts_key": ""}
            self.save_count = 0

        def get(self, key, default=None):
            return self.values.get(key, default)

        def __setitem__(self, key, value):
            self.values[key] = value

        def save(self):
            self.save_count += 1

    tested = []

    def fake_test(recogn_type, model_name):
        tested.append((recogn_type, model_name))
        return "Hello from the ASR sample"

    settings = FakeSettings()
    app = create_app(
        upload_dir=tmp_path / "uploads",
        settings_store=settings,
        asr_tester=fake_test,
    )

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post("/api/asr-settings/deepgram/test", json={
                "apiKey": "deepgram-secret",
                "model": "nova-3",
            })
            assert response.status == 200
            assert await response.json() == {
                "ok": True,
                "message": "Connection successful",
                "model": "nova-3",
                "result": "Hello from the ASR sample",
            }
            assert settings.values["deepgram_apikey"] == "deepgram-secret"
            assert tested == [(recognition.Deepgram, "nova-3")]

            response = await client.post("/api/asr-settings/google-stt/test", json={"model": "google-web-speech"})
            assert response.status == 200
            assert tested[-1] == (recognition.GOOGLE_SPEECH, "google-web-speech")

            response = await client.post("/api/asr-settings/deepgram/test", json={"model": "large-v3"})
            assert response.status == 400
            response = await client.post("/api/asr-settings/qwen-asr/test", json={"model": "1.7B"})
            assert response.status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_asr_connection_tester_uses_production_recognition_with_selected_model(tmp_path, monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        assert Path(kwargs["cache_folder"]).is_dir()
        return [{"text": "Sample transcript"}]

    monkeypatch.setattr(recognition, "run", fake_run)

    result = run_asr_provider_test(recognition.Deepgram, "nova-3", recognition_runner=fake_run, temp_dir=str(tmp_path))

    assert result == "Sample transcript"
    assert calls[0]["recogn_type"] == recognition.Deepgram
    assert calls[0]["model_name"] == "nova-3"
    assert calls[0]["detect_language"] == "zh-cn"
    assert calls[0]["audio_file"].endswith("videotrans/assets/no-remove.wav")
    assert not Path(calls[0]["cache_folder"]).exists()


def test_job_manager_reports_events_outputs_and_terminal_status(tmp_path, monkeypatch):
    output = tmp_path / "dubbed.mp4"
    output.write_bytes(b"result")

    def fake_run(_request, sink, _token):
        sink(TaskEvent("task", EventKind.RUNNING, "prepare"))
        sink(TaskEvent("task", EventKind.PROGRESS, "prepare", progress=42.0))
        return TaskResult("task", TaskStatus.SUCCEEDED, tmp_path, (output,))

    manager = JobManager(runner=fake_run)
    job = manager.submit({"name": "unused"}, media_id="media")

    deadline = time.time() + 2
    while job.snapshot()["status"] not in {"succeeded", "failed", "cancelled"} and time.time() < deadline:
        time.sleep(0.01)
    snapshot = job.snapshot()

    assert snapshot["status"] == "succeeded"
    assert snapshot["stage"] == "prepare"
    assert snapshot["progress"] == 100.0
    assert snapshot["outputs"] == [{
        "name": "dubbed.mp4",
        "url": f"/api/jobs/{job.id}/outputs/0",
    }]


def test_cancel_is_scoped_to_one_job():
    manager = JobManager()
    first = JobRecord("first", CancellationToken())
    second = JobRecord("second", CancellationToken())
    manager._jobs = {"first": first, "second": second}

    manager.cancel("first")

    assert first.token.is_cancelled()
    assert not second.token.is_cancelled()


def test_media_ingest_and_job_submission_are_end_to_end(tmp_path, monkeypatch):
    release = threading.Event()
    received = {}
    output = tmp_path / "dubbed.mp4"
    output.write_bytes(b"result")

    def fake_probe(path):
        assert Path(path).read_bytes() == b"synthetic video"
        return {
            "time": 12_345,
            "width": 1280,
            "height": 720,
            "video_fps": 25,
            "video_codec_name": "h264",
            "audio_codec_name": "aac",
            "format_name": "QuickTime / MOV",
            "bit_rate": 1_500_000,
            "audio_sample_rate": 48_000,
            "audio_channels": 2,
            "video_streams": 1,
            "streams_audio": 1,
        }

    def fake_run(request, sink, _token):
        received.update(request.params)
        sink(TaskEvent("task", EventKind.STAGE_STARTED, "prepare", message="Preparing media"))
        release.wait(2)
        return TaskResult("task", TaskStatus.SUCCEEDED, tmp_path, (output,))

    manager = JobManager(runner=fake_run)
    app = create_app(job_manager=manager, upload_dir=tmp_path / "uploads", media_probe=fake_probe, gpu_initializer=lambda: None)

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            form = FormData()
            form.add_field("file", io.BytesIO(b"synthetic video"), filename="sample clip.mp4", content_type="video/mp4")
            response = await client.post("/api/media", data=form)
            assert response.status == 201
            media = await response.json()
            assert media["filename"] == "sample clip.mp4"
            assert media["durationMs"] == 12_345
            assert media["sizeBytes"] == len(b"synthetic video")
            assert media["resolution"] == "1280x720"
            assert media["videoCodec"] == "h264"
            assert media["container"] == "QuickTime / MOV"
            assert media["bitrate"] == 1_500_000
            assert media["audioSampleRate"] == 48_000
            assert media["audioChannels"] == 2

            language_codes = list(translator.LANGNAME_DICT)
            payload = {
                "mediaId": media["id"],
                "options": {
                    "sourceLanguage": language_codes[0],
                    "targetLanguage": language_codes[-1],
                    "recognType": recognition.FASTER_WHISPER,
                    "modelName": "large-v3",
                    "translateType": 0,
                    "ttsType": 0,
                },
            }
            response = await client.post("/api/jobs", json=payload)
            assert response.status == 202
            job = await response.json()

            duplicate = await client.post("/api/jobs", json=payload)
            assert duplicate.status == 409
            assert "already running" in (await duplicate.text()).lower()

            release.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                response = await client.get(f"/api/jobs/{job['id']}")
                snapshot = await response.json()
                if snapshot["status"] == "succeeded":
                    break
                await asyncio.sleep(0.01)
            assert snapshot["status"] == "succeeded"
            assert received["source_language_code"] == language_codes[0]
            assert received["target_language_code"] == language_codes[-1]
            assert received["name"].endswith("sample clip.mp4")
        finally:
            release.set()
            await client.close()

    asyncio.run(scenario())


def test_media_ingest_rejects_unreadable_media_and_removes_upload(tmp_path):
    def failed_probe(_path):
        raise ValueError("not readable media")

    upload_dir = tmp_path / "uploads"
    app = create_app(upload_dir=upload_dir, media_probe=failed_probe)

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            form = FormData()
            form.add_field("file", io.BytesIO(b"broken"), filename="broken.mp4", content_type="video/mp4")
            response = await client.post("/api/media", data=form)
            assert response.status == 400
            assert "not readable media" in (await response.text()).lower()
            assert not list(upload_dir.glob("*"))
        finally:
            await client.close()

    asyncio.run(scenario())


def test_prepare_job_stops_at_transcript_checkpoint_and_skips_render_work(tmp_path, monkeypatch):
    received = {}

    def fake_run(request, sink, token, *, stop_after_stage=None):
        received["stop_after_stage"] = stop_after_stage
        received["params"] = dict(request.params)
        sink(TaskEvent("task", EventKind.STAGE_STARTED, "recogn"))
        return TaskResult("task", TaskStatus.SUCCEEDED, tmp_path, ())

    request = TaskRequest({
        "name": str(tmp_path / "input.mp4"),
        "target_dir": str(tmp_path / "output"),
    })
    (tmp_path / "input.mp4").write_bytes(b"video")

    result = run_prepare_review(request, lambda event: None, CancellationToken(), runner=fake_run)

    assert result.status == TaskStatus.SUCCEEDED
    assert received["stop_after_stage"] == "diariz"


def test_build_task_params_video_ocr_bypasses_asr_and_validates_roi(tmp_path, monkeypatch):
    from videotrans.task import orchestrator
    source = tmp_path / "silent.mp4"
    source.write_bytes(b"video")
    temp_dir = tmp_path / "temp"
    output_dir = tmp_path / "output"
    temp_dir.mkdir()
    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_dir))

    for job_type in ("asr", "full"):
        params = build_task_params(
            source,
            {
                "subtitleSource": "video_ocr",
                "ocrRoi": [0.1, 0.75, 0.8, 0.2],
                "ocrRoiConfirmed": True,
                "ocrOptions": {
                    "coarseIntervalMs": 250,
                    "useAngleCls": False,
                    "paragraphMode": "unwrap",
                },
                "sourceLanguage": "zh-cn",
                "targetLanguage": "en",
                "recognType": 9999,
                "modelName": "nonexistent-model",
                "translateType": translator.GOOGLE_INDEX,
                "ttsType": tts.VIENEU_TTS,
                "voiceRole": "No",
                "speakerDiarization": True,
                "speakerCount": 3,
                "removeNoise": True,
            },
            job_type=job_type,
            temp_dir=str(temp_dir),
            output_dir=output_dir,
        )
        assert params["subtitle_source"] == "video_ocr"
        assert params["ocr_roi"] == (0.1, 0.75, 0.8, 0.2)
        assert params["ocr_roi_confirmed"] is True
        assert params["ocr_options"] == {
            "coarse_interval_ms": 250,
            "use_angle_cls": False,
            "paragraph_mode": "unwrap",
        }
        assert params["enable_diariz"] is False
        assert params["nums_diariz"] == 0
        if job_type == "full":
            assert params["remove_noise"] is False
        normalized = TaskRequest(params).normalize()
        assert normalized["ocr_options"] == params["ocr_options"]

    # Dict ROI and snake_case keys
    params_snake = build_task_params(
        source,
        {
            "subtitle_source": "video_ocr",
            "ocr_roi": {"x": 0.05, "y": 0.8, "width": 0.9, "height": 0.15},
            "ocr_roi_confirmed": True,
            "ocr_options": {"coarse_interval_ms": 50},
            "recognType": 9999,
        },
        job_type="asr",
        temp_dir=str(temp_dir),
        output_dir=output_dir,
    )
    assert params_snake["ocr_roi"] == (0.05, 0.8, 0.9, 0.15)
    assert params_snake["ocr_options"]["coarse_interval_ms"] == 100

    with pytest.raises(ValueError, match="ocrRoi is required"):
        build_task_params(source, {"subtitleSource": "video_ocr"}, job_type="asr")

    for jtype in ("asr", "full", "vtv"):
        with pytest.raises(ValueError, match="ocrRoiConfirmed must be true when subtitleSource is video_ocr"):
            build_task_params(
                source,
                {"subtitleSource": "video_ocr", "ocrRoi": [0.1, 0.7, 0.8, 0.2], "ocrRoiConfirmed": False},
                job_type=jtype,
            )
        with pytest.raises(ValueError, match="ocrRoiConfirmed must be true when subtitleSource is video_ocr"):
            build_task_params(
                source,
                {"subtitleSource": "video_ocr", "ocrRoi": [0.1, 0.7, 0.8, 0.2]},
                job_type=jtype,
            )

    render_params = build_task_params(
        source,
        {"subtitleSource": "video_ocr", "ocrRoi": [0.1, 0.7, 0.8, 0.2], "ocrRoiConfirmed": False},
        job_type="render",
        temp_dir=str(temp_dir),
        output_dir=output_dir,
    )
    assert render_params["ocr_roi_confirmed"] is False

    with pytest.raises(ValueError, match="Unsupported subtitle source"):
        build_task_params(source, {"subtitleSource": "magic_sub", "recognType": 0}, job_type="asr")

    with pytest.raises(ValueError, match="paragraphMode"):
        build_task_params(
            source,
            {
                "subtitleSource": "video_ocr",
                "ocrRoi": [0.1, 0.7, 0.8, 0.2],
                "ocrRoiConfirmed": True,
                "ocrOptions": {"paragraphMode": "invalid"},
            },
            job_type="asr",
        )

    for bool_val in (True, False):
        with pytest.raises(ValueError, match="coarseIntervalMs must be an integer"):
            build_task_params(
                source,
                {
                    "subtitleSource": "video_ocr",
                    "ocrRoi": [0.1, 0.7, 0.8, 0.2],
                    "ocrRoiConfirmed": True,
                    "ocrOptions": {"coarseIntervalMs": bool_val},
                },
                job_type="asr",
            )


def test_jobs_api_video_ocr_skips_asr_validator_for_asr_and_vtv(tmp_path):
    asr_calls = []
    trans_calls = []

    def fake_probe(_path):
        return {"time": 5000, "width": 1280, "height": 720, "video_streams": 1, "streams_audio": 0}

    def fake_run(request, _sink, _token):
        return TaskResult("task", TaskStatus.SUCCEEDED, tmp_path, ())

    app = create_app(
        job_manager=JobManager(runner=fake_run),
        upload_dir=tmp_path / "uploads",
        media_probe=fake_probe,
        asr_validator=lambda r_type, _s: asr_calls.append(r_type),
        translation_validator=lambda t_type, _s: trans_calls.append(t_type),
        gpu_initializer=lambda: None,
    )

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for idx, jtype in enumerate(("asr", "full")):
                form = FormData()
                form.add_field("file", io.BytesIO(f"video-only-{idx}".encode()), filename=f"silent-{idx}.mp4", content_type="video/mp4")
                up = await client.post("/api/media", data=form)
                assert up.status == 201
                media_id = (await up.json())["id"]

                res = await client.post("/api/jobs", json={
                    "mediaId": media_id,
                    "jobType": jtype,
                    "options": {
                        "subtitleSource": "video_ocr",
                        "ocrRoi": [0.1, 0.75, 0.8, 0.2],
                        "ocrRoiConfirmed": True,
                        "sourceLanguage": "zh-cn",
                        "targetLanguage": "en",
                        "recognType": 9999,
                        "translateType": translator.GOOGLE_INDEX,
                        "voiceRole": "No",
                    },
                })
                assert res.status == 202
                job = await res.json()
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    snap = await (await client.get(f"/api/jobs/{job['id']}")).json()
                    if snap["status"] in {"succeeded", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0.01)

            assert asr_calls == []
            assert trans_calls == [translator.GOOGLE_INDEX]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_jobs_api_video_ocr_rejects_unconfirmed_roi(tmp_path):
    from fastapi.testclient import TestClient as FastAPITestClient

    def fake_probe(_path):
        return {"time": 5000, "width": 1280, "height": 720, "video_streams": 1, "streams_audio": 0}

    def fake_run(request, _sink, _token):
        return TaskResult("task", TaskStatus.SUCCEEDED, tmp_path, ())

    app = create_app(
        job_manager=JobManager(runner=fake_run),
        upload_dir=tmp_path / "uploads",
        media_probe=fake_probe,
        gpu_initializer=lambda: None,
    )

    with FastAPITestClient(app) as client:
        up = client.post(
            "/api/media",
            files={"file": ("silent.mp4", b"video-only", "video/mp4")},
        )
        assert up.status_code == 201
        media_id = up.json()["id"]

        for jtype in ("asr", "vtv", "full"):
            r_false = client.post(
                "/api/jobs",
                json={
                    "mediaId": media_id,
                    "jobType": jtype,
                    "options": {
                        "subtitleSource": "video_ocr",
                        "ocrRoi": [0.1, 0.75, 0.8, 0.2],
                        "ocrRoiConfirmed": False,
                        "sourceLanguage": "en",
                        "targetLanguage": "zh-cn",
                        "translateType": translator.GOOGLE_INDEX,
                    },
                },
            )
            assert r_false.status_code == 400
            assert "ocrRoiConfirmed must be true when subtitleSource is video_ocr" in r_false.text

            r_omitted = client.post(
                "/api/jobs",
                json={
                    "mediaId": media_id,
                    "jobType": jtype,
                    "options": {
                        "subtitleSource": "video_ocr",
                        "ocrRoi": [0.1, 0.75, 0.8, 0.2],
                        "sourceLanguage": "en",
                        "targetLanguage": "zh-cn",
                        "translateType": translator.GOOGLE_INDEX,
                    },
                },
            )
            assert r_omitted.status_code == 400
            assert "ocrRoiConfirmed must be true when subtitleSource is video_ocr" in r_omitted.text


def test_prepare_mixin_video_ocr_allows_audioless_video_and_skips_asr_audio_prep(tmp_path, monkeypatch):
    from videotrans.configure.excepts import VideoTransError
    from videotrans.task._stage_prepare import PrepareMixin
    from videotrans.task.taskcfg import TaskCfgVTT

    class DummyPrepare(PrepareMixin):
        def __init__(self, cfg, *, should_hebing=False, should_dubbing=False):
            self.cfg = cfg
            self.uuid = "prep-uuid"
            self.is_audio_trans = False
            self.is_copy_video = False
            self.should_hebing = should_hebing
            self.should_dubbing = should_dubbing
            self.should_separate = False
            self.clone_ref = ""
            self.video_info = {}
            self.video_time = 0
            self.split_calls = []

        def _exit(self):
            return False

        def signal(self, **_kwargs):
            pass

        def _unlink_size0(self, _paths):
            pass

        def _split_novoice_byraw(self):
            pass

        def _split_audio_byraw(self, is_separate=False):
            self.split_calls.append(is_separate)

    video = tmp_path / "silent.mp4"
    video.write_bytes(b"video")

    import videotrans.task._stage_prepare as prep_mod
    monkeypatch.setattr(
        prep_mod,
        "get_video_info",
        lambda _p: {"time": 5000, "video_streams": 1, "streams_audio": 0, "video_codec_name": "h264", "color": "yuv420p"},
    )

    cfg_audio = TaskCfgVTT(
        name=str(video),
        cache_folder=str(tmp_path / "cache1"),
        target_dir=str(tmp_path / "out1"),
        subtitle_source="audio_asr",
    )
    with pytest.raises(VideoTransError):
        DummyPrepare(cfg_audio).prepare()

    cfg_ocr = TaskCfgVTT(
        name=str(video),
        cache_folder=str(tmp_path / "cache2"),
        target_dir=str(tmp_path / "out2"),
        subtitle_source="video_ocr",
        only_out_dubbed_audio=True,
    )
    task_ocr = DummyPrepare(cfg_ocr)
    task_ocr.prepare()
    assert task_ocr.split_calls == []

    # Video WITH audio under video_ocr + ASR-only skips _split_audio_byraw
    monkeypatch.setattr(
        prep_mod,
        "get_video_info",
        lambda _p: {"time": 5000, "video_streams": 1, "streams_audio": 1, "video_codec_name": "h264", "color": "yuv420p"},
    )
    task_ocr_with_audio = DummyPrepare(cfg_ocr)
    task_ocr_with_audio.prepare()
    assert task_ocr_with_audio.split_calls == []


def test_recogn_and_diariz_video_ocr_options_duration_and_null_lang(tmp_path, monkeypatch):
    from videotrans.task._stage_diariz import DiarizMixin
    from videotrans.task._stage_recogn import RecognMixin
    from videotrans.task.taskcfg import TaskCfgVTT, SrtItem

    class DummyRecogn(RecognMixin, DiarizMixin):
        def __init__(self, cfg):
            self.cfg = cfg
            self.uuid = "rec-uuid"
            self.should_recogn = True
            self.precent = 0
            self.video_time = 4000
            self.max_speakers = 2
            self.asr_duration = None
            self.source_srt_list = []
            self.signals = []
            self.event_sink = None
            self.clone_ref = ""
            self.queue_tts = []

        def _exit(self):
            return False

        def signal(self, **kwargs):
            self.signals.append(kwargs)

        def _save_srt_target(self, items, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    import videotrans.task._stage_recogn as rec_mod
    import videotrans.core.proc_registry as proc_reg

    monkeypatch.setattr(proc_reg, "get_current_job_id", lambda: "job-xyz")
    captured = {}

    def fake_scan(video_path, **kwargs):
        captured["video_path"] = video_path
        captured.update(kwargs)
        kwargs["progress_callback"]({
            "latest_text": "hello",
            "percent": 50.0,
            "timestamp_ms": 2000,
            "duration_ms": 4000,
            "segments": 1,
            "ocr_calls": 3,
            "skipped_frames": 2,
            "device": "cpu",
        })
        return [SrtItem(line=1, time="00:00:00,000 --> 00:00:01,000", start_time=0, end_time=1000, startraw="00:00:00,000", endraw="00:00:01,000", text="hello")], []

    monkeypatch.setattr(rec_mod, "scan_video_to_srt", fake_scan)

    cfg = TaskCfgVTT(
        name=str(tmp_path / "in.mp4"),
        basename="in.mp4",
        noextname="in",
        target_dir=str(tmp_path / "out"),
        cache_folder=str(tmp_path / "cache"),
        source_sub=str(tmp_path / "out" / "zh-cn.srt"),
        app_mode="biaozhun",
        subtitle_source="video_ocr",
        ocr_roi=(0.1, 0.7, 0.8, 0.2),
        ocr_roi_confirmed=True,
        ocr_options={"coarse_interval_ms": 250, "use_angle_cls": True, "paragraph_mode": "unwrap"},
        detect_language=None,
        enable_diariz=True,
    )
    Path(cfg.cache_folder).mkdir(parents=True, exist_ok=True)

    task = DummyRecogn(cfg)
    task.recogn()
    assert captured["job_id"] == "job-xyz"
    assert captured["stop_event"] == task._exit
    assert captured["coarse_interval_ms"] == 250
    assert captured["use_angle_cls"] is True
    assert captured["paragraph_mode"] == "unwrap"
    assert task.asr_duration is not None and task.asr_duration >= 0.0
    assert any(s.get("type") == "ocr_progress" and s.get("ocr_calls") == 3 for s in task.signals)

    # Diarization skips cleanly for video_ocr even when enable_diariz=True and detect_language=None
    task.diariz()

    # Cached source_sub early return sets asr_duration
    task2 = DummyRecogn(TaskCfgVTT(
        name=str(tmp_path / "in.mp4"),
        basename="in.mp4",
        noextname="in",
        target_dir=str(tmp_path / "out"),
        cache_folder=str(tmp_path / "cache"),
        source_sub=cfg.source_sub,
        subtitle_source="audio_asr",
        force_recogn=False,
    ))
    task2.recogn()
    assert task2.asr_duration is not None

    # detect_language=None with fix_punc=1 in audio_asr does not raise TypeError
    cfg3 = TaskCfgVTT(
        name=str(tmp_path / "in.mp4"),
        basename="in.mp4",
        noextname="in",
        target_dir=str(tmp_path / "out3"),
        cache_folder=str(tmp_path / "cache3"),
        source_sub=str(tmp_path / "out3" / "sub.srt"),
        source_wav=str(tmp_path / "in.wav"),
        subtitle_source="audio_asr",
        detect_language=None,
        fix_punc=1,
        recogn_type=0,
        model_name="tiny",
        force_recogn=True,
    )
    Path(cfg3.cache_folder).mkdir(parents=True, exist_ok=True)
    Path(cfg3.source_wav).write_bytes(b"wav")
    monkeypatch.setattr(rec_mod, "vail_file", lambda p: bool(p and Path(p).exists() and Path(p).stat().st_size > 0))
    monkeypatch.setattr(rec_mod, "run_recogn", lambda **_kw: [{"line": 1, "time": "00:00:00,000 --> 00:00:01,000", "start_time": 0, "end_time": 1000, "startraw": "00:00:00,000", "endraw": "00:00:01,000", "text": "ok"}])
    task3 = DummyRecogn(cfg3)
    task3.recogn()
    assert task3.asr_duration is not None


def test_orchestrator_video_ocr_progress_and_audio_hash_bypass(tmp_path, monkeypatch):
    from videotrans.core import project_store
    from videotrans.task import orchestrator

    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "output"
    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))

    created_proj = project_store.create_project("ocr-test", media_path=str(source))
    pid = created_proj["id"]

    class FakeOcrTask:
        def __init__(self, cfg, event_sink=None, cancellation_token=None):
            self.cfg = cfg
            self.event_sink = event_sink
            self.cancellation_token = cancellation_token
            self.should_recogn = True
            self.should_trans = False
            self.should_dubbing = False
            self.should_recogn2 = False
            self.should_hebing = False
            self.source_srt_list = []
            self.cfg.source_sub = str(Path(cfg.target_dir) / "source.srt")
            self.cfg.target_sub = str(Path(cfg.target_dir) / "target.srt")
            self.cfg.source_wav_output = str(Path(cfg.target_dir) / "source.m4a")
            self.cfg.target_wav_output = str(Path(cfg.target_dir) / "target.m4a")
            self.cfg.targetdir_mp4 = str(Path(cfg.target_dir) / "result.mp4")

        def prepare(self):
            pass

        def recogn(self):
            self.event_sink({
                "type": "ocr_progress",
                "text": "hello ocr",
                "progress": 64.5,
                "timestamp_ms": 3200,
                "duration_ms": 5000,
                "segments": 2,
                "ocr_calls": 4,
                "skipped_frames": 3,
                "device": "cpu",
            })

        def diariz(self):
            pass

        def task_done(self):
            pass

    monkeypatch.setattr(orchestrator, "TransCreate", FakeOcrTask)
    monkeypatch.setattr(
        project_store,
        "find_project_by_audio_hash",
        lambda _h: (_ for _ in ()).throw(AssertionError("video_ocr must bypass find_project_by_audio_hash")),
    )

    events = []
    result = orchestrator.run_staged_asr(
        TaskRequest({
            "name": str(source),
            "target_dir": str(output),
            "cache_folder": str(temp_root / "job"),
            "project_id": pid,
            "subtitle_source": "video_ocr",
            "ocr_roi": (0.1, 0.7, 0.8, 0.2),
            "ocr_roi_confirmed": True,
        }),
        events.append,
    )
    assert result.status == TaskStatus.SUCCEEDED
    prog_events = [e for e in events if e.kind == EventKind.PROGRESS and e.details.get("source_type") == "ocr_progress"]
    assert len(prog_events) == 1
    assert prog_events[0].progress == 64.5
    assert prog_events[0].details["ocr_calls"] == 4
    assert not project_store.get_project(pid).get("audio_hash")


def test_speech_to_text_video_ocr_runs_without_audio_loop(tmp_path, monkeypatch):
    from videotrans.task.speech2text import SpeechToText
    from videotrans.task.taskcfg import SrtItem, TaskCfgSTT
    from videotrans.configure import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "TEMP_DIR", str(tmp_path / "temp"))
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"video")

    import videotrans.ocr as ocr_pkg
    import videotrans.util.help_ffmpeg as ffmpeg_mod

    monkeypatch.setattr(ffmpeg_mod, "get_video_info", lambda _p: {"time": 3000})
    monkeypatch.setattr(
        ffmpeg_mod,
        "conver_to_16k",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("conver_to_16k must not run for video_ocr")),
    )
    monkeypatch.setattr(
        ocr_pkg,
        "scan_video_to_srt",
        lambda *_a, **_kw: (
            [SrtItem(line=1, time="00:00:00,000 --> 00:00:02,000", start_time=0, end_time=2000, startraw="00:00:00,000", endraw="00:00:02,000", text="Subtitle")],
            [],
        ),
    )

    stt = SpeechToText(
        cfg=TaskCfgSTT(
            name=str(video),
            basename="silent.mp4",
            noextname="silent",
            target_dir=str(tmp_path / "out"),
            detect_language="en",
            subtitle_source="video_ocr",
            ocr_roi=(0.1, 0.75, 0.8, 0.2),
            ocr_roi_confirmed=True,
            ocr_options={"coarse_interval_ms": 250},
            enable_diariz=True,
        ),
        out_format="srt",
    )
    stt.prepare()
    stt.recogn()
    stt.diariz()
    stt.task_done()
    assert Path(stt.cfg.target_sub).is_file()
    assert "Subtitle" in Path(stt.cfg.target_sub).read_text(encoding="utf-8")


def test_app_lifespan_shutdown_calls_release_cached_providers(tmp_path, monkeypatch):
    import videotrans.ocr as ocr_pkg
    from fastapi.testclient import TestClient as FastAPITestClient

    called = []
    monkeypatch.setattr(ocr_pkg, "release_cached_providers", lambda: called.append(True), raising=False)
    app = create_app(upload_dir=tmp_path / "uploads")
    with FastAPITestClient(app):
        pass
    assert called == [True]


def test_jobs_api_video_ocr_rejects_boolean_coarse_interval_ms(tmp_path):
    from fastapi.testclient import TestClient as FastAPITestClient

    def fake_probe(_path):
        return {"time": 5000, "width": 1280, "height": 720, "video_streams": 1, "streams_audio": 0}

    def fake_run(request, _sink, _token):
        return TaskResult("task", TaskStatus.SUCCEEDED, tmp_path, ())

    app = create_app(
        job_manager=JobManager(runner=fake_run),
        upload_dir=tmp_path / "uploads",
        media_probe=fake_probe,
        gpu_initializer=lambda: None,
    )

    with FastAPITestClient(app) as client:
        up = client.post(
            "/api/media",
            files={"file": ("silent.mp4", b"video-only", "video/mp4")},
        )
        assert up.status_code == 201
        media_id = up.json()["id"]

        for bool_val in (True, False):
            res = client.post(
                "/api/jobs",
                json={
                    "mediaId": media_id,
                    "jobType": "asr",
                    "options": {
                        "subtitleSource": "video_ocr",
                        "ocrRoi": [0.05, 0.75, 0.9, 0.2],
                        "ocrRoiConfirmed": True,
                        "ocrOptions": {"coarseIntervalMs": bool_val},
                    },
                },
            )
            assert res.status_code == 400
            assert "ocrOptions.coarseIntervalMs must be an integer" in res.text
