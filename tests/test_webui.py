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
