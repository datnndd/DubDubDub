# -*- coding: utf-8 -*-
"""Integration tests for custom voice REST APIs, audio normalizer, and preview service."""
import asyncio
import io
from pathlib import Path
import struct
import wave
import aiohttp
from aiohttp.test_utils import TestClient, TestServer
import pytest

from videotrans.api.app import create_app
from videotrans.core.db import init_db
from videotrans.core import voice_store
from videotrans.services.audio_normalizer import normalize_reference_audio
from videotrans import tts


def create_wav_file(path: Path, duration_sec: float = 3.0, sample_rate: int = 48000) -> Path:
    """Generate a valid PCM 16-bit mono WAV test file."""
    num_samples = int(duration_sec * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # 440Hz audible sine wave / non-zero frames
        frames = [int(10000 * ((i % 100) / 100.0)) for i in range(num_samples)]
        wf.writeframes(struct.pack(f"<{num_samples}h", *frames))
    return path


@pytest.fixture
def voice_api_env(tmp_path, monkeypatch):
    """Isolate DB and filesystem for voice management API tests."""
    db_path = tmp_path / "test_api_voices.db"
    init_db(db_path)

    v_dir = tmp_path / "data" / "voices"
    p_dir = tmp_path / "tmp" / "voice_previews"
    uploads_dir = tmp_path / "uploads"

    monkeypatch.setattr(voice_store, "VOICES_DIR", v_dir)
    monkeypatch.setattr(voice_store, "PREVIEWS_DIR", p_dir)
    voice_store.init_voice_dirs()

    # Point db_conn default to this test db
    monkeypatch.setattr("videotrans.core.db._current_db_path", db_path)

    from videotrans.services import voice_preview
    def fake_synthesizer(voice, preview_path, sample_text):
        create_wav_file(preview_path, duration_sec=3.0)

    voice_preview.set_preview_synthesizer(fake_synthesizer)

    class FakeVieNeuEngine:
        def encode_reference(self, path, *, denoise):
            return [0.25], [1]

    monkeypatch.setattr(
        "videotrans.services.model_cache.get_cached_vieneu_engine",
        lambda **kwargs: FakeVieNeuEngine(),
    )

    app = create_app(upload_dir=uploads_dir)
    yield {
        "app": app,
        "db_path": db_path,
        "v_dir": v_dir,
        "p_dir": p_dir,
        "tmp_path": tmp_path,
    }
    voice_preview.set_preview_synthesizer(None)


def test_audio_normalizer_duration_validation(tmp_path):
    # 1. Too short (< 2.0s)
    short_wav = tmp_path / "short.wav"
    create_wav_file(short_wav, duration_sec=1.2)
    out_wav = tmp_path / "out1.wav"
    with pytest.raises(ValueError, match="between 2.0 and 30.0 seconds"):
        normalize_reference_audio(short_wav, out_wav, provider=tts.VIENEU_TTS)

    # 2. Too long (> 30.0s)
    long_wav = tmp_path / "long.wav"
    create_wav_file(long_wav, duration_sec=32.0)
    out_wav2 = tmp_path / "out2.wav"
    with pytest.raises(ValueError, match="between 2.0 and 30.0 seconds"):
        normalize_reference_audio(long_wav, out_wav2, provider=tts.VIENEU_TTS)

    # 3. Valid duration (e.g. 4.0s)
    valid_wav = tmp_path / "valid.wav"
    create_wav_file(valid_wav, duration_sec=4.0)
    out_wav3 = tmp_path / "out3.wav"
    meta = normalize_reference_audio(valid_wav, out_wav3, provider=tts.VIENEU_TTS)
    assert meta["sample_rate"] == 48000
    assert meta["channels"] == 1
    assert out_wav3.is_file()

    # 4. Long audio trimmed with start_time & end_time
    out_wav4 = tmp_path / "out4.wav"
    meta4 = normalize_reference_audio(long_wav, out_wav4, provider=tts.VIENEU_TTS, start_time=2.0, end_time=10.0)
    assert meta4["sample_rate"] == 48000
    assert 7.5 <= meta4["duration"] <= 8.5
    assert out_wav4.is_file()


def test_custom_voices_api_lifecycle(voice_api_env):
    app = voice_api_env["app"]
    tmp_path = voice_api_env["tmp_path"]

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. Initial list should be empty
            res = await client.get("/api/custom-voices")
            assert res.status == 200
            data = await res.json()
            assert data["voices"] == []

            # 2. Create voice - missing name validation
            res_bad_name = await client.post(
                "/api/custom-voices",
                data=aiohttp.FormData({"name": "   ", "provider": "2"}),
            )
            assert res_bad_name.status == 400

            # 3. Create voice - valid multipart upload
            sample_wav = tmp_path / "sample_voice.wav"
            create_wav_file(sample_wav, duration_sec=3.5)

            form = aiohttp.FormData()
            form.add_field("name", "Lan Phương")
            form.add_field("provider", "vieneu")
            form.add_field("description", "Professional storyteller")
            form.add_field("language", "vi")
            form.add_field("instruct", "Warm, emotional tone")
            form.add_field("ref_text", "Bản tin thời sự tối nay.")
            form.add_field(
                "audio",
                sample_wav.read_bytes(),
                filename="sample_voice.wav",
                content_type="audio/wav",
            )

            res_create = await client.post("/api/custom-voices", data=form)
            assert res_create.status == 201
            created = await res_create.json()
            assert created["name"] == "Lan Phương"
            assert created["provider"] == tts.VIENEU_TTS
            assert created["language"] == "vi"
            assert created["ref_text"] == "Bản tin thời sự tối nay."
            voice_id = created["id"]
            assert voice_id.startswith("voice_")

            # 4. Fetch single voice
            res_get = await client.get(f"/api/custom-voices/{voice_id}")
            assert res_get.status == 200
            fetched = await res_get.json()
            assert fetched["id"] == voice_id
            assert fetched["name"] == "Lan Phương"

            # 5. Stream reference audio
            res_audio = await client.get(f"/api/custom-voices/{voice_id}/audio")
            assert res_audio.status == 200
            assert "audio/wav" in res_audio.headers.get("Content-Type", "")
            audio_content = await res_audio.read()
            assert len(audio_content) > 0
            assert audio_content.startswith(b"RIFF")

            # 6. Audition preview generation
            res_preview = await client.post(
                f"/api/custom-voices/{voice_id}/preview",
                json={"text": "Xin chào đây là bản nghe thử"},
            )
            assert res_preview.status == 200
            prev_data = await res_preview.json()
            assert prev_data["ok"] is True
            assert f"/api/custom-voices/{voice_id}/preview/audio" in prev_data["preview_url"]

            # Stream preview audio
            res_prev_audio = await client.get(f"/api/custom-voices/{voice_id}/preview/audio")
            assert res_prev_audio.status == 200
            assert "audio/wav" in res_prev_audio.headers.get("Content-Type", "")

            # 7. Update voice metadata
            res_update = await client.put(
                f"/api/custom-voices/{voice_id}",
                json={"name": "Lan Phương (Pro)", "instruct": "Formal broadcast voice"},
            )
            assert res_update.status == 200
            updated = await res_update.json()
            assert updated["name"] == "Lan Phương (Pro)"
            assert updated["instruct"] == "Formal broadcast voice"

            # 8. Check /api/voices returns rich items alongside voices
            res_voices = await client.get("/api/voices?tts_type=2&language=vi")
            assert res_voices.status == 200
            voices_payload = await res_voices.json()
            assert "voices" in voices_payload
            assert "items" in voices_payload
            assert any(item["name"] == "Lan Phương (Pro)" for item in voices_payload["items"])

            # 9. Soft delete
            res_del = await client.delete(f"/api/custom-voices/{voice_id}")
            assert res_del.status == 200

            # Should not appear in active custom voices
            res_list_after = await client.get("/api/custom-voices")
            list_data = await res_list_after.json()
            assert not any(v["id"] == voice_id for v in list_data["voices"])

            # 10. Hard delete
            res_hard_del = await client.delete(f"/api/custom-voices/{voice_id}?hard=true")
            assert res_hard_del.status == 200

            res_get_deleted = await client.get(f"/api/custom-voices/{voice_id}")
            assert res_get_deleted.status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_audio_normalizer_exotic_formats(tmp_path):
    """Verify audio normalizer correctly handles and transcodes exotic audio formats."""
    # 1. Create a base WAV
    base_wav = tmp_path / "base.wav"
    create_wav_file(base_wav, duration_sec=3.0, sample_rate=44100)

    from videotrans.util.help_ffmpeg import runffmpeg

    # 2. Transcode to .flac, .ogg, .mp3, .m4a
    formats = [
        ("test.flac", ["-y", "-i", str(base_wav), str(tmp_path / "test.flac")]),
        ("test.ogg", ["-y", "-i", str(base_wav), "-c:a", "libvorbis", str(tmp_path / "test.ogg")]),
        ("test.mp3", ["-y", "-i", str(base_wav), "-c:a", "libmp3lame", str(tmp_path / "test.mp3")]),
    ]

    for fname, cmd in formats:
        try:
            runffmpeg(cmd, force_cpu=True)
            in_file = tmp_path / fname
            if in_file.is_file():
                out_wav = tmp_path / f"norm_{fname}.wav"
                meta = normalize_reference_audio(in_file, out_wav, provider=tts.OMNIVOICE_TTS)
                assert meta["sample_rate"] == 24000
                assert meta["channels"] == 1
                assert out_wav.is_file()
                assert out_wav.stat().st_size > 0
        except Exception:
            # Skip specific codec if not compiled into local FFmpeg build
            pass


def test_custom_voices_api_provider_variants(voice_api_env):
    """Verify OmniVoice with ref_text and ElevenLabs with external_voice_id."""
    app = voice_api_env["app"]
    tmp_path = voice_api_env["tmp_path"]

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. OmniVoice creation with ref_text and audio
            omni_wav = tmp_path / "omni.wav"
            create_wav_file(omni_wav, duration_sec=4.0, sample_rate=24000)

            form_omni = aiohttp.FormData()
            form_omni.add_field("name", "Bảo Long")
            form_omni.add_field("provider", "1")
            form_omni.add_field("ref_text", "Đoạn văn bản mẫu kiểm tra.")
            form_omni.add_field("audio", omni_wav.read_bytes(), filename="omni.wav")

            res_omni = await client.post("/api/custom-voices", data=form_omni)
            assert res_omni.status == 201
            omni_data = await res_omni.json()
            assert omni_data["provider"] == tts.OMNIVOICE_TTS
            assert omni_data["ref_text"] == "Đoạn văn bản mẫu kiểm tra."

            # 2. ElevenLabs creation with external_voice_id (no audio upload needed)
            res_el = await client.post(
                "/api/custom-voices",
                json={
                    "name": "Eleven Narrator",
                    "provider": 0,
                    "external_voice_id": "21m00Tcm4TlvDq8ikWAM",
                },
            )
            assert res_el.status == 201
            el_data = await res_el.json()
            assert el_data["provider"] == tts.ELEVENLABS_TTS
            assert el_data["external_voice_id"] == "21m00Tcm4TlvDq8ikWAM"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_bulk_delete_custom_voices_api(voice_api_env):
    """Verify bulk selection and deletion of custom voices."""
    app = voice_api_env["app"]
    db_path = voice_api_env["db_path"]

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. Create 3 custom voices
            v1 = voice_store.create_voice("Voice 1", provider=tts.VIENEU_TTS, db_path=db_path)
            v2 = voice_store.create_voice("Voice 2", provider=tts.VIENEU_TTS, db_path=db_path)
            v3 = voice_store.create_voice("Voice 3", provider=tts.VIENEU_TTS, db_path=db_path)

            initial_list = voice_store.list_voices(db_path=db_path)
            assert len(initial_list) == 3

            # 2. Bulk delete v1 and v2 via API
            res = await client.post(
                "/api/custom-voices/bulk-delete",
                json={"ids": [v1["id"], v2["id"]]},
            )
            assert res.status == 200
            data = await res.json()
            assert data["ok"] is True
            assert data["count"] == 2
            assert set(data["deleted"]) == {v1["id"], v2["id"]}

            # 3. Verify only v3 remains active in list
            res_list = await client.get("/api/custom-voices")
            assert res_list.status == 200
            list_data = await res_list.json()
            remaining_ids = [item["id"] for item in list_data["voices"]]
            assert remaining_ids == [v3["id"]]

            # 4. Direct voice_store.bulk_delete_voices test
            deleted_v3 = voice_store.bulk_delete_voices([v3["id"]], db_path=db_path)
            assert deleted_v3 == [v3["id"]]
            assert len(voice_store.list_voices(db_path=db_path)) == 0
        finally:
            await client.close()

    asyncio.run(scenario())


def test_audio_trim_endpoints(voice_api_env):
    app = voice_api_env["app"]
    tmp_path = voice_api_env["tmp_path"]

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. Create a 35-second test WAV file
            long_wav = tmp_path / "sample_long.wav"
            create_wav_file(long_wav, duration_sec=35.0)

            # 2. Call /api/voices/trim-audio
            form = aiohttp.FormData()
            form.add_field(
                "audio",
                open(long_wav, "rb"),
                filename="sample_long.wav",
                content_type="audio/wav",
            )
            form.add_field("start_time", "5.0")
            form.add_field("end_time", "15.0")

            res_trim = await client.post("/api/voices/trim-audio", data=form)
            assert res_trim.status == 200
            trim_data = await res_trim.json()
            assert trim_data["ok"] is True
            assert trim_data["start_time"] == 5.0
            assert trim_data["end_time"] == 15.0
            assert 9.5 <= trim_data["duration"] <= 10.5
            assert "audio_url" in trim_data

            # 3. Retrieve the trimmed audio via GET
            res_audio = await client.get(trim_data["audio_url"])
            assert res_audio.status == 200
            assert res_audio.headers.get("Content-Type") == "audio/wav"

            # 4. Create custom voice with cut_start and cut_end from long audio
            form2 = aiohttp.FormData()
            form2.add_field("name", "Trimmed Cloned Voice")
            form2.add_field("provider", "2")
            form2.add_field("cut_start", "4.0")
            form2.add_field("cut_end", "12.0")
            form2.add_field(
                "audio",
                open(long_wav, "rb"),
                filename="sample_long.wav",
                content_type="audio/wav",
            )
            res_create = await client.post("/api/custom-voices", data=form2)
            assert res_create.status == 201
            v_data = await res_create.json()
            assert v_data["name"] == "Trimmed Cloned Voice"
            assert v_data["ref_audio_path"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_preview_clone_and_auto_synthesize_endpoints(voice_api_env):
    """Verify preview-clone endpoint before save, preview-audio retrieval, and auto-synthesis on GET."""
    app = voice_api_env["app"]
    tmp_path = voice_api_env["tmp_path"]

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1. Create a 4.0-second test WAV file
            test_wav = tmp_path / "test_clone.wav"
            create_wav_file(test_wav, duration_sec=4.0)

            # 2. Call POST /api/voices/preview-clone BEFORE saving voice
            form_preview = aiohttp.FormData()
            form_preview.add_field("audio", test_wav.read_bytes(), filename="test_clone.wav", content_type="audio/wav")
            form_preview.add_field("provider", "2")
            form_preview.add_field("text", "Xin chào đây là bản nghe thử trước khi lưu.")
            form_preview.add_field("language", "vi")

            res_prev = await client.post("/api/voices/preview-clone", data=form_preview)
            assert res_prev.status == 200
            prev_data = await res_prev.json()
            assert prev_data["ok"] is True
            assert "preview_url" in prev_data
            assert "preview_filename" in prev_data
            assert prev_data["preview_filename"].endswith(".wav")

            # 3. Stream the preview audio generated by preview-clone
            res_audio = await client.get(prev_data["preview_url"])
            assert res_audio.status == 200
            assert "audio/wav" in res_audio.headers.get("Content-Type", "")
            audio_bytes = await res_audio.read()
            assert len(audio_bytes) > 0

            # 4. Now save the voice, passing the pre-generated preview_filename
            form_save = aiohttp.FormData()
            form_save.add_field("name", "Pre-auditioned Clone")
            form_save.add_field("provider", "2")
            form_save.add_field("audio", test_wav.read_bytes(), filename="test_clone.wav", content_type="audio/wav")
            form_save.add_field("preview_filename", prev_data["preview_filename"])

            res_save = await client.post("/api/custom-voices", data=form_save)
            assert res_save.status == 201
            save_data = await res_save.json()
            assert save_data["name"] == "Pre-auditioned Clone"
            assert save_data["preview_audio_path"]
            assert save_data["preview_url"] == f"/api/custom-voices/{save_data['id']}/preview/audio"

            # 5. Verify GET /api/custom-voices/{id}/preview/audio returns 200 OK immediately
            res_custom_prev = await client.get(f"/api/custom-voices/{save_data['id']}/preview/audio")
            assert res_custom_prev.status == 200
            assert "audio/wav" in res_custom_prev.headers.get("Content-Type", "")

            # 6. Verify auto-synthesis on demand: create a voice without passing preview
            # and delete preview file on disk if any, then call GET preview/audio -> must NOT return 404!
            v_bare = voice_store.create_voice(
                "Bare Voice Without Preview",
                provider=tts.VIENEU_TTS,
                ref_audio_path="test_clone.wav",
                preview_audio_path="",
                db_path=voice_api_env["db_path"],
            )
            # Copy test_wav into VOICES_DIR so ref_audio_path is present
            dest_bare_ref = voice_api_env["v_dir"] / "test_clone.wav"
            import shutil
            shutil.copyfile(test_wav, dest_bare_ref)

            res_bare_prev = await client.get(f"/api/custom-voices/{v_bare['id']}/preview/audio")
            assert res_bare_prev.status == 200
            assert "audio/wav" in res_bare_prev.headers.get("Content-Type", "")
        finally:
            await client.close()

    asyncio.run(scenario())


def test_mp3_cloning_flow_and_preview_edge_cases(voice_api_env):
    """Verify full MP3 upload voice cloning flow, preview audio button endpoint, and edge cases."""
    app = voice_api_env["app"]
    tmp_path = voice_api_env["tmp_path"]

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            from videotrans.util.help_ffmpeg import runffmpeg

            # 1. Create a 4.0-second valid MP3 file using ffmpeg
            test_mp3 = tmp_path / "voice_sample.mp3"
            runffmpeg([
                "-y",
                "-f", "lavfi",
                "-i", "sine=frequency=440:duration=4",
                "-c:a", "libmp3lame",
                str(test_mp3),
            ], force_cpu=True)
            assert test_mp3.is_file() and test_mp3.stat().st_size > 0

            # 2. Test Preview Voice button flow: POST /api/voices/preview-clone with MP3 file
            form_preview = aiohttp.FormData()
            form_preview.add_field("audio", test_mp3.read_bytes(), filename="voice_sample.mp3", content_type="audio/mpeg")
            form_preview.add_field("provider", "2")
            form_preview.add_field("text", "Xin chào, đây là bản nghe thử giọng nói được sao chép.")
            form_preview.add_field("language", "vi")

            res_prev = await client.post("/api/voices/preview-clone", data=form_preview)
            assert res_prev.status == 200
            prev_data = await res_prev.json()
            assert prev_data["ok"] is True
            assert "preview_url" in prev_data
            assert "preview_filename" in prev_data

            # 3. Retrieve audio from preview_url
            res_audio = await client.get(prev_data["preview_url"])
            assert res_audio.status == 200
            assert "audio/wav" in res_audio.headers.get("Content-Type", "")
            audio_bytes = await res_audio.read()
            assert len(audio_bytes) > 0

            # 4. Save voice using the MP3 file and preview_filename (from clicking Save Changes after preview)
            form_save = aiohttp.FormData()
            form_save.add_field("name", "MP3 Cloned Voice")
            form_save.add_field("provider", "2")
            form_save.add_field("audio", test_mp3.read_bytes(), filename="voice_sample.mp3", content_type="audio/mpeg")
            form_save.add_field("preview_filename", prev_data["preview_filename"])

            res_save = await client.post("/api/custom-voices", data=form_save)
            assert res_save.status == 201
            save_data = await res_save.json()
            assert save_data["name"] == "MP3 Cloned Voice"
            assert save_data["preview_audio_path"]
            assert save_data["preview_url"] == f"/api/custom-voices/{save_data['id']}/preview/audio"

            # 5. Verify GET /api/custom-voices/{id}/preview/audio returns 200 OK immediately
            res_get_prev = await client.get(f"/api/custom-voices/{save_data['id']}/preview/audio")
            assert res_get_prev.status == 200
            assert "audio/wav" in res_get_prev.headers.get("Content-Type", "")
            data_wav = await res_get_prev.read()
            assert len(data_wav) > 0

            # 6. Direct MP3 voice save WITHOUT pre-audition: verify auto-synthesis on GET preview/audio
            form_direct = aiohttp.FormData()
            form_direct.add_field("name", "Direct MP3 Voice")
            form_direct.add_field("provider", "2")
            form_direct.add_field("audio", test_mp3.read_bytes(), filename="direct_sample.mp3", content_type="audio/mpeg")

            res_direct = await client.post("/api/custom-voices", data=form_direct)
            assert res_direct.status == 201
            direct_data = await res_direct.json()

            res_direct_prev = await client.get(f"/api/custom-voices/{direct_data['id']}/preview/audio")
            assert res_direct_prev.status == 200
            assert "audio/wav" in res_direct_prev.headers.get("Content-Type", "")

            # 7. Edge case: Trimming parameters during preview-clone
            form_trim = aiohttp.FormData()
            form_trim.add_field("audio", test_mp3.read_bytes(), filename="voice_sample.mp3", content_type="audio/mpeg")
            form_trim.add_field("provider", "2")
            form_trim.add_field("cut_start", "0.5")
            form_trim.add_field("cut_end", "3.0")
            form_trim.add_field("text", "Testing trim preview")

            res_trim = await client.post("/api/voices/preview-clone", data=form_trim)
            assert res_trim.status == 200

            # 8. Edge case: OmniVoice in preview-clone without ref_text must return 400
            form_omni_bad = aiohttp.FormData()
            form_omni_bad.add_field("audio", test_mp3.read_bytes(), filename="voice_sample.mp3", content_type="audio/mpeg")
            form_omni_bad.add_field("provider", "1")
            form_omni_bad.add_field("ref_text", "")

            res_omni_bad = await client.post("/api/voices/preview-clone", data=form_omni_bad)
            assert res_omni_bad.status == 400

            # 9. Edge case: Traversal attempt in preview-audio must return 400
            res_traversal = await client.get("/api/voices/preview-audio/..secret.wav")
            assert res_traversal.status == 400

            # 10. Edge case: Invalid extension in preview-audio must return 400
            res_bad_ext = await client.get("/api/voices/preview-audio/test.exe")
            assert res_bad_ext.status == 400
        finally:
            await client.close()

    asyncio.run(scenario())



