# -*- coding: utf-8 -*-
import io
import json
from pathlib import Path
import tempfile
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient

from videotrans import tts
from videotrans.core import db, voice_store
from videotrans.api.app import app
from videotrans.tts._vieneutts import VieNeuTTS


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_voices.db"
    db.init_db(db_file)
    original_voices_dir = voice_store.VOICES_DIR
    voice_store.VOICES_DIR = tmp_path / "voices"
    voice_store.VOICES_DIR.mkdir(parents=True, exist_ok=True)
    yield db_file
    voice_store.VOICES_DIR = original_voices_dir


def test_embedding_caching(temp_db, tmp_path):
    voice_id = "test_voice_123"
    fake_emb = np.random.randn(192).astype(np.float32)
    fake_codes = np.random.randint(0, 1024, size=(50, 16)).astype(np.int64)

    npz_path = voice_store.cache_voice_embedding(
        voice_id,
        fake_emb,
        fake_codes,
        base_dir=tmp_path / "voices",
    )
    assert npz_path.is_file()

    loaded = voice_store.get_voice_embedding(voice_id, base_dir=tmp_path / "voices")
    assert loaded is not None
    loaded_emb, loaded_codes = loaded
    assert np.allclose(fake_emb, loaded_emb)
    assert np.array_equal(fake_codes, loaded_codes)


def test_resolve_voice_params_cloned_and_designed(temp_db):
    # 1. Create a cloned base voice
    clone_v = voice_store.create_voice(
        name="Cloned Speaker",
        provider=tts.VIENEU_TTS,
        ref_audio_path="cloned_sample.wav",
        kind="clone",
        tuning_params={"denoise": True},
        db_path=temp_db,
    )

    resolved_clone = voice_store.resolve_voice_params(clone_v["id"], db_path=temp_db)
    assert resolved_clone is not None
    assert resolved_clone["kind"] == "clone"
    assert resolved_clone["ref_audio_path"] == "cloned_sample.wav"

    # 2. Create a designed voice based on the cloned voice
    design_v1 = voice_store.create_voice(
        name="Dramatic Narrator",
        provider=tts.VIENEU_TTS,
        kind="design",
        external_voice_id=clone_v["id"],
        tuning_params={
            "style": "doc_truyen",
            "temperature": 0.95,
            "base_voice": clone_v["id"],
        },
        db_path=temp_db,
    )

    resolved_design1 = voice_store.resolve_voice_params(design_v1["id"], db_path=temp_db)
    assert resolved_design1 is not None
    assert resolved_design1["kind"] == "design"
    assert resolved_design1["base_type"] == "custom"
    assert resolved_design1["ref_audio_path"] == "cloned_sample.wav"
    assert resolved_design1["tuning_params"]["style"] == "doc_truyen"

    # 3. Create a designed voice based on a preset
    design_v2 = voice_store.create_voice(
        name="News Anchor Female",
        provider=tts.VIENEU_TTS,
        kind="design",
        external_voice_id="Ly (nữ miền Bắc)",
        tuning_params={
            "style": "tin_tuc",
            "temperature": 0.7,
            "base_voice": "Ly (nữ miền Bắc)",
        },
        db_path=temp_db,
    )

    resolved_design2 = voice_store.resolve_voice_params(design_v2["id"], db_path=temp_db)
    assert resolved_design2 is not None
    assert resolved_design2["kind"] == "design"
    assert resolved_design2["base_type"] == "preset"
    assert resolved_design2["preset_voice"] == "Ly (nữ miền Bắc)"
    assert resolved_design2["tuning_params"]["style"] == "tin_tuc"


def test_api_voice_design_assist():
    client = TestClient(app)

    # Test news prompt
    resp_news = client.post("/api/voices/design-assist", json={"prompt": "MC đọc bản tin thời sự sáng nay trang trọng"})
    assert resp_news.status_code == 200
    data_news = resp_news.json()
    assert data_news["style"] == "tin_tuc"
    assert data_news["temperature"] <= 0.75

    # Test storytelling with emotions
    resp_story = client.post("/api/voices/design-assist", json={"prompt": "Giọng kể chuyện cổ tích đêm khuya vui vẻ có tiếng cười"})
    assert resp_story.status_code == 200
    data_story = resp_story.json()
    assert data_story["style"] == "doc_truyen"
    assert "[cười]" in data_story["suggested_tags"]


def test_api_create_designed_voice(temp_db, monkeypatch):
    from videotrans.services import voice_preview

    def fake_synthesizer(voice, preview_path, sample_text):
        with wave.open(str(preview_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x01\x00" * 24000)

    monkeypatch.setattr(voice_preview, "_preview_synthesizer", fake_synthesizer)
    client = TestClient(app)
    monkeypatch.setattr(voice_store, "db_conn", lambda p=None: db.db_conn(temp_db))

    payload = {
        "name": "Storyteller V3",
        "provider": 2,
        "kind": "design",
        "external_voice_id": "Bình (nam miền Bắc)",
        "tuning_params": {
            "style": "doc_truyen",
            "temperature": 0.9,
            "base_voice": "Bình (nam miền Bắc)",
            "emotions": ["[thở dài]"],
        },
    }
    resp = client.post("/api/custom-voices", json=payload)
    assert resp.status_code == 201
    created = resp.json()
    assert created["kind"] == "design"
    assert created["name"] == "Storyteller V3"
    assert created["external_voice_id"] == "Bình (nam miền Bắc)"


def test_api_unified_preview_with_style(temp_db, monkeypatch):
    from videotrans.services import voice_preview
    # Mock synthesize_unified_tts_preview or synthesizer double
    recorded_calls = []

    def fake_double(voice_dict, p_path, sample_text):
        recorded_calls.append((voice_dict, sample_text))
        p_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    monkeypatch.setattr(voice_preview, "_preview_synthesizer", fake_double)

    client = TestClient(app)
    resp = client.post("/api/tts/preview", json={
        "voice": "Bình (nam miền Bắc)",
        "provider": 2,
        "text": "Chào bạn [cười] đây là bản thử",
        "style": "doc_truyen",
        "tuningParams": {"temperature": 0.85},
    })
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["ok"] is True
    assert "preview_url" in res_data
    assert len(recorded_calls) == 1
    assert recorded_calls[0][0]["style"] == "doc_truyen"


def test_api_unified_preview_unmocked_standard_cache(temp_db, monkeypatch):
    from videotrans.services import voice_preview
    # A failed VieNeu engine must not be reported as a successful preview.
    monkeypatch.setattr(voice_preview, "_preview_synthesizer", None)
    def fail_synthesis(*args, **kwargs):
        raise RuntimeError("VieNeu engine unavailable")
    monkeypatch.setattr(voice_preview, "_run_vieneu_synthesis", fail_synthesis)
    client = TestClient(app)
    resp = client.post("/api/tts/preview", json={
        "voice": "Bình (nam miền Bắc)",
        "provider": 2,
        "text": "Kiểm tra âm thanh xem thử giọng đọc",
        "style": "doc_truyen",
        "tuningParams": {"temperature": 0.85},
    })
    assert resp.status_code == 500
    assert "VieNeu engine unavailable" in resp.json()["detail"]


def test_vieneu_tts_inference_options(temp_db, monkeypatch):
    monkeypatch.setattr(voice_store, "db_conn", lambda p=None: db.db_conn(temp_db))

    # Mock engine creation for VieNeuTTS
    class FakeEngine:
        sample_rate = 48000
        def get_preset_voice(self, name):
            return {"name": name, "preset": True}

    monkeypatch.setattr(VieNeuTTS, "_create_engine", lambda self: FakeEngine())

    tts_engine = VieNeuTTS(queue_tts=[{"text": "test", "filename": "test.wav"}])

    # 1. Preset voice with default style
    v_key, opts = tts_engine._inference_options({"role": "Bình (nam miền Bắc)"})
    assert v_key[0] == "preset"
    assert opts.get("style", "tu_nhien") == "tu_nhien"

    # 2. Preset voice with segment-level style override
    v_key, opts = tts_engine._inference_options({"role": "Bình (nam miền Bắc)", "style": "tin_tuc"})
    assert opts["style"] == "tin_tuc"

    # 3. Custom designed voice
    dv = voice_store.create_voice(
        name="Storyteller Test",
        provider=tts.VIENEU_TTS,
        kind="design",
        external_voice_id="Ly (nữ miền Bắc)",
        tuning_params={"style": "doc_truyen", "temperature": 0.9},
        db_path=temp_db,
    )

    v_key, opts = tts_engine._inference_options({"role": dv["id"]})
    assert opts["style"] == "doc_truyen"
    assert opts["temperature"] == 0.9
