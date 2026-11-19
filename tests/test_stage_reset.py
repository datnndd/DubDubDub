import json

import pytest
from fastapi.testclient import TestClient

from videotrans.api.app import create_app
from videotrans.api.task_params import build_task_params
from videotrans.core import project_store
from videotrans.core.db import get_db_path, init_db, set_db_path
from videotrans.core.job_store import append_event, create_job, mark_done
from videotrans.core.project_store import create_project, get_project, get_project_dir, update_project
from videotrans.core.stage_reset import BaselineMissingError, reset_project_stage
from videotrans.task.orchestrator import TaskRequest


@pytest.fixture
def isolated_projects(tmp_path, monkeypatch):
    previous_db = get_db_path()
    set_db_path(tmp_path / "projects.db")
    init_db()
    monkeypatch.setattr(project_store, "get_projects_root", lambda: tmp_path / "projects")
    yield tmp_path
    set_db_path(previous_db)


def _project(tmp_path):
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original media")
    state = {
        "project": {"filename": "source.mp4", "verified": True},
        "backend": {"mediaId": "media-1", "config": {"recognType": 2, "translateType": 2, "ttsType": 3}},
        "engines": {"speakerDiarization": True},
        "segments": [{"id": 1, "sourceText": "edited", "targetText": "translated", "previewAudioUrl": "/preview/1"}],
        "speakerVoiceMap": {"spk_1": "voice-1"},
        "dubbingStatus": "completed",
        "stage3Baseline": [{"id": 1, "sourceText": "ASR", "targetText": "translated", "previewAudioUrl": "/preview/1"}],
        "subtitleStyles": {"fontSize": 42},
        "editVideo": {"audioMix": {"original": 50}},
    }
    create_project(project_id="reset-test", media_id="media-1", media_path=str(source), stage=4, state=state, audio_hash="abc")
    base = get_project_dir("reset-test")
    (base / "transcripts" / "asr_baseline.json").write_text(json.dumps({"segments": [{"id": 1, "sourceText": "ASR", "targetText": ""}]}), encoding="utf-8")
    (base / "transcripts" / "target.srt").write_text("translated", encoding="utf-8")
    (base / "dubbing" / "voice.wav").write_bytes(b"voice")
    (base / "exports" / "final.mp4").write_bytes(b"export")
    return base


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_reset_boundaries_preserve_media_and_exports(isolated_projects, stage):
    base = _project(isolated_projects)
    updated = reset_project_stage("reset-test", stage)
    state = updated["state"]
    assert updated["stage"] == stage
    assert state["backend"]["mediaId"] == "media-1"
    assert (isolated_projects / "original.mp4").read_bytes() == b"original media"
    assert state["project"]["verified"] is True
    assert (base / "exports" / "final.mp4").is_file()
    assert state["editVideo"] is None
    if stage == 1:
        assert state["segments"] == []
        assert state["forceAsr"] is True
        assert updated["audio_hash"] == ""
        assert not (base / "transcripts" / "asr_baseline.json").exists()
    elif stage == 2:
        assert state["segments"][0]["sourceText"] == "ASR"
        assert state["segments"][0]["targetText"] == ""
        assert not (base / "transcripts" / "target.srt").exists()
    elif stage == 3:
        assert state["segments"][0]["targetText"] == "translated"
        assert "previewAudioUrl" not in state["segments"][0]
        assert (base / "transcripts" / "target.srt").is_file()
    else:
        assert state["segments"] == state["stage3Baseline"]
        assert (base / "dubbing" / "voice.wav").is_file()
    if stage <= 3:
        assert state["dubbingStatus"] == "idle"
        assert not (base / "dubbing" / "voice.wav").exists()


def test_legacy_stage2_recovery_and_missing_baseline(isolated_projects):
    base = _project(isolated_projects)
    (base / "transcripts" / "asr_baseline.json").unlink()
    with pytest.raises(BaselineMissingError):
        reset_project_stage("reset-test", 2)
    create_job("asr-legacy", type="asr", project_id="reset-test")
    append_event("asr-legacy", {"kind": "succeeded", "details": {"segments": [{"id": 1, "sourceText": "original"}]}})
    mark_done("asr-legacy")
    assert reset_project_stage("reset-test", 2)["state"]["segments"][0]["sourceText"] == "original"


def test_reset_endpoint_rejects_active_job(isolated_projects):
    _project(isolated_projects)
    app = create_app()
    with TestClient(app) as client:
        create_job("active-reset", type="asr", project_id="reset-test")
        response = client.post("/api/projects/reset-test/stages/3/reset")
    assert response.status_code == 409


def test_reset_endpoint_persists_result_for_project_reload(isolated_projects):
    _project(isolated_projects)
    with TestClient(create_app()) as client:
        reset = client.post("/api/projects/reset-test/stages/2/reset")
        restored = client.get("/api/projects/reset-test")
    assert reset.status_code == 200
    assert restored.status_code == 200
    assert restored.json()["stage"] == 2
    assert restored.json()["state"]["segments"][0]["sourceText"] == "ASR"


def test_stage4_reset_restores_baseline_after_editor_text_change(isolated_projects):
    _project(isolated_projects)
    project = get_project("reset-test")
    update_project("reset-test", state={**project["state"], "dubbingStatus": "idle"})
    restored = reset_project_stage("reset-test", 4)
    assert restored["state"]["segments"][0]["sourceText"] == "ASR"


def test_deepgram_diarization_request_has_only_supported_task_fields(tmp_path):
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video")
    params = build_task_params(source, {
        "recognType": 1, "modelName": "nova-3", "speakerDiarization": True,
        "speakerCount": 2, "forceAsr": True, "autoFitVoiceSpeed": True,
        "maxSpeedRate": 1.25,
    }, job_type="asr")
    values = TaskRequest(params).normalize()
    assert values["enable_diariz"] is True
    assert values["force_recogn"] is True
    assert "auto_speed" not in values and "max_speed_rate" not in values
    with pytest.raises(ValueError, match="Unsupported task parameters: unknown_option"):
        TaskRequest({**params, "unknown_option": True}).normalize()


def test_stage2_reset_clears_ocr_crop_permanently(isolated_projects):
    _project(isolated_projects)
    proj = get_project("reset-test")
    update_project("reset-test", state={**proj["state"], "ocrCrop": {"roi": [0.1, 0.7, 0.8, 0.2]}})
    assert "ocrCrop" in get_project("reset-test")["state"]

    with TestClient(create_app()) as client:
        reset_res = client.post("/api/projects/reset-test/stages/2/reset")
        assert reset_res.status_code == 200
        assert "ocrCrop" not in reset_res.json()["state"]

        get_res = client.get("/api/projects/reset-test")
        assert get_res.status_code == 200
        assert "ocrCrop" not in get_res.json()["state"]
