# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from videotrans.core.db import init_db, set_db_path, get_db_path
from videotrans.core.db import db_conn
from videotrans.core.project_store import (
    create_project,
    get_project,
    list_projects,
    update_project,
    update_project_state,
    delete_project,
    find_project_by_audio_hash,
    get_project_dir,
)
from videotrans.util._srt_parse import get_subtitle_from_srt


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    orig_path = get_db_path()
    test_db = tmp_path / "test_projects.db"
    set_db_path(test_db)
    init_db()
    yield test_db
    set_db_path(orig_path)


def test_create_and_get_project():
    proj = create_project(
        name="Test Video Dub",
        media_id="media_123",
        duration=120.5,
        stage=1,
        status="pending",
        audio_hash="abc123hash",
        state={"segments": [{"id": 1, "text": "hello"}]},
    )
    assert proj["id"] is not None
    assert proj["name"] == "Test Video Dub"
    assert proj["media_id"] == "media_123"
    assert proj["duration"] == 120.5
    assert proj["stage"] == 1
    assert proj["status"] == "pending"
    assert proj["audio_hash"] == "abc123hash"
    assert proj["state"]["segments"][0]["text"] == "hello"

    fetched = get_project(proj["id"])
    assert fetched is not None
    assert fetched["id"] == proj["id"]
    assert fetched["name"] == "Test Video Dub"
    assert fetched["state"]["segments"][0]["text"] == "hello"


def test_nonfinite_duration_is_rejected_and_legacy_record_is_readable():
    with pytest.raises(ValueError, match="duration must be finite"):
        create_project(name="Invalid", duration=float("nan"))

    project = create_project(name="Legacy")
    with pytest.raises(ValueError, match="duration must be finite"):
        update_project(project["id"], duration=float("inf"))

    with db_conn() as conn:
        conn.execute("UPDATE projects SET duration = ? WHERE id = ?", ("nan", project["id"]))
    assert get_project(project["id"])["duration"] == 0.0
    assert list_projects()[0]["duration"] == 0.0


def test_update_project_state_and_stage():
    proj = create_project(name="Project Stage Flow")
    assert proj["stage"] == 1
    assert proj["status"] == "pending"

    updated = update_project_state(
        proj["id"],
        state_dict={"segments": [{"id": 1, "text": "translated"}]},
        stage=2,
        status="completed",
    )
    assert updated is not None
    assert updated["stage"] == 2
    assert updated["status"] == "completed"
    assert updated["state"]["segments"][0]["text"] == "translated"


def test_source_srt_replaces_only_changed_cue_with_canonical_timestamps(tmp_path, monkeypatch):
    monkeypatch.setattr("videotrans.core.project_store.get_projects_root", lambda: tmp_path)
    project = create_project(name="OCR subtitles")
    segments = [
        {"id": 10, "startSec": 1.25, "endSec": 2.5, "startTime": "00:01.250", "endTime": "00:02.500", "sourceText": "Old", "targetText": "Target"},
        {"id": 30, "startSec": 2.5, "endSec": 3.75, "startTime": "00:02.500", "endTime": "00:03.750", "sourceText": "Keep"},
    ]
    update_project_state(project["id"], {"segments": segments})
    segments[0] = {**segments[0], "sourceText": "Xin chào"}
    update_project_state(project["id"], {"segments": segments})

    srt = (get_project_dir(project["id"]) / "transcripts" / "source.srt").read_text(encoding="utf-8")
    assert srt == "1\n00:00:01,250 --> 00:00:02,500\nXin chào\n\n2\n00:00:02,500 --> 00:00:03,750\nKeep\n"
    parsed = get_subtitle_from_srt(srt, is_file=False)
    assert [cue["text"] for cue in parsed] == ["Xin chào", "Keep"]
    assert get_project(project["id"])["state"]["segments"] == segments


def test_source_srt_write_failure_is_not_reported_as_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("videotrans.core.project_store.get_projects_root", lambda: tmp_path)
    project = create_project(name="OCR save failure", state={"segments": []})
    original_write = Path.write_text

    def fail_srt_write(path, *args, **kwargs):
        if path.name == "source.srt":
            raise OSError("disk unavailable")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_srt_write)
    with pytest.raises(OSError, match="disk unavailable"):
        update_project_state(project["id"], {"segments": [{"id": 1, "startSec": 0, "endSec": 1, "sourceText": "OCR"}]})
    assert get_project(project["id"])["state"] == {"segments": []}


def test_source_srt_normalizes_legacy_time_strings(tmp_path, monkeypatch):
    monkeypatch.setattr("videotrans.core.project_store.get_projects_root", lambda: tmp_path)
    project = create_project(name="Legacy OCR subtitles")
    update_project_state(project["id"], {"segments": [
        {"id": 9, "startTime": "00:01.250", "endTime": "00:02.500", "sourceText": "Legacy"},
    ]})
    srt = (get_project_dir(project["id"]) / "transcripts" / "source.srt").read_text(encoding="utf-8")
    assert srt == "1\n00:00:01,250 --> 00:00:02,500\nLegacy\n"


def test_list_projects_order():
    p1 = create_project(name="Old Project")
    p2 = create_project(name="Newer Project")

    # Update p1 so it becomes newest
    update_project(p1["id"], name="Old Project Updated")

    projects = list_projects(limit=10)
    assert len(projects) >= 2
    assert projects[0]["id"] == p1["id"]
    assert projects[1]["id"] == p2["id"]


def test_find_project_by_audio_hash():
    proj = create_project(
        name="Audio Dub",
        audio_hash="unique_hash_987",
        state={"segments": [{"id": 1, "text": "cached"}]},
    )
    found = find_project_by_audio_hash("unique_hash_987")
    assert found is not None
    assert found["id"] == proj["id"]
    assert found["state"]["segments"][0]["text"] == "cached"

    assert find_project_by_audio_hash("nonexistent_hash") is None
    assert find_project_by_audio_hash("") is None


def test_delete_project_and_dirs():
    proj = create_project(name="To Delete")
    pdir = get_project_dir(proj["id"])
    assert pdir.exists()

    deleted = delete_project(proj["id"], delete_files=True)
    assert deleted is True
    assert get_project(proj["id"]) is None
    assert not pdir.exists()


def test_update_project_state_merges_by_default_and_replaces_when_merge_false(tmp_path, monkeypatch):
    monkeypatch.setattr("videotrans.core.project_store.get_projects_root", lambda: tmp_path)
    proj = create_project(
        name="Partial Merge Test",
        parent_id="parent_2026-11-18",
        state={
            "parent_id": "parent_2026-11-18",
            "project": {"filename": "clip.mp4", "verified": True},
            "languages": {"source": {"code": "zh-cn"}, "target": {"code": "vi"}},
            "backend": {"mediaId": "m_1", "config": {"recognType": 1}},
            "engines": {"speakerDiarization": True},
            "ocrCrop": {"roi": [0.1, 0.7, 0.8, 0.2]},
        },
    )

    merged = update_project_state(
        proj["id"],
        {"segments": [{"id": 1, "startSec": 0, "endSec": 1, "sourceText": "Hi"}], "currentStep": 2},
        stage=2,
    )
    assert merged is not None
    assert merged["state"]["parent_id"] == "parent_2026-11-18"
    assert merged["state"]["project"]["filename"] == "clip.mp4"
    assert merged["state"]["languages"]["target"]["code"] == "vi"
    assert merged["state"]["backend"]["mediaId"] == "m_1"
    assert merged["state"]["engines"]["speakerDiarization"] is True
    assert merged["state"]["currentStep"] == 2
    assert len(merged["state"]["segments"]) == 1

    replaced = update_project_state(
        proj["id"],
        {"currentStep": 2, "segments": []},
        stage=2,
        merge=False,
    )
    assert replaced is not None
    assert "ocrCrop" not in replaced["state"]
    assert "engines" not in replaced["state"]
