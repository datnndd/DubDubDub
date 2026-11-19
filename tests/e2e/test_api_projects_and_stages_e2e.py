# -*- coding: utf-8 -*-
"""Opaque-box E2E tests for Backend API, Project State & Stages (F1, F2, F5)."""
from __future__ import annotations

import asyncio
import datetime
import io
import json
from pathlib import Path
import time
import wave
import zipfile

import httpx
import numpy as np
import pytest
import soundfile as sf

from tests.e2e.conftest import make_mp4, make_png, make_wav


def _poll_job_until_done(client, job_id: str, timeout_s: float = 25.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last.get("status") in {"succeeded", "failed", "cancelled"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not finish within {timeout_s}s; last={last}")


# ============================================================================
# F1: Batch Date-Parent Grouping & Folder Scan (Tier 1 & Tier 2)
# ============================================================================


def test_f1_scan_folders_recursive_and_non_recursive_discovers_mp4s(e2e_client, tmp_path: Path):
    root_dir = tmp_path / "batch_media"
    ep1 = root_dir / "ep1"
    ep2 = root_dir / "ep2"
    make_mp4(root_dir / "top.mp4", duration_s=1.0)
    make_mp4(ep1 / "video.mp4", duration_s=1.0)
    make_mp4(ep2 / "video.mp4", duration_s=1.0)
    (ep1 / "notes.txt").write_text("ignore", encoding="utf-8")

    res_rec = e2e_client.post("/api/media/scan-folders", json={"paths": [str(root_dir)], "recursive": True})
    assert res_rec.status_code == 200, res_rec.text
    body_rec = res_rec.json()
    assert body_rec["ok"] is True
    assert body_rec["count"] == 3
    assert len({v["path"] for v in body_rec["videos"]}) == 3

    res_flat = e2e_client.post("/api/media/scan-folders", json={"paths": [str(root_dir)], "recursive": False})
    assert res_flat.status_code == 200
    body_flat = res_flat.json()
    assert body_flat["count"] == 1
    assert body_flat["videos"][0]["filename"] == "top.mp4"


def test_f1_register_local_and_upload_media_probe_real_mp4(e2e_client, tmp_path: Path):
    local_mp4 = make_mp4(tmp_path / "local_clip.mp4", duration_s=1.5, width=320, height=240, with_audio=True)
    reg_res = e2e_client.post("/api/media/register-local", json={"paths": [str(local_mp4)]})
    assert reg_res.status_code == 201, reg_res.text
    reg_body = reg_res.json()
    assert reg_body["ok"] is True
    assert reg_body["count"] == 1
    rec = reg_body["media"][0]
    assert rec["filename"] == "local_clip.mp4"
    assert rec["resolution"] == "320x240"
    assert rec["hasVideo"] is True
    assert rec["hasAudio"] is True
    assert rec["durationMs"] >= 1000

    with local_mp4.open("rb") as fh:
        up_res = e2e_client.post("/api/media", files={"file": ("uploaded_clip.mp4", fh, "video/mp4")})
    assert up_res.status_code == 201, up_res.text
    up_body = up_res.json()
    assert up_body["filename"] == "uploaded_clip.mp4"
    assert up_body["resolution"] == "320x240"

    dl_res = e2e_client.get(f"/api/media/{up_body['id']}/file")
    assert dl_res.status_code == 200
    assert len(dl_res.content) == local_mp4.stat().st_size


def test_f1_today_parent_uses_local_calendar_date(e2e_client):
    expected_today = datetime.date.today().isoformat()
    res = e2e_client.get("/api/projects/today-parent")
    assert res.status_code == 200, res.text
    data = res.json()
    parent = data["parent"]
    assert parent["id"] == f"parent_{expected_today}"
    assert parent["name"] == expected_today
    assert parent["state"]["is_parent"] is True
    assert parent["state"]["date"] == expected_today
    assert data["subprojects"] == []


def test_f1_batch_create_and_list_subprojects_under_today_parent(e2e_client, tmp_path: Path):
    v1 = make_mp4(tmp_path / "ep1" / "lesson.mp4", duration_s=1.2)
    v2 = make_mp4(tmp_path / "ep2" / "lesson.mp4", duration_s=1.8)
    reg = e2e_client.post("/api/media/register-local", json={"paths": [str(v1), str(v2)]})
    assert reg.status_code == 201
    m1, m2 = reg.json()["media"]

    batch_res = e2e_client.post(
        "/api/projects/batch-create",
        json={
            "items": [
                {"name": "ep1/lesson.mp4", "mediaId": m1["id"], "mediaPath": str(v1)},
                {"name": "ep2/lesson.mp4", "mediaId": m2["id"], "mediaPath": str(v2)},
            ]
        },
    )
    assert batch_res.status_code == 201, batch_res.text
    batch_body = batch_res.json()
    assert batch_body["ok"] is True
    parent_id = batch_body["parent"]["id"]
    assert parent_id == f"parent_{datetime.date.today().isoformat()}"
    assert len(batch_body["projects"]) == 2
    assert all(p["parent_id"] == parent_id for p in batch_body["projects"])

    sub_res = e2e_client.get(f"/api/projects/{parent_id}/subprojects")
    assert sub_res.status_code == 200
    subs = sub_res.json()["subprojects"]
    assert [s["name"] for s in subs] == ["ep1/lesson.mp4", "ep2/lesson.mp4"]
    assert all(s["duration"] >= 1.0 for s in subs)


def test_f1_batch_create_explicit_date_appends_without_overwriting(e2e_client, tmp_path: Path):
    v1 = make_mp4(tmp_path / "a.mp4", duration_s=1.0)
    v2 = make_mp4(tmp_path / "b.mp4", duration_s=1.0)
    reg = e2e_client.post("/api/media/register-local", json={"paths": [str(v1), str(v2)]}).json()["media"]

    res1 = e2e_client.post(
        "/api/projects/batch-create",
        json={"date": "2026-11-18", "items": [{"name": "a.mp4", "mediaId": reg[0]["id"]}]},
    )
    assert res1.status_code == 201
    parent_id = res1.json()["parent"]["id"]
    assert parent_id == "parent_2026-11-18"

    res2 = e2e_client.post(
        "/api/projects/batch-create",
        json={"parentId": parent_id, "items": [{"name": "b.mp4", "mediaId": reg[1]["id"]}]},
    )
    assert res2.status_code == 201
    subs = e2e_client.get(f"/api/projects/{parent_id}/subprojects").json()["subprojects"]
    assert [s["name"] for s in subs] == ["a.mp4", "b.mp4"]


def test_f1_scan_folders_empty_nonexistent_and_non_mp4_dir(e2e_client, tmp_path: Path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    non_mp4_dir = tmp_path / "audio_only"
    make_wav(non_mp4_dir / "clip.wav", duration_s=0.5)
    (non_mp4_dir / "readme.txt").write_text("hi", encoding="utf-8")

    res = e2e_client.post(
        "/api/media/scan-folders",
        json={"paths": [str(tmp_path / "does_not_exist"), str(empty_dir), str(non_mp4_dir)], "recursive": True},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["count"] == 0
    assert body["videos"] == []


def test_f1_scan_folders_preserves_distinct_paths_with_duplicate_filenames(e2e_client, tmp_path: Path):
    f1 = tmp_path / "season1" / "episode.mp4"
    f2 = tmp_path / "season2" / "episode.mp4"
    make_mp4(f1, duration_s=1.0)
    make_mp4(f2, duration_s=1.0)

    res = e2e_client.post("/api/media/scan-folders", json={"paths": [str(tmp_path / "season1"), str(tmp_path / "season2")]})
    assert res.status_code == 200
    videos = res.json()["videos"]
    assert len(videos) == 2
    assert {v["parentDir"] for v in videos} == {str(f1.parent), str(f2.parent)}


def test_f1_register_local_skips_missing_and_corrupt_files(e2e_client, tmp_path: Path):
    valid = make_mp4(tmp_path / "valid.mp4", duration_s=1.0)
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a real mp4 file")

    res = e2e_client.post(
        "/api/media/register-local",
        json={"paths": ["", str(tmp_path / "missing.mp4"), str(corrupt), str(valid)]},
    )
    assert res.status_code == 201
    body = res.json()
    assert body["count"] == 1
    assert body["media"][0]["filename"] == "valid.mp4"


def test_f1_media_upload_rejects_empty_and_unsupported_extension(e2e_client):
    bad_ext = e2e_client.post("/api/media", files={"file": ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
    assert bad_ext.status_code == 400

    empty_mp4 = e2e_client.post("/api/media", files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")})
    assert empty_mp4.status_code == 400


def test_f1_batch_upload_filters_unsupported_files(e2e_client, tmp_path: Path):
    valid = make_mp4(tmp_path / "good.mp4", duration_s=1.0)
    with valid.open("rb") as fh:
        res = e2e_client.post(
            "/api/media/batch-upload",
            files=[
                ("files", ("good.mp4", fh, "video/mp4")),
                ("files", ("notes.txt", io.BytesIO(b"ignore"), "text/plain")),
            ],
        )
    assert res.status_code == 201
    body = res.json()
    assert body["ok"] is True
    assert body["count"] == 1
    assert body["media"][0]["filename"] == "good.mp4"


# ============================================================================
# F2: Batch Prepare Concurrency & State Merge/Reset (Tier 1 & Tier 2)
# ============================================================================


def test_f2_put_project_state_shallow_merge_preserves_metadata(e2e_client):
    """M2 Contract: PUT /api/projects/{id}/state shallow-merges state_json when merge=True."""
    initial_state = {
        "parent_id": "parent_2026-11-18",
        "project": {"filename": "demo.mp4", "duration": 12.5},
        "languages": {"source": {"code": "en"}, "target": {"code": "vi"}},
        "backend": {"mediaId": "media_123", "config": {"recognType": 1}},
        "engines": {"speakerDiarization": True, "speakerCount": 2},
        "currentStep": 1,
    }
    created = e2e_client.post(
        "/api/projects",
        json={"name": "Merge Test", "parentId": "parent_2026-11-18", "state": initial_state},
    )
    assert created.status_code == 201
    pid = created.json()["id"]

    partial_segments = [{"id": 1, "startSec": 0.0, "endSec": 1.5, "sourceText": "Hello", "targetText": ""}]
    upd = e2e_client.put(
        f"/api/projects/{pid}/state",
        json={"stage": 2, "state": {"segments": partial_segments, "currentStep": 2}},
    )
    assert upd.status_code == 200, upd.text

    fetched = e2e_client.get(f"/api/projects/{pid}").json()
    st = fetched["state"]
    assert fetched["stage"] == 2
    assert st["segments"] == partial_segments
    assert st["currentStep"] == 2
    assert st["parent_id"] == "parent_2026-11-18"
    assert st["project"] == {"filename": "demo.mp4", "duration": 12.5}
    assert st["languages"] == {"source": {"code": "en"}, "target": {"code": "vi"}}
    assert st["backend"] == {"mediaId": "media_123", "config": {"recognType": 1}}
    assert st["engines"] == {"speakerDiarization": True, "speakerCount": 2}


def test_f2_put_project_state_writes_transcript_artifacts(e2e_client):
    from videotrans.core.project_store import get_project_dir

    pid = e2e_client.post("/api/projects", json={"name": "Artifacts Test"}).json()["id"]
    segs = [{"id": 1, "startSec": 0.5, "endSec": 2.0, "sourceText": "First subtitle", "targetText": ""}]
    res = e2e_client.put(
        f"/api/projects/{pid}/state",
        json={"state": {"segments": segs, "transcript_options": {"utterances": segs}}},
    )
    assert res.status_code == 200

    t_dir = get_project_dir(pid) / "transcripts"
    assert (t_dir / "source.srt").is_file()
    assert "First subtitle" in (t_dir / "source.srt").read_text(encoding="utf-8")
    assert json.loads((t_dir / "segments.json").read_text(encoding="utf-8")) == segs
    assert json.loads((t_dir / "transcript_options.json").read_text(encoding="utf-8")) == {"utterances": segs}


def test_f2_stage2_reset_restores_asr_baseline_and_clears_ocr_crop(e2e_client):
    from videotrans.core.project_store import get_project_dir

    pid = e2e_client.post("/api/projects", json={"name": "Reset Stage 2"}).json()["id"]
    baseline_segs = [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Original ASR", "targetText": ""}]
    edited_segs = [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "OCR Edited", "targetText": "Translated"}]

    t_dir = get_project_dir(pid) / "transcripts"
    t_dir.mkdir(parents=True, exist_ok=True)
    (t_dir / "asr_baseline.json").write_text(
        json.dumps({"segments": baseline_segs, "transcriptOptions": {"source": "asr"}}),
        encoding="utf-8",
    )
    (t_dir / "target.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nTranslated\n", encoding="utf-8")

    e2e_client.put(
        f"/api/projects/{pid}/state",
        json={"stage": 3, "state": {"segments": edited_segs, "ocrCrop": {"x": 0.1, "y": 0.8, "w": 0.8, "h": 0.15}}},
    )

    reset_res = e2e_client.post(f"/api/projects/{pid}/stages/2/reset")
    assert reset_res.status_code == 200, reset_res.text
    reset_proj = reset_res.json()
    assert reset_proj["stage"] == 2
    assert reset_proj["state"]["segments"] == baseline_segs
    assert "ocrCrop" not in reset_proj["state"]
    assert not (t_dir / "target.srt").exists()


def test_f2_stage1_reset_clears_segments_audio_hash_and_files(e2e_client):
    from videotrans.core.project_store import get_project_dir, update_project

    pid = e2e_client.post("/api/projects", json={"name": "Reset Stage 1"}).json()["id"]
    update_project(pid, audio_hash="deadbeef1234")
    pdir = get_project_dir(pid)
    (pdir / "media" / "clip.wav").write_bytes(b"RIFF")
    (pdir / "transcripts" / "source.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    e2e_client.put(
        f"/api/projects/{pid}/state",
        json={"stage": 2, "state": {"segments": [{"id": 1, "startSec": 0, "endSec": 1, "sourceText": "Hi"}]}},
    )

    res = e2e_client.post(f"/api/projects/{pid}/stages/1/reset")
    assert res.status_code == 200, res.text
    proj = res.json()
    assert proj["stage"] == 1
    assert proj["audio_hash"] == ""
    assert proj["state"]["segments"] == []
    assert proj["state"]["forceAsr"] is True
    assert not (pdir / "media" / "clip.wav").exists()
    assert not (pdir / "transcripts" / "source.srt").exists()


def test_f2_job_sse_stream_emits_events_and_terminal_done(e2e_app, e2e_client, local_llm_server, tmp_path: Path):
    from videotrans import translator
    from videotrans.configure import config as runtime_config

    runtime_config.params.chatgpt_api = local_llm_server.openai_url
    runtime_config.params["chatgpt_key"] = "test-openai-key"
    runtime_config.params.chatgpt_model = "gpt-5-mini"

    mp4 = make_mp4(tmp_path / "sse_demo.mp4", duration_s=1.2)
    media = e2e_client.post("/api/media/register-local", json={"paths": [str(mp4)]}).json()["media"][0]
    proj = e2e_client.post("/api/projects", json={"name": "SSE Project", "mediaId": media["id"]}).json()

    job_res = e2e_client.post(
        "/api/jobs",
        json={
            "mediaId": media["id"],
            "projectId": proj["id"],
            "jobType": "translation",
            "options": {
                "sourceLanguage": "en",
                "targetLanguage": "vi",
                "translateType": translator.CHATGPT_INDEX,
                "translationMode": "srt",
                "segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Hello world", "targetText": ""}],
            },
        },
    )
    assert job_res.status_code == 202, job_res.text
    job_id = job_res.json()["id"]

    async def _consume_sse():
        transport = httpx.ASGITransport(app=e2e_app)
        events = []
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream("GET", f"/api/jobs/{job_id}/stream", timeout=15.0) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers.get("content-type", "")
                cur_event = "message"
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        cur_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        events.append((cur_event, json.loads(line.split(":", 1)[1].strip())))
                        if cur_event == "done":
                            break
        return events

    events = asyncio.run(_consume_sse())
    assert any(ev == "done" and payload.get("terminal") is True for ev, payload in events)
    done_payload = [p for ev, p in events if ev == "done"][-1]
    assert done_payload["status"] == "succeeded"


def test_f2_reset_stage_returns_409_when_active_job_exists(e2e_client):
    from videotrans.core.job_store import create_job, mark_running

    pid = e2e_client.post("/api/projects", json={"name": "Active Job Guard"}).json()["id"]
    create_job("job_active_reset_guard", type="asr", project_id=pid, stage="recogn")
    mark_running("job_active_reset_guard", stage="recogn", message="Running")

    res = e2e_client.post(f"/api/projects/{pid}/stages/1/reset")
    assert res.status_code == 409


def test_f2_stage2_reset_returns_409_when_asr_baseline_missing(e2e_client):
    pid = e2e_client.post("/api/projects", json={"name": "No Baseline", "stage": 2}).json()["id"]
    res = e2e_client.post(f"/api/projects/{pid}/stages/2/reset")
    assert res.status_code == 409


def test_f2_stage4_reset_returns_409_when_stage3_baseline_missing(e2e_client):
    pid = e2e_client.post(
        "/api/projects",
        json={"name": "No Stage3 Baseline", "stage": 4, "state": {"currentStep": 4, "segments": [{"id": 1, "sourceText": "hi"}]}},
    ).json()["id"]
    res = e2e_client.post(f"/api/projects/{pid}/stages/4/reset")
    assert res.status_code == 409


def test_f2_reset_stage_rejects_invalid_stage_and_unknown_project(e2e_client):
    pid = e2e_client.post("/api/projects", json={"name": "Stage Bounds"}).json()["id"]
    assert e2e_client.post(f"/api/projects/{pid}/stages/0/reset").status_code == 400
    assert e2e_client.post(f"/api/projects/{pid}/stages/5/reset").status_code == 400
    assert e2e_client.post("/api/projects/nonexistent-id/stages/1/reset").status_code == 404


def test_f2_duplicate_active_media_job_returns_409_and_cancel_works(e2e_app, e2e_client, tmp_path: Path):
    from videotrans.core.job_manager import JobManager
    from videotrans.task.orchestrator import TaskResult, TaskStatus

    gate = asyncio.Event()
    started = []

    def blocking_runner(request, event_sink=None, cancellation_token=None, stage_limit=None):
        started.append(True)
        while not cancellation_token.is_cancelled():
            time.sleep(0.02)
        return TaskResult("blocked", TaskStatus.CANCELLED, tmp_path)

    e2e_app.state.job_manager = JobManager(runner=blocking_runner)
    mp4 = make_mp4(tmp_path / "dup_guard.mp4", duration_s=1.0)
    media = e2e_client.post("/api/media/register-local", json={"paths": [str(mp4)]}).json()["media"][0]
    proj = e2e_client.post("/api/projects", json={"name": "Dup Guard", "mediaId": media["id"]}).json()

    opts = {"sourceLanguage": "en", "targetLanguage": "vi", "recognType": 5, "modelName": "large-v3"}
    r1 = e2e_client.post("/api/jobs", json={"mediaId": media["id"], "projectId": proj["id"], "jobType": "asr", "options": opts})
    assert r1.status_code == 202
    job_id = r1.json()["id"]

    r2 = e2e_client.post("/api/jobs", json={"mediaId": media["id"], "projectId": proj["id"], "jobType": "asr", "options": opts})
    assert r2.status_code == 409

    cancel_res = e2e_client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel_res.status_code == 200
    done = _poll_job_until_done(e2e_client, job_id)
    assert done["status"] == "cancelled"


# ============================================================================
# F5: Stage 3 Dubbing UX, Voice Management & Stage 4 / CapCut (Tier 1 & Tier 2)
# ============================================================================


def test_f5_custom_voice_crud_lifecycle(e2e_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace
    from videotrans.services import model_cache, voice_preview

    def _synth(voice_dict, out_path: Path, sample_text: str):
        make_wav(out_path, duration_s=1.0, sr=24000, freq=520.0)

    voice_preview.set_preview_synthesizer(_synth)
    monkeypatch.setattr(
        model_cache,
        "get_cached_vieneu_engine",
        lambda **kw: SimpleNamespace(encode_reference=lambda *a, **k: ([0.1, 0.2], [1, 2])),
    )
    ref_wav = make_wav(tmp_path / "ref_voice.wav", duration_s=2.5, sr=24000, freq=330.0)

    with ref_wav.open("rb") as fh:
        create_res = e2e_client.post(
            "/api/custom-voices",
            data={"name": "Studio Narrator", "provider": "2", "language": "vi", "description": "Warm narrator"},
            files={"audio": ("ref_voice.wav", fh, "audio/wav")},
        )
    assert create_res.status_code == 201, create_res.text
    voice = create_res.json()
    vid = voice["id"]
    assert voice["name"] == "Studio Narrator"

    listed = e2e_client.get("/api/custom-voices").json()["voices"]
    assert any(v["id"] == vid for v in listed)

    fetched = e2e_client.get(f"/api/custom-voices/{vid}")
    assert fetched.status_code == 200
    assert fetched.json()["description"] == "Warm narrator"

    upd = e2e_client.put(f"/api/custom-voices/{vid}", json={"name": "Studio Narrator V2", "description": "Updated"})
    assert upd.status_code == 200
    assert upd.json()["name"] == "Studio Narrator V2"

    ref_dl = e2e_client.get(f"/api/custom-voices/{vid}/audio")
    assert ref_dl.status_code == 200
    assert len(ref_dl.content) > 44

    prev_dl = e2e_client.get(f"/api/custom-voices/{vid}/preview/audio")
    assert prev_dl.status_code == 200
    assert len(prev_dl.content) > 44

    del_res = e2e_client.delete(f"/api/custom-voices/{vid}?hard=true")
    assert del_res.status_code == 200
    assert e2e_client.get(f"/api/custom-voices/{vid}").status_code == 404


def test_f5_tts_preview_with_synthesizer_and_auto_speed_fit(e2e_client):
    from videotrans.services import voice_preview

    def _synth(voice_dict, out_path: Path, sample_text: str):
        make_wav(out_path, duration_s=2.0, sr=24000, freq=440.0)

    voice_preview.set_preview_synthesizer(_synth)
    res = e2e_client.post(
        "/api/tts/preview",
        json={
            "provider": 2,
            "voice": "Mai Anh",
            "text": "Testing auto speed fitting",
            "language": "vi",
            "autoSpeed": True,
            "slotDuration": 1.5,
            "maxSpeedRate": 1.5,
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["ok"] is True
    assert body["applied_speed"] > 1.0

    audio_res = e2e_client.get(body["audio_url"])
    assert audio_res.status_code == 200
    info = sf.info(io.BytesIO(audio_res.content))
    assert info.duration == pytest.approx(1.5, abs=0.2)


def test_f5_dubbing_assemble_mixed_sample_rates_and_channels_to_48khz_mono(e2e_client):
    from videotrans.core import voice_store
    from videotrans.core.project_store import get_project_dir

    voice_store.init_voice_dirs()
    pid = e2e_client.post("/api/projects", json={"name": "Mixed Rate Assembly", "duration": 3.0, "stage": 3}).json()["id"]

    # Segment 1: 24 kHz mono (e.g. VieNeu output)
    make_wav(voice_store.get_preview_audio_path("prev_111111111111.wav"), duration_s=0.5, sr=24000, channels=1, freq=440.0)
    # Segment 2: 44.1 kHz stereo (e.g. ElevenLabs/Edge output)
    make_wav(voice_store.get_preview_audio_path("prev_222222222222.wav"), duration_s=0.5, sr=44100, channels=2, freq=880.0)

    res = e2e_client.post(
        f"/api/projects/{pid}/dubbing/assemble",
        json={
            "segments": [
                {"id": 1, "startSec": 0.25, "endSec": 0.75, "targetText": "Clip one", "previewAudioId": "prev_111111111111"},
                {"id": 2, "startSec": 1.50, "endSec": 2.00, "targetText": "Clip two", "previewAudioId": "prev_222222222222"},
            ]
        },
    )
    assert res.status_code == 200, res.text
    audio_url = res.json()["audio_url"]
    audio_resp = e2e_client.get(audio_url)
    assert audio_resp.status_code == 200

    out_path = get_project_dir(pid) / "dubbing" / "voiceover_merged.wav"
    assert out_path.is_file()
    info = sf.info(str(out_path))
    assert info.samplerate == 48000
    assert info.channels == 1
    assert info.duration == pytest.approx(3.0, abs=0.05)

    samples, sr = sf.read(str(out_path), dtype="float32")
    assert np.max(np.abs(samples[int(0.02 * sr):int(0.18 * sr)])) < 1e-4
    assert np.max(np.abs(samples[int(0.35 * sr):int(0.65 * sr)])) > 0.05
    assert np.max(np.abs(samples[int(0.95 * sr):int(1.35 * sr)])) < 1e-4
    assert np.max(np.abs(samples[int(1.60 * sr):int(1.90 * sr)])) > 0.05


def test_f5_capcut_export_subtitles_target_and_bundle_zip(e2e_client):
    pid = e2e_client.post("/api/projects", json={"name": "CapCut Project", "duration": 2.0, "stage": 4}).json()["id"]
    segments = [
        {
            "id": 1,
            "startSec": 0.0,
            "endSec": 1.5,
            "sourceText": "Original English Line",
            "rawTranslation": "Ban dich tho ban dau",
            "targetText": "Ban dich da chinh sua",
        }
    ]
    e2e_client.put(f"/api/projects/{pid}/state", json={"stage": 4, "state": {"segments": segments}})

    edited = e2e_client.get(f"/api/projects/{pid}/export-capcut/subtitles_edited.srt")
    assert edited.status_code == 200
    assert "Ban dich da chinh sua" in edited.text

    target = e2e_client.get(f"/api/projects/{pid}/export-capcut/subtitles_target.srt")
    assert target.status_code == 200
    assert "Ban dich tho ban dau" in target.text
    assert "Original English Line" not in target.text

    bundle = e2e_client.get(f"/api/projects/{pid}/export-capcut/bundle.zip")
    assert bundle.status_code == 200
    with zipfile.ZipFile(io.BytesIO(bundle.content), "r") as zf:
        names = zf.namelist()
        assert any(n.endswith("subtitles_edited.srt") for n in names)
        assert any(n.endswith("subtitles_target.srt") for n in names)
        assert any(n.endswith("voiceover_merged.wav") for n in names)


def test_f5_stage4_assets_upload_and_render_job_execution(e2e_client, tmp_path: Path):
    from videotrans.util._ffprobe import get_video_info, runffprobe

    video = make_mp4(tmp_path / "stage4_input.mp4", duration_s=1.5, width=320, height=240, with_audio=True)
    bgm = make_wav(tmp_path / "bgm.wav", duration_s=1.5, sr=44100, freq=220.0)
    thumb = make_png(tmp_path / "thumb.png", width=320, height=240, text="Cover")

    media = e2e_client.post("/api/media/register-local", json={"paths": [str(video)]}).json()["media"][0]
    proj = e2e_client.post("/api/projects", json={"name": "Stage 4 Render", "mediaId": media["id"]}).json()

    with bgm.open("rb") as fh:
        bgm_res = e2e_client.post("/api/assets/background-audio", files={"file": ("bgm.wav", fh, "audio/wav")})
    assert bgm_res.status_code == 201, bgm_res.text
    bgm_id = bgm_res.json()["id"]

    with thumb.open("rb") as fh:
        thumb_res = e2e_client.post("/api/assets/thumbnail", files={"file": ("thumb.png", fh, "image/png")})
    assert thumb_res.status_code == 201, thumb_res.text
    thumb_id = thumb_res.json()["id"]

    render_res = e2e_client.post(
        "/api/render",
        json={
            "mediaId": media["id"],
            "projectId": proj["id"],
            "options": {
                "sourceLanguage": "en",
                "targetLanguage": "vi",
                "subtitles": "1\n00:00:00,000 --> 00:00:01,200\nXin chao\n",
                "backgroundAudioId": bgm_id,
                "backgroundAudioVolume": 0.4,
                "originalAudioVolume": 0.5,
                "thumbnailId": thumb_id,
                "subtitleStyle": {"fontSize": 20, "primaryColor": "#FFFFFF"},
            },
        },
    )
    assert render_res.status_code == 202, render_res.text
    done = _poll_job_until_done(e2e_client, render_res.json()["id"], timeout_s=30.0)
    assert done["status"] == "succeeded", done
    mp4_outputs = [o for o in done["outputs"] if str(o.get("name", "")).endswith(".mp4")]
    assert mp4_outputs, done
    dl = e2e_client.get(mp4_outputs[0]["url"])
    assert dl.status_code == 200, dl.text
    rendered_path = tmp_path / mp4_outputs[0]["name"]
    rendered_path.write_bytes(dl.content)
    info = get_video_info(rendered_path.as_posix())
    assert info.get("video_streams", 0) >= 1
    probe_json = json.loads(
        runffprobe(["-v", "quiet", "-print_format", "json", "-show_streams", rendered_path.as_posix()])
    )
    attached_pic_streams = [
        s for s in probe_json.get("streams", [])
        if s.get("codec_type") == "video" and (s.get("disposition") or {}).get("attached_pic") == 1
    ]
    assert attached_pic_streams, f"Expected attached_pic thumbnail stream in {rendered_path}, got streams={probe_json.get('streams')}"


def test_f5_dubbing_assemble_rejects_missing_or_invalid_preview_id(e2e_client):
    from videotrans.core.project_store import get_project_dir

    pid = e2e_client.post("/api/projects", json={"name": "Invalid Assembly", "duration": 2.0}).json()["id"]
    assert e2e_client.post(f"/api/projects/{pid}/dubbing/assemble", json={"segments": []}).status_code == 400
    assert e2e_client.post(
        f"/api/projects/{pid}/dubbing/assemble",
        json={"segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "targetText": "Hi", "previewAudioId": "../evil"}]},
    ).status_code == 400
    assert e2e_client.post(
        f"/api/projects/{pid}/dubbing/assemble",
        json={"segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "targetText": "Hi", "previewAudioId": "prev_deadbeef1234"[:17]}]},
    ).status_code == 400
    assert not (get_project_dir(pid) / "dubbing" / "voiceover_merged.wav").exists()


def test_f5_dubbing_assemble_clamps_overlapping_segments(e2e_client):
    from videotrans.core import voice_store
    from videotrans.core.project_store import get_project_dir

    voice_store.init_voice_dirs()
    pid = e2e_client.post("/api/projects", json={"name": "Overlap Clamp", "duration": 3.0}).json()["id"]
    make_wav(voice_store.get_preview_audio_path("prev_aaaa1111aaaa.wav"), duration_s=1.5, sr=24000, freq=440.0)
    make_wav(voice_store.get_preview_audio_path("prev_bbbb2222bbbb.wav"), duration_s=0.6, sr=24000, freq=660.0)

    res = e2e_client.post(
        f"/api/projects/{pid}/dubbing/assemble",
        json={
            "segments": [
                {"id": 1, "startSec": 0.2, "endSec": 1.7, "targetText": "Long", "previewAudioId": "prev_aaaa1111aaaa"},
                {"id": 2, "startSec": 1.0, "endSec": 1.6, "targetText": "Next", "previewAudioId": "prev_bbbb2222bbbb"},
            ]
        },
    )
    assert res.status_code == 200, res.text
    out_wav = get_project_dir(pid) / "dubbing" / "voiceover_merged.wav"
    assert sf.info(str(out_wav)).samplerate == 48000


def test_f5_capcut_export_rejects_video_and_unknown_assets(e2e_client):
    pid = e2e_client.post("/api/projects", json={"name": "CapCut Error"}).json()["id"]
    assert e2e_client.get(f"/api/projects/{pid}/export-capcut/video.mp4").status_code == 404
    assert e2e_client.get(f"/api/projects/{pid}/export-capcut/unknown.txt").status_code == 404
    assert e2e_client.get("/api/projects/no-such-project/export-capcut/bundle.zip").status_code == 404


def test_f5_edit_asset_rejects_invalid_kind_and_extension(e2e_client):
    assert e2e_client.post("/api/assets/unknown-kind", files={"file": ("a.wav", io.BytesIO(b"123"), "audio/wav")}).status_code == 404
    assert e2e_client.post("/api/assets/thumbnail", files={"file": ("bad.wav", io.BytesIO(b"123"), "audio/wav")}).status_code == 400
    assert e2e_client.post("/api/assets/background-audio", files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}).status_code == 400


def test_f5_task_params_deepgram_validation_parity(tmp_path: Path):
    """M2 Contract: build_task_params validates Deepgram model against asr_models() instead of bypassing."""
    from videotrans import recognition
    from videotrans.api.task_params import build_task_params

    sample = tmp_path / "sample.mp4"
    sample.write_bytes(b"video")

    with pytest.raises(ValueError, match="not supported by Deepgram"):
        build_task_params(
            sample,
            {
                "sourceLanguage": "zh-cn",
                "targetLanguage": "vi",
                "recognType": recognition.Deepgram,
                "modelName": "large-v3",
                "translateType": 0,
                "ttsType": 0,
            },
        )

    ok_params = build_task_params(
        sample,
        {
            "sourceLanguage": "zh-cn",
            "targetLanguage": "vi",
            "recognType": recognition.Deepgram,
            "modelName": "nova-3",
            "translateType": 0,
            "ttsType": 0,
        },
        job_type="asr",
    )
    assert ok_params["model_name"] == "nova-3"
