# -*- coding: utf-8 -*-
"""Tier 5 adversarial regression tests for M7 Phase 2 findings (Gaps 1-7 and 12-15)."""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from videotrans.api.app import create_app
from videotrans.api.ocr_helpers import extract_ocr_segment_text
from videotrans.api.task_params import _parse_ocr_roi, build_task_params
from videotrans.configure.excepts import FFmpegError
from videotrans.core.db import get_db_path, init_db, set_db_path
from videotrans.core.job_manager import JobManager
from videotrans.core.proc_registry import kill_process_tree
from videotrans.core.project_store import create_project, get_project, get_project_dir, update_project_state
from videotrans.ocr._scanner import OcrScanner
from videotrans.ocr._types import OcrConfig, OcrResult
from videotrans.task import orchestrator
from videotrans.task._rate import SpeedRate
from videotrans.task.orchestrator import CancellationToken, EventKind, TaskRequest, TaskStatus
from videotrans.tts._base import BaseTTS


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    orig_db = get_db_path()
    db_path = tmp_path / "tier5.db"
    set_db_path(db_path)
    init_db()
    monkeypatch.setattr("videotrans.core.project_store.get_projects_root", lambda: tmp_path / "projects")
    yield tmp_path
    set_db_path(orig_db)


# ---------------------------------------------------------------------------
# Gap 1: orchestrator.run() cancel check after stage loop on stop_after_stage
# ---------------------------------------------------------------------------


def test_orchestrator_cancel_during_stop_after_stage_persist_returns_cancelled(isolated_db, monkeypatch):
    tmp_path = isolated_db
    src_video = tmp_path / "sample.mp4"
    src_video.write_bytes(b"\x00" * 128)
    out_dir = tmp_path / "out"
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))

    events = []
    token = CancellationToken()
    joined = {"called": False}

    class _FakeTransCreate:
        def __init__(self, cfg, event_sink=None, cancellation_token=None):
            self.cfg = cfg
            self.event_sink = event_sink
            self.cancellation_token = cancellation_token
            self.should_recogn = True
            self.should_trans = False
            self.should_dubbing = False
            self.should_recogn2 = False
            self.should_hebing = False
            self.source_srt_list = [{"line": 1, "time": "00:00:00,000 --> 00:00:01,000", "text": "hello"}]
            self.target_srt_list = []
            self._novoice_thread = None

        def prepare(self):
            return None

        def recogn(self):
            return None

        def diariz(self):
            return None

    monkeypatch.setattr(orchestrator, "TransCreate", _FakeTransCreate)
    monkeypatch.setattr(orchestrator, "_join_novoice_thread", lambda task: joined.__setitem__("called", True))

    def _cancel_on_diariz(task, stage_name):
        if stage_name == "diariz":
            token.cancel()

    monkeypatch.setattr(orchestrator, "_persist_stage_artifacts", _cancel_on_diariz)

    req = TaskRequest({
        "name": str(src_video),
        "target_dir": str(out_dir),
        "project_id": "proj_cancel_stop_after",
        "source_language_code": "en",
        "target_language_code": "vi",
    })
    res = orchestrator.run(
        req,
        event_sink=events.append,
        cancellation_token=token,
        stage_limit="asr",
        stop_after_stage="diariz",
    )

    assert res.status == TaskStatus.CANCELLED
    assert joined["called"] is True
    assert EventKind.CANCELLED in [e.kind for e in events]
    assert EventKind.SUCCEEDED not in [e.kind for e in events]
    assert not (get_project_dir("proj_cancel_stop_after") / "transcripts" / "asr_baseline.json").exists()


# ---------------------------------------------------------------------------
# Gap 2: Pin _last_sig / last_sig to last frame that actually ran OCR
# ---------------------------------------------------------------------------


def _render_text_crop(intensity: int) -> np.ndarray:
    img = np.zeros((48, 200, 3), dtype=np.uint8)
    if intensity > 0:
        img[12:36, 20:180] = intensity
    return img


def test_scanner_and_stage2_detect_gradual_fade_out_without_creeping_drift(tmp_path, monkeypatch):
    # 0..1000ms bright (220), 1500..20500ms linear fade (-5/step -> 0 at 23500ms? wait: 220 - 5*k reaches 0 at k=44)
    # 20 steps of -5 drops intensity from 220 -> 120, well past _sig_is_static threshold vs pinned 220!
    frames = []
    ts = 0
    for _ in range(3):  # 0, 500, 1000
        frames.append((ts, _render_text_crop(220)))
        ts += 500
    intensity = 220
    while intensity > 0:
        intensity = max(0, intensity - 5)
        frames.append((ts, _render_text_crop(intensity)))
        ts += 500
    for _ in range(10):  # 5000ms pure black
        frames.append((ts, _render_text_crop(0)))
        ts += 500

    class _FadeProvider:
        def recognize(self, crop, lang=None):
            bright = float(np.mean(crop))
            if bright >= 15.0:
                return OcrResult(text="Hello World", confidence=0.95)
            return OcrResult(text="", confidence=0.95)

    cfg = OcrConfig(
        roi=(0.0, 0.0, 1.0, 1.0),
        coarse_interval_ms=500,
        min_stable_samples=1,
        grace_gap_ms=500,
    )
    scanner = OcrScanner(_FadeProvider(), cfg, frames_are_cropped=True)
    total_dur = frames[-1][0] + 500
    segs = scanner.scan(frames, duration_ms=total_dur)

    assert len(segs) == 1
    assert segs[0].text == "Hello World"
    # Before fix, creeping drift kept ocr_calls=3 and segment lasted only to 20000ms on different schedule or never re-fired until heartbeat;
    # with pinned _last_sig, OCR re-fires well before black end and detects empty frames.
    assert scanner._stats["ocr_calls"] > 3
    assert segs[0].end_ms < total_dur - 4000

    # Also verify Stage 2 extract_ocr_segment_text does NOT overwrite last_sig on skipped frames
    fake_media = tmp_path / "fade.mp4"
    fake_media.write_bytes(b"mp4")

    monkeypatch.setattr(
        "videotrans.ocr._frame_source.sampled_roi_iter",
        lambda *args, **kwargs: iter(frames[:30]),
    )
    ocr_call_count = {"n": 0}

    def _stage2_ocr(crop):
        ocr_call_count["n"] += 1
        bright = float(np.mean(crop))
        return [{"text": "Hello World" if bright >= 15.0 else "", "confidence": 0.95, "bbox": [0, 0, 100, 20]}]

    res = extract_ocr_segment_text(fake_media, 0.0, 15.0, [0.0, 0.0, 1.0, 1.0], ocr_runner=_stage2_ocr)
    assert res["text"] == "Hello World"
    assert ocr_call_count["n"] >= 3


# ---------------------------------------------------------------------------
# Gap 3: Delta state persistence preserves concurrent edits
# ---------------------------------------------------------------------------


def test_persist_stage_artifacts_and_translation_preserve_concurrent_state_updates(isolated_db, monkeypatch):
    tmp_path = isolated_db
    proj = create_project(
        name="Concurrent State",
        state={"ttsVoice": "old_voice", "languages": {"source": {"code": "zh-cn"}, "target": {"code": "vi"}, "timingMode": "voice"}},
    )
    pid = proj["id"]

    orig_update = update_project_state
    injected = {"done": False}

    def _interleaved_update(project_id, state_dict, stage=None, status=None, **kwargs):
        if not injected["done"] and state_dict.get("segments"):
            injected["done"] = True
            orig_update(pid, {"ttsVoice": "new_user_voice", "userEdited": True, "languages": {"timingMode": "video"}}, merge=True)
        return orig_update(project_id, state_dict, stage=stage, status=status, **kwargs)

    monkeypatch.setattr("videotrans.core.project_store.update_project_state", _interleaved_update)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    src_srt = tmp_path / "source.srt"
    src_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

    fake_task = SimpleNamespace(
        cfg=SimpleNamespace(
            project_id=pid,
            cache_folder=str(cache_dir),
            source_sub=str(src_srt),
            target_sub=str(tmp_path / "target.srt"),
            source_code="en",
            target_code="vi",
            subtitle_source="audio_asr",
        ),
        source_srt_list=[{"line": 1, "time": "00:00:00,000 --> 00:00:01,000", "text": "Hello"}],
        target_srt_list=[],
        detect_language="en",
        cancellation_token=CancellationToken(),
    )

    orchestrator._persist_stage_artifacts(fake_task, "diariz")
    after_diariz = get_project(pid)
    assert after_diariz is not None
    assert after_diariz["state"]["ttsVoice"] == "new_user_voice"
    assert after_diariz["state"]["userEdited"] is True
    assert after_diariz["state"]["languages"]["timingMode"] == "video"
    assert after_diariz["state"]["languages"]["source"]["code"] == "en"
    assert len(after_diariz["state"]["segments"]) == 1

    # Also test run_staged_translation delta persistence
    injected["done"] = False
    monkeypatch.setattr(
        "videotrans.translator.run",
        lambda **kwargs: [{"line": 1, "time": "00:00:00,000 --> 00:00:01,000", "text": "Xin chao"}],
    )
    src_video = tmp_path / "trans.mp4"
    src_video.write_bytes(b"\x00" * 64)
    out_dir = tmp_path / "trans_out"
    temp_root = tmp_path / "trans_temp"
    temp_root.mkdir()
    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))

    def _interleaved_trans_update(project_id, state_dict, stage=None, status=None, **kwargs):
        if not injected["done"] and state_dict.get("segments"):
            injected["done"] = True
            orig_update(pid, {"ttsVoice": "trans_concurrent_voice"}, merge=True)
        return orig_update(project_id, state_dict, stage=stage, status=status, **kwargs)

    monkeypatch.setattr("videotrans.core.project_store.update_project_state", _interleaved_trans_update)

    res = orchestrator.run_staged_translation(TaskRequest({
        "name": str(src_video),
        "target_dir": str(out_dir),
        "project_id": pid,
        "source_language_code": "en",
        "target_language_code": "vi",
        "segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Hello"}],
    }))
    assert res.status == TaskStatus.SUCCEEDED
    after_trans = get_project(pid)
    assert after_trans["state"]["ttsVoice"] == "trans_concurrent_voice"
    assert after_trans["state"]["segments"][0]["targetText"] == "Xin chao"


# ---------------------------------------------------------------------------
# Gap 4: kill_process_tree handles PermissionError & psutil.AccessDenied
# ---------------------------------------------------------------------------


def test_kill_process_tree_handles_permission_error_and_access_denied(monkeypatch):
    import psutil
    from videotrans.util import _ffmpeg_runner

    class _FakeProc:
        def __init__(self):
            self.pid = 999999
            self.wait_called = False

        def poll(self):
            return None

        def kill(self):
            raise PermissionError(5, "Access is denied")

        def wait(self, timeout=None):
            self.wait_called = True
            return 0

    class _FakePsutilProc:
        def __init__(self, pid):
            pass

        def children(self, recursive=True):
            raise psutil.AccessDenied(999999)

    monkeypatch.setattr(psutil, "Process", _FakePsutilProc)
    proc = _FakeProc()
    kill_process_tree(proc)
    assert proc.wait_called is True

    # Also verify runffmpeg surfaces FFmpegError on cancellation when proc.kill raises PermissionError
    class _RunnerProc:
        def __init__(self, *args, **kwargs):
            self.pid = 888888
            self.returncode = 0

        def poll(self):
            return None

        def kill(self):
            raise PermissionError(5, "Access is denied")

        def wait(self, timeout=None):
            return 0

        def communicate(self, timeout=None):
            return "", ""

    token = CancellationToken()
    monkeypatch.setattr(_ffmpeg_runner.subprocess, "Popen", lambda *a, **kw: _RunnerProc())
    checks = {"n": 0}

    def _cancel_in_loop():
        checks["n"] += 1
        return checks["n"] >= 2

    monkeypatch.setattr(token, "is_cancelled", _cancel_in_loop)
    with pytest.raises(FFmpegError, match="cancelled"):
        _ffmpeg_runner.runffmpeg(["-version"], cancellation_token=token)


# ---------------------------------------------------------------------------
# Gap 5: SpeedRate._prepare_data initializes start_time_source/end_time_source
# ---------------------------------------------------------------------------


def test_speed_rate_handles_zero_or_inverted_duration_first_item(tmp_path):
    cache_dir = tmp_path / "rate_cache"
    cache_dir.mkdir()
    sr = SpeedRate(
        queue_tts=[
            {"line": 1, "start_time": 1000, "end_time": 1000, "filename": ""},
            {"line": 2, "start_time": 2000, "end_time": 3000, "filename": ""},
        ],
        cache_folder=str(cache_dir),
        should_audiorate=True,
        should_videorate=False,
        target_audio=str(tmp_path / "out.wav"),
    )
    sr._prepare_data()
    sr._calculate_adjustments()
    assert sr.queue_tts[0]["start_time_source"] == 1000
    assert sr.queue_tts[0]["end_time_source"] == 1000
    assert sr.queue_tts[1]["start_time_source"] == 2000


# ---------------------------------------------------------------------------
# Gap 6: BaseTTS._cleantts safely coerces None prosody fields
# ---------------------------------------------------------------------------


def test_base_tts_cleantts_coerces_none_prosody_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("videotrans.configure.config.TEMP_DIR", str(tmp_path))
    inst = BaseTTS(
        queue_tts=[{
            "text": "hello",
            "filename": str(tmp_path / "x.wav"),
            "volume": None,
            "rate": None,
            "pitch": None,
        }],
        uuid="tier5_none_prosody",
    )
    assert inst.volume == "+0%"
    assert inst.rate == "+0%"
    assert inst.pitch == "+0Hz"


# ---------------------------------------------------------------------------
# Gap 7: Cancel race never flips projects.status to "completed" after cancel()
# ---------------------------------------------------------------------------


def test_cancel_race_never_flips_project_status_to_completed(isolated_db, monkeypatch):
    tmp_path = isolated_db
    proj = create_project(name="Cancel Race", state={"segments": []})
    pid = proj["id"]

    src_video = tmp_path / "race.mp4"
    src_video.write_bytes(b"\x00" * 64)
    out_dir = tmp_path / "race_out"
    temp_root = tmp_path / "race_temp"
    temp_root.mkdir()
    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))
    monkeypatch.setattr(
        "videotrans.translator.run",
        lambda **kwargs: [{"line": 1, "time": "00:00:00,000 --> 00:00:01,000", "text": "Xin chao"}],
    )

    entered_persist = threading.Event()
    release_persist = threading.Event()
    observed_statuses: list[str] = []

    orig_update_state = update_project_state

    def _synchronized_update_state(project_id, state_dict, stage=None, status=None, **kwargs):
        entered_persist.set()
        release_persist.wait(timeout=5.0)
        res = orig_update_state(project_id, state_dict, stage=stage, status=status, **kwargs)
        p_now = get_project(pid)
        if p_now:
            observed_statuses.append(p_now["status"])
        return res

    monkeypatch.setattr("videotrans.core.project_store.update_project_state", _synchronized_update_state)

    jm = JobManager(runner=orchestrator.run, translation_runner=orchestrator.run_staged_translation)
    job = jm.submit(
        {
            "name": str(src_video),
            "target_dir": str(out_dir),
            "project_id": pid,
            "source_language_code": "en",
            "target_language_code": "vi",
            "segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Hello"}],
        },
        media_id="m_race",
        job_type="translation",
        project_id=pid,
    )

    assert entered_persist.wait(timeout=5.0)
    jm.cancel(job.id)
    assert get_project(pid)["status"] == "paused"
    release_persist.set()

    # Wait for job thread to finish
    import time
    deadline = time.monotonic() + 5.0
    while jm.get(job.id).status not in {"cancelled", "succeeded", "failed"} and time.monotonic() < deadline:
        time.sleep(0.02)

    assert "completed" not in observed_statuses
    assert get_project(pid)["status"] == "paused"


# ---------------------------------------------------------------------------
# Gap 12: POST /api/voices/preview-clone fallback returns 200 valid WAV
# ---------------------------------------------------------------------------


def test_preview_clone_fallback_returns_200_valid_wav(tmp_path, monkeypatch):
    previews_dir = tmp_path / "previews"
    previews_dir.mkdir()
    monkeypatch.setattr("videotrans.core.voice_store.PREVIEWS_DIR", previews_dir)
    monkeypatch.setattr(
        "videotrans.core.voice_store.get_preview_audio_path",
        lambda filename: previews_dir / filename,
    )

    async def _boom(*args, **kwargs):
        raise RuntimeError("clone synthesis crashed")

    monkeypatch.setattr("videotrans.api.routes.voices.synthesize_clone_preview", _boom)
    monkeypatch.setattr(
        "videotrans.api.routes.voices.normalize_reference_audio",
        lambda src, dst, **kwargs: Path(dst).write_bytes(b"RIFF" + b"\x00" * 40),
    )

    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/api/voices/preview-clone",
        data={"provider": "vieneu", "text": "Hello", "language": "en"},
        files={"audio": ("ref.wav", b"RIFF" + b"\x00" * 64, "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["preview_url"].startswith("/api/voices/preview-audio/")

    wav_resp = client.get(body["preview_url"])
    assert wav_resp.status_code == 200
    assert wav_resp.content.startswith(b"RIFF")


# ---------------------------------------------------------------------------
# Gap 13: ROI None-valued width/height fallback across task_params, ocr_helpers, stages
# ---------------------------------------------------------------------------


def test_roi_none_valued_width_height_fallback_across_api_and_helpers(tmp_path, monkeypatch):
    roi_dict = {"x": 0.1, "y": 0.7, "width": None, "w": 0.8, "height": None, "h": 0.2}
    assert _parse_ocr_roi(roi_dict) == pytest.approx((0.1, 0.7, 0.8, 0.2))

    fake_media = tmp_path / "roi.mp4"
    fake_media.write_bytes(b"mp4")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = extract_ocr_segment_text(
        fake_media,
        0.0,
        1.0,
        roi_dict,
        frame_fetcher=lambda *args: (frame, 100, 100),
        ocr_runner=lambda crop: [{"text": "OK", "confidence": 0.95, "bbox": [0, 0, 10, 10]}],
    )
    assert res["text"] == "OK"

    app = create_app()
    monkeypatch.setattr(app.state.media_store, "_probe", lambda p: {"time": 1000})
    app.state.media_store.inspect(fake_media, filename="roi.mp4", media_id="m_roi_none")
    app.state.ocr_extractor = lambda *args, **kwargs: {"ok": True, "text": "API OK", "roi": list(args[3])}
    client = TestClient(app)
    resp = client.post("/api/ocr/extract", json={
        "mediaId": "m_roi_none",
        "startSec": 0.0,
        "endSec": 1.0,
        "roi": roi_dict,
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["roi"] == pytest.approx([0.1, 0.7, 0.8, 0.2])


# ---------------------------------------------------------------------------
# Gap 14: Empty ocrOptions: {} does not shadow non-empty ocr_options
# ---------------------------------------------------------------------------


def test_build_task_params_empty_ocr_options_does_not_shadow_snake_case(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"\x00" * 64)
    params = build_task_params(
        media,
        {
            "subtitleSource": "video_ocr",
            "ocrRoi": [0.05, 0.75, 0.9, 0.2],
            "ocrRoiConfirmed": True,
            "ocrOptions": {},
            "ocr_options": {
                "coarse_interval_ms": 250,
                "paragraph_mode": "unwrap",
                "use_angle_cls": True,
            },
            "sourceLanguage": "en",
            "targetLanguage": "vi",
        },
        job_type="asr",
        output_dir=tmp_path / "out",
        temp_dir=tmp_path / "tmp",
    )
    assert params["ocr_options"] == {
        "coarse_interval_ms": 250,
        "paragraph_mode": "unwrap",
        "use_angle_cls": True,
    }


# ---------------------------------------------------------------------------
# Gap 15: PUT /api/projects/{id}/state guards backend=None and avoids 404 orphan writes
# ---------------------------------------------------------------------------


def test_project_state_backend_none_and_missing_project_no_orphan_files(isolated_db):
    tmp_path = isolated_db
    proj = create_project(name="Backend None Guard")
    app = create_app()
    client = TestClient(app)

    resp_ok = client.put(
        f"/api/projects/{proj['id']}/state",
        json={"state": {"backend": None, "currentStep": 1}},
    )
    assert resp_ok.status_code == 200, resp_ok.text

    missing_id = "missing_proj_404"
    resp_404 = client.put(
        f"/api/projects/{missing_id}/state",
        json={"state": {"segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Orphan"}]}},
    )
    assert resp_404.status_code == 404
    assert not (tmp_path / "projects" / missing_id).exists()


# ---------------------------------------------------------------------------
# Iteration 2 Finding 1: run_staged_translation same-language bypass persists state
# ---------------------------------------------------------------------------


def test_run_staged_translation_same_language_persists_stage3_completed_and_respects_cancel(isolated_db, monkeypatch):
    import json

    tmp_path = isolated_db
    proj = create_project(
        name="Same Lang Proj",
        stage=2,
        status="processing",
        state={"segments": [{"id": 1, "startSec": 0.0, "endSec": 1.5, "sourceText": "Hello", "targetText": ""}]},
    )
    pid = proj["id"]

    src_video = tmp_path / "same.mp4"
    src_video.write_bytes(b"\x00" * 64)
    out_dir = tmp_path / "same_out"
    temp_root = tmp_path / "same_temp"
    temp_root.mkdir()
    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))

    events = []
    res = orchestrator.run_staged_translation(
        TaskRequest({
            "name": str(src_video),
            "target_dir": str(out_dir),
            "project_id": pid,
            "source_language_code": "en",
            "target_language_code": "en",
            "segments": [{"id": 1, "startSec": 0.0, "endSec": 1.5, "sourceText": "Hello", "targetText": ""}],
        }),
        event_sink=events.append,
    )
    assert res.status == TaskStatus.SUCCEEDED
    assert EventKind.SUCCEEDED in [e.kind for e in events]

    saved = get_project(pid)
    assert saved is not None
    assert saved["stage"] == 3
    assert saved["status"] == "completed"
    assert saved["state"]["segments"][0]["targetText"] == "Hello"

    seg_file = get_project_dir(pid) / "transcripts" / "segments.json"
    assert seg_file.is_file()
    disk_segs = json.loads(seg_file.read_text(encoding="utf-8"))
    assert disk_segs[0]["targetText"] == "Hello"

    # Cancellation during same-language persistence must return CANCELLED without SUCCEEDED
    proj2 = create_project(
        name="Same Lang Cancel",
        stage=2,
        status="processing",
        state={"segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Stay", "targetText": "ORIG"}]},
    )
    pid2 = proj2["id"]
    token = CancellationToken()
    orig_update = update_project_state

    def _cancel_mid_update(project_id, state_dict, stage=None, status=None, **kwargs):
        token.cancel()
        return orig_update(project_id, state_dict, stage=stage, status=status, **kwargs)

    monkeypatch.setattr("videotrans.core.project_store.update_project_state", _cancel_mid_update)
    cancel_events = []
    res_cancel = orchestrator.run_staged_translation(
        TaskRequest({
            "name": str(src_video),
            "target_dir": str(out_dir),
            "project_id": pid2,
            "source_language_code": "en",
            "target_language_code": "en",
            "segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Stay", "targetText": ""}],
        }),
        event_sink=cancel_events.append,
        cancellation_token=token,
    )
    assert res_cancel.status == TaskStatus.CANCELLED
    assert EventKind.SUCCEEDED not in [e.kind for e in cancel_events]
    assert get_project(pid2)["state"]["segments"][0]["targetText"] == "ORIG"


# ---------------------------------------------------------------------------
# Iteration 2 Finding 2: No eager disk writes outside update_project_state
# ---------------------------------------------------------------------------


def test_orchestrator_no_eager_disk_writes_on_missing_or_cancelled_project(isolated_db, monkeypatch):
    import json

    tmp_path = isolated_db
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    src_srt = tmp_path / "source.srt"
    src_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    tgt_srt = tmp_path / "target.srt"
    tgt_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n", encoding="utf-8")
    (cache_dir / "speaker.json").write_text("[]", encoding="utf-8")

    missing_pid = "missing_proj_no_orphan"
    fake_task = SimpleNamespace(
        cfg=SimpleNamespace(
            project_id=missing_pid,
            cache_folder=str(cache_dir),
            source_sub=str(src_srt),
            target_sub=str(tgt_srt),
            source_code="en",
            target_code="es",
            subtitle_source="audio_asr",
            targetdir_mp4=str(tmp_path / "out.mp4"),
        ),
        source_srt_list=[{"line": 1, "time": "00:00:00,000 --> 00:00:01,000", "text": "Hello"}],
        target_srt_list=[{"line": 1, "time": "00:00:00,000 --> 00:00:01,000", "text": "Hola"}],
        detect_language="en",
        cancellation_token=CancellationToken(),
    )
    for stg in ("prepare", "recogn", "diariz", "trans", "dubbing", "assembling"):
        orchestrator._persist_stage_artifacts(fake_task, stg)
    assert not get_project_dir(missing_pid).exists()

    # run_staged_translation on missing project must not create orphan dir
    src_video = tmp_path / "orphan.mp4"
    src_video.write_bytes(b"\x00" * 64)
    out_dir = tmp_path / "orphan_out"
    temp_root = tmp_path / "orphan_temp"
    temp_root.mkdir()
    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))
    monkeypatch.setattr(
        "videotrans.translator.run",
        lambda **kwargs: [{"line": 1, "time": "00:00:00,000 --> 00:00:01,000", "text": "Hola"}],
    )
    res_missing = orchestrator.run_staged_translation(
        TaskRequest({
            "name": str(src_video),
            "target_dir": str(out_dir),
            "project_id": missing_pid,
            "source_language_code": "en",
            "target_language_code": "es",
            "segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Hello"}],
        })
    )
    assert res_missing.status == TaskStatus.SUCCEEDED
    assert not get_project_dir(missing_pid).exists()

    # Mid-save cancellation must leave disk segments.json and SQLite consistent
    proj = create_project(
        name="Disk DB Sync",
        stage=2,
        state={"segments": []},
    )
    pid = proj["id"]
    update_project_state(
        pid,
        {"segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Hello", "targetText": "ORIG"}]},
        stage=2,
    )
    token = CancellationToken()
    orig_update = update_project_state

    def _cancel_before_write(project_id, state_dict, stage=None, status=None, **kwargs):
        token.cancel()
        return orig_update(project_id, state_dict, stage=stage, status=status, **kwargs)

    monkeypatch.setattr("videotrans.core.project_store.update_project_state", _cancel_before_write)
    res_cancelled = orchestrator.run_staged_translation(
        TaskRequest({
            "name": str(src_video),
            "target_dir": str(out_dir),
            "project_id": pid,
            "source_language_code": "en",
            "target_language_code": "es",
            "segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Hello"}],
        }),
        cancellation_token=token,
    )
    assert res_cancelled.status == TaskStatus.CANCELLED
    disk_segs = json.loads((get_project_dir(pid) / "transcripts" / "segments.json").read_text(encoding="utf-8"))
    db_segs = get_project(pid)["state"]["segments"]
    assert disk_segs[0]["targetText"] == "ORIG"
    assert db_segs[0]["targetText"] == "ORIG"


# ---------------------------------------------------------------------------
# Iteration 2 Finding 3: Segment timestamp validation & negative clamping
# ---------------------------------------------------------------------------


def test_project_state_rejects_null_segment_timestamps_with_400_and_clamps_negative_start(isolated_db):
    proj = create_project(name="Timestamp Validation")
    pid = proj["id"]
    client = TestClient(create_app())

    resp_none = client.put(
        f"/api/projects/{pid}/state",
        json={"state": {"segments": [{"id": 1, "startSec": None, "endSec": 1.0, "sourceText": "Hi"}]}},
    )
    assert resp_none.status_code == 400

    resp_non_dict = client.put(
        f"/api/projects/{pid}/state",
        json={"state": {"segments": ["bad-seg"]}},
    )
    assert resp_non_dict.status_code == 400

    resp_fallback = client.put(
        f"/api/projects/{pid}/state",
        json={"state": {"segments": [{"id": 1, "startSec": None, "startTime": "00:00:01,500", "endSec": 2.5, "sourceText": "Fallback"}]}},
    )
    assert resp_fallback.status_code == 200
    srt_fb = (get_project_dir(pid) / "transcripts" / "source.srt").read_text(encoding="utf-8")
    assert "00:00:01,500 --> 00:00:02,500" in srt_fb

    resp_neg = client.put(
        f"/api/projects/{pid}/state",
        json={"state": {"segments": [{"id": 1, "startSec": -1.5, "endSec": 1.0, "sourceText": "Clamped"}]}},
    )
    assert resp_neg.status_code == 200
    srt_neg = (get_project_dir(pid) / "transcripts" / "source.srt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,000" in srt_neg
    assert "23:59:" not in srt_neg


# ---------------------------------------------------------------------------
# Iteration 2 Finding 4: _pyav_sampled_roi_iter VFR timestamp tracking
# ---------------------------------------------------------------------------


def test_pyav_sampled_roi_iter_tracks_actual_vfr_pts_without_drift(tmp_path, monkeypatch):
    import sys
    from fractions import Fraction
    from videotrans.ocr._frame_source import _pyav_sampled_roi_iter

    def _make_fake_av(pts_list: list[int]):
        class _FakeFrame:
            def __init__(self, pts):
                self.pts = pts

            def to_ndarray(self, format="bgr24"):
                return np.zeros((100, 100, 3), dtype=np.uint8)

        class _FakePacket:
            def __init__(self, pts):
                self._pts = pts

            def decode(self):
                return [_FakeFrame(self._pts)]

        class _FakeStream:
            width = 100
            height = 100
            time_base = Fraction(1, 1000)

        class _FakeContainer:
            def __init__(self):
                self.streams = SimpleNamespace(video=[_FakeStream()])

            def seek(self, *args, **kwargs):
                return None

            def demux(self, stream):
                return [_FakePacket(p) for p in pts_list]

            def close(self):
                return None

        return SimpleNamespace(open=lambda *_a, **_kw: _FakeContainer())

    monkeypatch.setitem(sys.modules, "av", _make_fake_av([0, 1000, 2000]))
    vfr_yielded = [
        ts
        for ts, _crop in _pyav_sampled_roi_iter(
            str(tmp_path / "vfr.mp4"),
            roi=(0.0, 0.5, 1.0, 0.5),
            coarse_interval_ms=500,
            start_ms=0,
            end_ms=2500,
            max_roi_width=100,
        )
    ]
    assert vfr_yielded == [0, 1000, 2000]

    # Near-boundary CFR tolerance must still advance next_ts so consecutive frames aren't double-sampled
    monkeypatch.setitem(sys.modules, "av", _make_fake_av([0, 490, 523, 1000]))
    cfr_yielded = [
        ts
        for ts, _crop in _pyav_sampled_roi_iter(
            str(tmp_path / "cfr.mp4"),
            roi=(0.0, 0.5, 1.0, 0.5),
            coarse_interval_ms=500,
            start_ms=0,
            end_ms=1500,
            max_roi_width=100,
        )
    ]
    assert cfr_yielded == [0, 490, 1000]


# ---------------------------------------------------------------------------
# Iteration 2 Finding 7: API input validation returns 400 (never 500)
# ---------------------------------------------------------------------------


def test_api_input_validation_returns_400_for_malformed_translate_and_voice_payloads(tmp_path, monkeypatch):
    voices_dir = tmp_path / "voices"
    previews_dir = tmp_path / "previews"
    voices_dir.mkdir()
    previews_dir.mkdir()
    monkeypatch.setattr("videotrans.core.voice_store.VOICES_DIR", voices_dir)
    monkeypatch.setattr("videotrans.core.voice_store.PREVIEWS_DIR", previews_dir)

    client = TestClient(create_app(), raise_server_exceptions=False)

    # 7a: POST /api/translate with invalid translationMode
    resp_trans = client.post(
        "/api/translate",
        json={
            "segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Hello"}],
            "translationMode": "bogus_mode",
        },
    )
    assert resp_trans.status_code == 400

    # 7b: POST /api/custom-voices with non-dict tuning_params JSON or invalid JSON
    resp_cv_list = client.post(
        "/api/custom-voices",
        data={"name": "TestVoice", "provider": "vieneu", "tuning_params": "[1, 2]"},
    )
    assert resp_cv_list.status_code == 400

    resp_cv_bad_json = client.post(
        "/api/custom-voices",
        data={"name": "TestVoice", "provider": "vieneu", "tuning_params": "{not-json"},
    )
    assert resp_cv_bad_json.status_code == 400

    resp_cv_body_list = client.post(
        "/api/custom-voices",
        content="[1, 2]",
        headers={"content-type": "application/json"},
    )
    assert resp_cv_body_list.status_code == 400

    # 7c: POST /api/voices/preview-clone with malformed JSON or JSON array
    resp_pc_bad = client.post(
        "/api/voices/preview-clone",
        content="{bad-json",
        headers={"content-type": "application/json"},
    )
    assert resp_pc_bad.status_code == 400

    resp_pc_list = client.post(
        "/api/voices/preview-clone",
        content="[1, 2]",
        headers={"content-type": "application/json"},
    )
    assert resp_pc_list.status_code == 400


# ---------------------------------------------------------------------------
# Iteration 3 Finding 1: CapCut export fallback & invalid timestamp -> 400
# ---------------------------------------------------------------------------


def test_capcut_export_handles_none_sec_with_time_fallback_and_rejects_unbacked_none(isolated_db):
    client = TestClient(create_app(), raise_server_exceptions=False)

    # 1a: PUT state with startSec/endSec None + valid startTime/endTime fallback -> CapCut export 200
    proj_fb = create_project(name="CapCut Fallback", duration=5.0, stage=4)
    pid_fb = proj_fb["id"]
    put_res = client.put(
        f"/api/projects/{pid_fb}/state",
        json={
            "stage": 4,
            "state": {
                "segments": [
                    {
                        "id": 1,
                        "startSec": None,
                        "startTime": "00:00:01,500",
                        "endSec": None,
                        "endTime": "00:00:02,500",
                        "sourceText": "Hi",
                        "rawTranslation": "Xin chao",
                        "targetText": "Chao",
                    }
                ]
            },
        },
    )
    assert put_res.status_code == 200

    edited_res = client.get(f"/api/projects/{pid_fb}/export-capcut/subtitles_edited.srt")
    assert edited_res.status_code == 200
    assert "00:00:01,500 --> 00:00:02,500" in edited_res.text
    assert "Chao" in edited_res.text

    target_res = client.get(f"/api/projects/{pid_fb}/export-capcut/subtitles_target.srt")
    assert target_res.status_code == 200
    assert "00:00:01,500 --> 00:00:02,500" in target_res.text
    assert "Xin chao" in target_res.text

    bundle_res = client.get(f"/api/projects/{pid_fb}/export-capcut/bundle.zip")
    assert bundle_res.status_code == 200

    # 1b: POST /api/projects with unbacked startSec: None -> CapCut export returns 400 (never 500)
    create_bad = client.post(
        "/api/projects",
        json={
            "name": "CapCut Invalid None",
            "duration": 2.0,
            "state": {"segments": [{"id": 1, "startSec": None, "endSec": 2.5, "sourceText": "Bad"}]},
        },
    )
    assert create_bad.status_code == 201
    bad_pid = create_bad.json()["id"]
    bad_export = client.get(f"/api/projects/{bad_pid}/export-capcut/subtitles_edited.srt")
    assert bad_export.status_code == 400

    # 1c: POST /api/projects with omitted timing fields preserves default 0.0 -> 1.0 behavior
    create_default = client.post(
        "/api/projects",
        json={
            "name": "CapCut Default Timing",
            "duration": 2.0,
            "state": {"segments": [{"id": 1, "sourceText": "Default"}]},
        },
    )
    assert create_default.status_code == 201
    def_pid = create_default.json()["id"]
    def_export = client.get(f"/api/projects/{def_pid}/export-capcut/subtitles_edited.srt")
    assert def_export.status_code == 200
    assert "00:00:00,000 --> 00:00:01,000" in def_export.text


# ---------------------------------------------------------------------------
# Iteration 3 Finding 2: split_segment & run_staged_translation None fallback
# ---------------------------------------------------------------------------


def test_split_segment_and_run_staged_translation_fallback_on_none_sec(isolated_db, monkeypatch):
    client = TestClient(create_app())
    split_res = client.post(
        "/api/segments/split",
        json={
            "segment": {
                "id": 1,
                "startSec": None,
                "startTime": "00:00:01,500",
                "endSec": None,
                "endTime": "00:00:03,500",
                "sourceText": "Hello world here",
            },
            "splitTime": 2.5,
        },
    )
    assert split_res.status_code == 200, split_res.text
    segs = split_res.json()["segments"]
    assert segs[0]["startSec"] == 1.5
    assert segs[0]["endSec"] == 2.5
    assert segs[1]["startSec"] == 2.5
    assert segs[1]["endSec"] == 3.5

    # Verify run_staged_translation forwards parsed startTime/endTime when startSec/endSec are None
    tmp_path = isolated_db
    src_video = tmp_path / "trans_fb.mp4"
    src_video.write_bytes(b"\x00" * 64)
    out_dir = tmp_path / "trans_out"
    temp_root = tmp_path / "trans_temp"
    temp_root.mkdir()
    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))

    captured = {}

    def _fake_trans_run(**kwargs):
        captured["text_list"] = kwargs["text_list"]
        return [{"line": 1, "time": "00:00:01,500 --> 00:00:03,500", "text": "Xin chao"}]

    monkeypatch.setattr("videotrans.translator.run", _fake_trans_run)
    res = orchestrator.run_staged_translation(
        TaskRequest({
            "name": str(src_video),
            "target_dir": str(out_dir),
            "source_language_code": "en",
            "target_language_code": "vi",
            "segments": [
                {
                    "id": 1,
                    "startSec": None,
                    "startTime": "00:00:01,500",
                    "endSec": None,
                    "endTime": "00:00:03,500",
                    "sourceText": "Hello",
                }
            ],
        })
    )
    assert res.status == TaskStatus.SUCCEEDED
    assert captured["text_list"][0]["start_time"] == 1500
    assert captured["text_list"][0]["end_time"] == 3500
    # Same-language bypass should also compute CPS using 2.0s duration (5 / 2.0 = 2.5)
    res_same = orchestrator.run_staged_translation(
        TaskRequest({
            "name": str(src_video),
            "target_dir": str(out_dir),
            "source_language_code": "en",
            "target_language_code": "en",
            "segments": [
                {
                    "id": 1,
                    "startSec": None,
                    "startTime": "00:00:01,500",
                    "endSec": None,
                    "endTime": "00:00:03,500",
                    "sourceText": "Hello",
                }
            ],
        })
    )
    assert res_same.status == TaskStatus.SUCCEEDED
    assert res_same.segments[0]["cps"] == 2.5


# ---------------------------------------------------------------------------
# Iteration 3 Finding 3: _segment_srt_timestamp OverflowError -> HTTP 400
# ---------------------------------------------------------------------------


def test_project_state_rejects_overflow_timestamps_with_400(isolated_db):
    proj = create_project(name="Overflow Timestamp")
    pid = proj["id"]
    client = TestClient(create_app(), raise_server_exceptions=False)

    for huge_val in (1e20, 1e308):
        resp = client.put(
            f"/api/projects/{pid}/state",
            json={
                "state": {
                    "segments": [
                        {"id": 1, "startSec": huge_val, "endSec": huge_val + 1.0, "sourceText": "Huge"}
                    ]
                }
            },
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Iteration 3 Finding 4: Malformed multipart form -> HTTP 400 (never 500)
# ---------------------------------------------------------------------------


def test_malformed_multipart_form_returns_400_for_trim_audio_and_batch_upload(tmp_path):
    client = TestClient(create_app(upload_dir=tmp_path / "uploads"), raise_server_exceptions=False)
    bad_headers = {"content-type": "multipart/form-data; boundary=xyz"}
    bad_body = b"not-a-valid-multipart-payload"

    resp_trim = client.post("/api/voices/trim-audio", content=bad_body, headers=bad_headers)
    assert resp_trim.status_code == 400

    resp_batch = client.post("/api/media/batch-upload", content=bad_body, headers=bad_headers)
    assert resp_batch.status_code == 400


# ---------------------------------------------------------------------------
# Iteration 4 Findings 1 & 2: Dubbing assemble & CapCut bounds + None-sec fallback
# ---------------------------------------------------------------------------


def test_dubbing_assemble_and_capcut_reject_extreme_durations_with_400(isolated_db, monkeypatch):
    from videotrans.core import voice_store
    from videotrans.services.voice_preview import generate_fallback_preview_wav

    tmp_path = isolated_db
    monkeypatch.setattr(voice_store, "PREVIEWS_DIR", tmp_path / "previews")
    voice_store.init_voice_dirs()
    generate_fallback_preview_wav(voice_store.get_preview_audio_path("prev_aaaaaaaaaaaa.wav"), duration_sec=0.5)

    proj_ok = create_project(name="Assemble Huge Seg", duration=5.0, stage=3)
    proj_big_dur = create_project(name="Assemble Huge ProjDur", duration=1e10, stage=3)
    proj_capcut_1e6 = create_project(name="CapCut 1e6", duration=1e6, stage=4)
    proj_capcut_1e306 = create_project(name="CapCut 1e306", duration=1e306, stage=4)

    client = TestClient(create_app(), raise_server_exceptions=False)

    # Extreme segment start/end values must return 400, never MemoryError / OverflowError 500
    for start_val, end_val in ((1e10, 1e10 + 1.0), (1e305, 2e305)):
        resp = client.post(
            f"/api/projects/{proj_ok['id']}/dubbing/assemble",
            json={
                "segments": [
                    {
                        "id": 1,
                        "startSec": start_val,
                        "endSec": end_val,
                        "targetText": "Huge",
                        "previewAudioId": "prev_aaaaaaaaaaaa",
                    }
                ]
            },
        )
        assert resp.status_code == 400, resp.text

    # Extreme project.duration must return 400, never MemoryError 500
    resp_big_dur = client.post(
        f"/api/projects/{proj_big_dur['id']}/dubbing/assemble",
        json={
            "segments": [
                {
                    "id": 1,
                    "startSec": 0.0,
                    "endSec": 1.0,
                    "targetText": "Normal",
                    "previewAudioId": "prev_aaaaaaaaaaaa",
                }
            ]
        },
    )
    assert resp_big_dur.status_code == 400, resp_big_dur.text

    # CapCut fallback WAV export with huge durations must return 400, never 500
    for huge_proj in (proj_capcut_1e6, proj_capcut_1e306):
        resp_cap = client.get(f"/api/projects/{huge_proj['id']}/export-capcut/voiceover_merged.wav")
        assert resp_cap.status_code == 400, resp_cap.text


def test_dubbing_assemble_none_sec_falls_back_to_time_and_rejects_unbacked_none(isolated_db, monkeypatch):
    import wave
    from videotrans.core import voice_store
    from videotrans.services.voice_preview import generate_fallback_preview_wav

    tmp_path = isolated_db
    monkeypatch.setattr(voice_store, "PREVIEWS_DIR", tmp_path / "previews")
    voice_store.init_voice_dirs()
    generate_fallback_preview_wav(voice_store.get_preview_audio_path("prev_cccccccccccc.wav"), duration_sec=1.0)

    proj = create_project(name="Assemble None Fallback", duration=2.0, stage=3)
    pid = proj["id"]
    client = TestClient(create_app(), raise_server_exceptions=False)

    put_res = client.put(
        f"/api/projects/{pid}/state",
        json={
            "stage": 3,
            "state": {
                "segments": [
                    {
                        "id": 1,
                        "startSec": None,
                        "startTime": "00:00:01,500",
                        "endSec": None,
                        "endTime": "00:00:02,500",
                        "sourceText": "Hi",
                        "targetText": "Xin chao",
                        "previewAudioId": "prev_cccccccccccc",
                    }
                ]
            },
        },
    )
    assert put_res.status_code == 200, put_res.text

    assemble_ok = client.post(
        f"/api/projects/{pid}/dubbing/assemble",
        json={
            "segments": [
                {
                    "id": 1,
                    "startSec": None,
                    "startTime": "00:00:01,500",
                    "endSec": None,
                    "endTime": "00:00:02,500",
                    "targetText": "Xin chao",
                    "previewAudioId": "prev_cccccccccccc",
                }
            ]
        },
    )
    assert assemble_ok.status_code == 200, assemble_ok.text

    output = get_project_dir(pid) / "dubbing" / "voiceover_merged.wav"
    assert output.is_file()
    with wave.open(str(output), "rb") as wav:
        dur_s = wav.getnframes() / float(wav.getframerate())
        assert dur_s >= 2.5 - 0.01
        wav.setpos(int(0.5 * wav.getframerate()))
        assert not any(wav.readframes(1000))
        wav.setpos(int(1.7 * wav.getframerate()))
        assert any(wav.readframes(1000))

    # Unbacked None or malformed fallback string must still return 400
    for bad_seg in (
        {"id": 1, "startSec": None, "endSec": 1.0, "targetText": "Bad", "previewAudioId": "prev_cccccccccccc"},
        {"id": 1, "startSec": None, "startTime": "not-a-time", "endSec": 2.0, "targetText": "Bad", "previewAudioId": "prev_cccccccccccc"},
    ):
        resp_bad = client.post(f"/api/projects/{pid}/dubbing/assemble", json={"segments": [bad_seg]})
        assert resp_bad.status_code == 400, resp_bad.text


# ---------------------------------------------------------------------------
# Iteration 4 Finding 3: POST /api/custom-voices rejects NaN/Infinity tuning_params
# ---------------------------------------------------------------------------


def test_custom_voices_rejects_nonfinite_tuning_params_without_poisoning(isolated_db, monkeypatch):
    from videotrans.core import voice_store

    tmp_path = isolated_db
    voices_dir = tmp_path / "voices"
    previews_dir = tmp_path / "previews"
    voices_dir.mkdir(exist_ok=True)
    previews_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(voice_store, "VOICES_DIR", voices_dir)
    monkeypatch.setattr(voice_store, "PREVIEWS_DIR", previews_dir)
    voice_store.init_voice_dirs()

    client = TestClient(create_app(), raise_server_exceptions=False)

    for bad_tuning in ('{"temp": NaN}', '{"temp": Infinity}', '{"nested": [-Infinity]}'):
        resp = client.post(
            "/api/custom-voices",
            json={
                "name": "Bad Tuning Voice",
                "kind": "design",
                "external_voice_id": "v1",
                "tuning_params": bad_tuning,
            },
        )
        assert resp.status_code == 400, resp.text

    # Also test raw JSON body with non-standard NaN literal in dict tuning_params
    resp_raw = client.post(
        "/api/custom-voices",
        content='{"name": "Raw NaN", "kind": "design", "external_voice_id": "v1", "tuning_params": {"temp": NaN}}',
        headers={"content-type": "application/json"},
    )
    assert resp_raw.status_code == 400, resp_raw.text

    # Listing custom voices must remain healthy (200) and unpoisoned
    list_resp = client.get("/api/custom-voices")
    assert list_resp.status_code == 200
    assert list_resp.json()["voices"] == []


# ---------------------------------------------------------------------------
# Iteration 4 Finding 4: POST /api/ocr/extract rejects NaN/Inf before ocr_extractor
# ---------------------------------------------------------------------------


def test_ocr_extract_rejects_nonfinite_start_end_and_roi_before_extractor(tmp_path, monkeypatch):
    fake_media = tmp_path / "ocr_nan.mp4"
    fake_media.write_bytes(b"mp4")

    app = create_app()
    monkeypatch.setattr(app.state.media_store, "_probe", lambda p: {"time": 5000})
    app.state.media_store.inspect(fake_media, filename="ocr_nan.mp4", media_id="m_ocr_nan")

    called = {"count": 0}

    def _spy_extractor(*args, **kwargs):
        called["count"] += 1
        return {"ok": True, "text": "Should not run"}

    app.state.ocr_extractor = _spy_extractor
    client = TestClient(app, raise_server_exceptions=False)

    bad_payloads = [
        {"mediaId": "m_ocr_nan", "startSec": "nan", "endSec": 1.0, "roi": [0.1, 0.7, 0.8, 0.2]},
        {"mediaId": "m_ocr_nan", "startSec": 0.0, "endSec": "nan", "roi": [0.1, 0.7, 0.8, 0.2]},
        {"mediaId": "m_ocr_nan", "startSec": 0.0, "endSec": "inf", "roi": [0.1, 0.7, 0.8, 0.2]},
        {"mediaId": "m_ocr_nan", "startSec": 0.0, "endSec": 1.0, "roi": ["nan", 0.7, 0.8, 0.2]},
        {"mediaId": "m_ocr_nan", "startSec": 0.0, "endSec": 1.0, "roi": {"x": 0.1, "y": "nan", "width": 0.8, "height": 0.2}},
    ]
    for payload in bad_payloads:
        resp = client.post("/api/ocr/extract", json=payload)
        assert resp.status_code == 400, f"Expected 400 for {payload}, got {resp.status_code}: {resp.text}"
        assert called["count"] == 0


# ---------------------------------------------------------------------------
# Iteration 5 Hardening: Stage 4 render routing, DeepSeek max_tokens,
# blank-cue translation ID alignment, dubbing prompt ellipses, OcrScanner
# checkpoint preexisting merge, and secret_store.is_from_env
# ---------------------------------------------------------------------------


def test_stage4_render_routing_audio_mix_dubbed_and_voiceover_merged_staging(isolated_db, monkeypatch):
    import soundfile as sf
    from videotrans.core.project_store import init_project_dirs

    tmp_path = isolated_db
    media = tmp_path / "render_src.mp4"
    media.write_bytes(b"\x00" * 128)
    proj = create_project(name="Render Stage4", duration=2.0, stage=4)
    pid = proj["id"]
    dirs = init_project_dirs(pid)

    # Create a real Stage 3 voiceover_merged.wav with amplitude 0.5
    merged_wav = dirs["dubbing"] / "voiceover_merged.wav"
    sr = 24000
    samples = np.full((sr,), 0.5, dtype=np.float32)
    sf.write(str(merged_wav), samples, sr, subtype="PCM_16", format="WAV")

    params = build_task_params(
        media,
        {
            "ttsType": 2,
            "audioMix": {"original": 25, "dubbed": 120, "background": 40},
            "segments": [
                {"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Hello", "targetText": "Xin chao", "speakerId": "spk_1"},
            ],
            "speakerVoiceMap": {"spk_1": "Bình (nam miền Bắc)"},
        },
        job_type="render",
        project_id=pid,
        output_dir=tmp_path / "out",
        temp_dir=tmp_path / "temp",
    )
    assert params["volume"] == "+20%"
    assert params["source_audio_volume"] == pytest.approx(0.25)
    assert params["backaudio_volume"] == pytest.approx(0.40)
    assert "Xin chao" in params["subtitles"]
    assert params["voice_role"] == "Bình (nam miền Bắc)"
    assert params["clear_cache"] is False

    # Verify orchestrator.run stages voiceover_merged.wav (scaled by +20%) and skips recogn/trans/dubbing/align
    temp_root = tmp_path / "temp"
    temp_root.mkdir(exist_ok=True)
    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))
    executed_stages: list[str] = []
    observed_target_wav_peak = {"peak": 0.0}

    class _RenderTask:
        def __init__(self, cfg, event_sink=None, cancellation_token=None):
            self.cfg = cfg
            self.uuid = cfg.uuid or "render-u1"
            self.should_recogn = True
            self.should_trans = True
            self.should_dubbing = True
            self.should_recogn2 = False
            self.should_hebing = True
            self.is_audio_trans = False
            self.source_srt_list = []
            self.target_srt_list = []
            self.cfg.source_sub = str(Path(cfg.target_dir) / "en.srt")
            self.cfg.target_sub = str(Path(cfg.target_dir) / "vi.srt")
            self.cfg.target_wav = str(Path(cfg.cache_folder) / "target.wav")
            self.cfg.targetdir_mp4 = str(Path(cfg.target_dir) / "out.mp4")

        def prepare(self):
            executed_stages.append("prepare")

        def recogn(self):
            executed_stages.append("recogn")

        def diariz(self):
            executed_stages.append("diariz")

        def trans(self):
            executed_stages.append("trans")

        def dubbing(self):
            executed_stages.append("dubbing")

        def align(self):
            executed_stages.append("align")

        def assembling(self):
            executed_stages.append("assembling")
            data, _ = sf.read(self.cfg.target_wav, dtype="float32")
            observed_target_wav_peak["peak"] = float(np.max(np.abs(data)))
            Path(self.cfg.targetdir_mp4).write_bytes(b"rendered-mp4")

        def task_done(self):
            executed_stages.append("task_done")

    monkeypatch.setattr(orchestrator, "TransCreate", _RenderTask)
    res = orchestrator.run(TaskRequest(params))
    assert res.status == TaskStatus.SUCCEEDED
    assert executed_stages == ["prepare", "assembling", "task_done"]
    # 0.5 * 1.2 (+20% volume) == 0.6
    assert observed_target_wav_peak["peak"] == pytest.approx(0.6, abs=0.02)


def test_deepseek_max_tokens_cap_and_prompt_no_leading_ellipses(monkeypatch):
    from videotrans.translator import _deepseek as ds_mod

    monkeypatch.setattr(
        ds_mod,
        "params",
        {"deepseek_model": "deepseek-chat", "deepseek_api": "https://api.deepseek.com/v1", "deepseek_key": "k", "deepseek_max_token": 65536, "deepseek_thinking": False},
    )
    ds_chat = ds_mod.DeepSeek()
    assert ds_chat.max_tokens == 8192

    monkeypatch.setattr(
        ds_mod,
        "params",
        {"deepseek_model": "deepseek-reasoner", "deepseek_api": "https://api.deepseek.com/v1", "deepseek_key": "k", "deepseek_max_token": 65536, "deepseek_thinking": True},
    )
    ds_reasoner = ds_mod.DeepSeek()
    assert ds_reasoner.max_tokens == 32768

    root = Path(__file__).resolve().parents[1] / "videotrans" / "prompts"
    for p in list((root / "srt").glob("*.txt")) + list((root / "text").glob("*.txt")):
        txt = p.read_text(encoding="utf-8")
        assert "...ir al hospital" not in txt
        assert "...no significa" not in txt


def test_blank_cue_translation_id_alignment_api_and_orchestrator(isolated_db, monkeypatch):
    seen_text_lists = []

    def _fake_translator_run(**kwargs):
        tl = list(kwargs.get("text_list") or [])
        seen_text_lists.append(tl)
        return [
            {**item, "text": f"VI_{item['text']}"}
            for item in tl
        ]

    monkeypatch.setattr("videotrans.translator.run", _fake_translator_run)

    client = TestClient(create_app())
    resp = client.post(
        "/api/translate",
        json={
            "sourceLanguage": "en",
            "targetLanguage": "vi",
            "translateType": 0,
            "segments": [
                {"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "First line", "text": "First line"},
                {"id": 2, "startSec": 1.0, "endSec": 2.0, "sourceText": "   ", "text": "Stale ASR text that was cleared"},
                {"id": 3, "startSec": 2.0, "endSec": 3.0, "sourceText": "Third line", "text": "Third line"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    out_segs = resp.json()["segments"]
    assert len(out_segs) == 3
    assert [s["id"] for s in out_segs] == [1, 2, 3]
    assert out_segs[0]["targetText"] == "VI_First line"
    assert out_segs[1]["targetText"] == ""
    assert out_segs[2]["targetText"] == "VI_Third line"
    # Translator only received the 2 non-empty segments
    assert len(seen_text_lists) == 1
    assert [it["text"] for it in seen_text_lists[0]] == ["First line", "Third line"]


def test_ocr_scanner_checkpoint_resume_merges_preexisting_segments_on_persist(tmp_path):
    from videotrans.ocr._checkpoint import OcrCheckpoint, load_checkpoint, save_checkpoint

    ckpt_path = tmp_path / "scan_ckpt.json"
    cfg = OcrConfig(
        roi=(0.0, 0.0, 1.0, 1.0),
        language="en",
        coarse_interval_ms=500,
        min_stable_samples=1,
        grace_gap_ms=400,
    )
    scanner_init = OcrScanner(SimpleNamespace(recognize=lambda *_a, **_kw: OcrResult()), cfg, checkpoint_path=str(ckpt_path), frames_are_cropped=True)
    # Seed a compatible checkpoint with an earlier completed segment [0..2000] and last_processed_ms=10000
    save_checkpoint(
        OcrCheckpoint(
            fingerprint=scanner_init._fingerprint(),
            roi=tuple(cfg.roi),
            language=cfg.language,
            provider="paddle",
            model_version=None,
            device="cpu",
            scanner_params=scanner_init._scanner_params(),
            last_processed_ms=10000,
            segments=[{"start_ms": 0, "end_ms": 2000, "text": "Early Cue", "confidence": 0.95, "samples": 3}],
        ),
        str(ckpt_path),
    )

    class _LateProvider:
        def recognize(self, crop, lang=None):
            return OcrResult(text="Late Cue", confidence=0.92)

    scanner_resume = OcrScanner(_LateProvider(), cfg, checkpoint_path=str(ckpt_path), frames_are_cropped=True)
    frame = np.full((32, 120, 3), 200, dtype=np.uint8)
    segs = scanner_resume.scan([(10000, frame), (10500, frame), (11000, frame)], duration_ms=11500, resume=True)

    assert [s.text for s in segs] == ["Early Cue", "Late Cue"]
    reloaded = load_checkpoint(str(ckpt_path))
    assert reloaded is not None
    assert [s["text"] for s in reloaded.segments] == ["Early Cue", "Late Cue"]
    assert reloaded.last_processed_ms >= 11000


def test_secret_store_is_from_env_distinguishes_saved_db_secret_from_external_env(isolated_db, monkeypatch):
    from videotrans.core import secret_store

    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_APIKEY", raising=False)
    secret_store.set_secret("deepgram_apikey", "saved-via-ui-key", persist_to_env=False)
    assert secret_store.get_secret("deepgram_apikey") == "saved-via-ui-key"
    assert secret_store.is_from_env("deepgram_apikey") is False

    # External OS env override with a different value reports fromEnv=True
    monkeypatch.setenv("DEEPGRAM_API_KEY", "external-override-key")
    assert secret_store.is_from_env("deepgram_apikey") is True

