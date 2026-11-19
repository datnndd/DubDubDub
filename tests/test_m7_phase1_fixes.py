# -*- coding: utf-8 -*-
"""Regression and concurrency tests for M7 Phase 1 fixes."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from videotrans.api import task_params as task_params_module
from videotrans.api.app import create_app
from videotrans.api.task_params import build_task_params
from videotrans.configure.config import app_cfg
from videotrans.core.db import get_db, get_db_path, init_db, set_db_path
from videotrans.core.job_manager import JobManager
from videotrans.core.job_store import append_event, create_job, events_since, get_job
from videotrans.core.proc_registry import get_current_job_id, set_current_job_id
from videotrans.core.project_store import create_project, get_project, update_project_state
from videotrans.task import _stage_prepare, orchestrator
from videotrans.task._stage_assemble import AssembleMixin
from videotrans.task.orchestrator import (
    CancellationToken,
    EventKind,
    TaskEvent,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from videotrans.task.taskcfg import InputFile
from videotrans.util import help_misc
from videotrans.util._ffmpeg_runner import runffmpeg
from videotrans.util._ffprobe import get_video_duration, get_video_info, runffprobe


# ---------------------------------------------------------------------------
# Fix 1: orchestrator._embed_thumbnail runs before task_done & cleans staged
# ---------------------------------------------------------------------------


def test_embed_thumbnail_runs_before_task_done_and_persists_to_only_out_mp4(tmp_path, monkeypatch):
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    source = tmp_path / "input.mp4"
    source.write_bytes(b"src")
    thumb = tmp_path / "cover.png"
    thumb.write_bytes(b"png-bytes")
    out_parent = tmp_path / "exports"
    out_parent.mkdir()
    output = out_parent / "job_out"
    output.mkdir()

    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))
    order: list[str] = []

    class _FakeTask:
        def __init__(self, cfg, event_sink=None, cancellation_token=None):
            self.cfg = cfg
            self.should_recogn = False
            self.should_trans = False
            self.should_dubbing = False
            self.should_recogn2 = False
            self.should_hebing = True
            self.cfg.only_out_mp4 = True
            self.cfg.thumbnail = str(thumb)
            self.cfg.targetdir_mp4 = str(output / "result.mp4")

        def prepare(self):
            order.append("prepare")

        def assembling(self):
            order.append("assembling")
            Path(self.cfg.targetdir_mp4).write_bytes(b"raw-mp4")

        def task_done(self):
            order.append("task_done")
            # Simulate AssembleMixin.task_done moving MP4 and deleting cache_folder
            dst = Path(self.cfg.target_dir).parent / Path(self.cfg.targetdir_mp4).name
            Path(self.cfg.targetdir_mp4).replace(dst)
            Path(self.cfg.cache_folder).rmdir()

    def _fake_embed(cfg):
        order.append("embed_thumbnail")
        assert Path(cfg.cache_folder).exists(), "cache_folder was deleted before _embed_thumbnail"
        assert Path(cfg.targetdir_mp4).is_file(), "targetdir_mp4 was moved before _embed_thumbnail"
        Path(cfg.targetdir_mp4).write_bytes(b"mp4-with-thumb")

    monkeypatch.setattr(orchestrator, "TransCreate", _FakeTask)
    monkeypatch.setattr(orchestrator, "_embed_thumbnail", _fake_embed)

    req = TaskRequest({
        "name": str(source),
        "target_dir": str(output),
        "cache_folder": str(temp_root / "job_cache"),
        "source_language_code": "en",
        "target_language_code": "vi",
        "only_out_mp4": True,
        "thumbnail": str(thumb),
    })
    res = orchestrator.run(req)
    assert res.status == TaskStatus.SUCCEEDED
    assert order == ["prepare", "assembling", "embed_thumbnail", "task_done"]
    final_mp4 = out_parent / "result.mp4"
    assert final_mp4.read_bytes() == b"mp4-with-thumb"


def test_embed_thumbnail_guards_and_cleans_staged_on_failure(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"orig-video")
    thumb = tmp_path / "thumb.png"
    thumb.write_bytes(b"png")

    # Guard: only_out_dubbed_audio skips
    called = []
    monkeypatch.setattr("videotrans.util.help_ffmpeg.runffmpeg", lambda cmd: called.append(cmd))
    orchestrator._embed_thumbnail(
        SimpleNamespace(only_out_dubbed_audio=True, thumbnail=str(thumb), targetdir_mp4=str(video))
    )
    assert called == []

    # Guard: missing thumbnail or missing video skips
    orchestrator._embed_thumbnail(
        SimpleNamespace(only_out_dubbed_audio=False, thumbnail=None, targetdir_mp4=str(video))
    )
    orchestrator._embed_thumbnail(
        SimpleNamespace(only_out_dubbed_audio=False, thumbnail=str(thumb), targetdir_mp4=str(tmp_path / "missing.mp4"))
    )
    assert called == []

    # Cleanup: if runffmpeg fails after creating staged file, finally removes it
    def _failing_ffmpeg(cmd):
        staged_path = Path(cmd[-1])
        staged_path.write_bytes(b"partial-staged")
        raise RuntimeError("simulated ffmpeg failure")

    monkeypatch.setattr("videotrans.util.help_ffmpeg.runffmpeg", _failing_ffmpeg)
    with pytest.raises(RuntimeError, match="simulated ffmpeg failure"):
        orchestrator._embed_thumbnail(
            SimpleNamespace(only_out_dubbed_audio=False, thumbnail=str(thumb), targetdir_mp4=str(video))
        )
    assert not (tmp_path / f".thumbnail-{video.name}").exists()
    assert video.read_bytes() == b"orig-video"


# ---------------------------------------------------------------------------
# Fix 2: _stage_assemble audio-less video_ocr across subtitle_type {0,1,2,3,4}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subtitle_type", [0, 1, 2, 3, 4])
def test_assemble_audioless_video_preserves_duration_across_subtitle_types(tmp_path, subtitle_type):
    cache_dir = tmp_path / "cache"
    target_dir = tmp_path / "out"
    cache_dir.mkdir()
    target_dir.mkdir()

    src_video = tmp_path / "silent_src.mp4"
    novoice_mp4 = cache_dir / "novoice.mp4"
    # Synthesize a real 2.5s silent H.264 MP4
    runffmpeg([
        "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10:d=2.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", src_video.as_posix(),
    ], force_cpu=True)
    novoice_mp4.write_bytes(src_video.read_bytes())

    srt_path = target_dir / "target.srt"
    # Short cue ending at 0.8s (well before 2.5s) to prove -shortest does NOT truncate soft subs
    srt_path.write_text("1\n00:00:00,100 --> 00:00:00,800\nHello silent video\n", encoding="utf-8")
    src_srt_path = target_dir / "source.srt"
    src_srt_path.write_text("1\n00:00:00,100 --> 00:00:00,800\nHello silent video\n", encoding="utf-8")

    class _AssembleTask(AssembleMixin):
        def __init__(self):
            self.uuid = "silent-assemble"
            self.hasend = False
            self.precent = 90
            self.cost_duration = time.time()
            self.should_hebing = True
            self.should_dubbing = False
            self.is_audio_trans = False
            self.is_copy_video = True
            self.video_codec_num = "264"
            self.queue_tts = []
            self.video_info = {
                "time": 2500,
                "streams_audio": 0,
                "video_streams": 1,
                "video_fps": 10,
            }
            self.cfg = SimpleNamespace(
                name=src_video.as_posix(),
                cache_folder=cache_dir.as_posix(),
                target_dir=target_dir.as_posix(),
                novoice_mp4=novoice_mp4.as_posix(),
                target_wav=(cache_dir / "target.wav").as_posix(),
                source_wav_output=(target_dir / "source.m4a").as_posix(),
                target_wav_output=(target_dir / "target.m4a").as_posix(),
                targetdir_mp4=(target_dir / "final.mp4").as_posix(),
                source_sub=src_srt_path.as_posix(),
                target_sub=srt_path.as_posix(),
                source_language_code="en",
                target_language_code="en",
                subtitle_type=subtitle_type,
                video_autorate=False,
                app_mode="biaozhun",
                only_out_mp4=False,
                only_out_dubbed_audio=False,
                subtitle_style=None,
            )

        def _exit(self):
            return False

        def signal(self, **_kwargs):
            return None

        def _process_subtitles(self):
            cached_srt = cache_dir / "end.srt"
            cached_srt.write_text(srt_path.read_text(encoding="utf-8"), encoding="utf-8")
            return "end.srt", "eng"

    task = _AssembleTask()
    task.assembling()
    task.hasend = True

    out_mp4 = Path(task.cfg.targetdir_mp4)
    assert out_mp4.is_file()
    info = get_video_info(out_mp4.as_posix())
    assert info["video_streams"] >= 1
    assert info["streams_audio"] == 0
    dur_ms = get_video_duration(out_mp4.as_posix())
    # Must preserve full ~2500ms duration (not truncated to 800ms by -shortest)
    assert dur_ms >= 2200, f"Video truncated for subtitle_type={subtitle_type}: dur_ms={dur_ms}"

    if subtitle_type in (2, 4):
        probe = json.loads(runffprobe(["-v", "quiet", "-print_format", "json", "-show_streams", out_mp4.as_posix()]))
        sub_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "subtitle"]
        assert sub_streams, f"Expected subtitle stream for subtitle_type={subtitle_type}"


def test_assemble_dubbing_on_audioless_video_skips_origin_audio_thread(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    target_dir = tmp_path / "out"
    cache_dir.mkdir()
    target_dir.mkdir()

    src_video = tmp_path / "silent_src.mp4"
    novoice_mp4 = cache_dir / "novoice.mp4"
    runffmpeg([
        "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10:d=1.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", src_video.as_posix(),
    ], force_cpu=True)
    novoice_mp4.write_bytes(src_video.read_bytes())

    dub_wav = cache_dir / "target.wav"
    runffmpeg([
        "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1.5",
        "-c:a", "pcm_s16le", dub_wav.as_posix(),
    ], force_cpu=True)

    spawned_threads: list[str] = []
    orig_thread = threading.Thread

    class _SpyThread(orig_thread):
        def __init__(self, *args, **kwargs):
            target = kwargs.get("target")
            spawned_threads.append(getattr(target, "__name__", str(target)))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("videotrans.task._stage_assemble.threading.Thread", _SpyThread)

    class _DubTask(AssembleMixin):
        def __init__(self):
            self.uuid = "dub-silent"
            self.hasend = False
            self.precent = 90
            self.cost_duration = time.time()
            self.should_hebing = True
            self.should_dubbing = True
            self.is_audio_trans = False
            self.is_copy_video = True
            self.video_codec_num = "264"
            self.queue_tts = []
            self.video_info = {"time": 1500, "streams_audio": 0, "video_streams": 1, "video_fps": 10}
            self.cfg = SimpleNamespace(
                name=src_video.as_posix(),
                cache_folder=cache_dir.as_posix(),
                target_dir=target_dir.as_posix(),
                novoice_mp4=novoice_mp4.as_posix(),
                target_wav=dub_wav.as_posix(),
                source_wav_output=(target_dir / "source.m4a").as_posix(),
                target_wav_output=(target_dir / "target.m4a").as_posix(),
                targetdir_mp4=(target_dir / "final.mp4").as_posix(),
                source_sub=(target_dir / "source.srt").as_posix(),
                target_sub=(target_dir / "target.srt").as_posix(),
                source_language_code="en",
                target_language_code="en",
                subtitle_type=0,
                video_autorate=False,
                app_mode="biaozhun",
                only_out_mp4=False,
                only_out_dubbed_audio=False,
                subtitle_style=None,
            )

        def _exit(self):
            return False

        def signal(self, **_kwargs):
            return None

        def _back_music(self):
            return None

        def _mix_original_audio(self):
            return None

        def _separate(self):
            return None

    task = _DubTask()
    task.assembling()
    task.hasend = True

    assert "_output" not in spawned_threads
    info = get_video_info(task.cfg.targetdir_mp4)
    assert info["streams_audio"] == 1


# ---------------------------------------------------------------------------
# Fix 3: SUCCEEDED vs job.outputs atomicity & SSE subscribe ordering
# ---------------------------------------------------------------------------


def test_succeeded_status_is_atomic_with_outputs_and_sse_never_stalls(tmp_path):
    orig_db = get_db_path()
    set_db_path(tmp_path / "race.db")
    try:
        out_mp4 = tmp_path / "rendered.mp4"
        out_mp4.write_bytes(b"mp4-content")
        src_mp4 = tmp_path / "src.mp4"
        src_mp4.write_bytes(b"src-content")

        def racing_runner(req, sink, _token):
            sink(TaskEvent("job", EventKind.RUNNING, "assembling", "Running"))
            # Emit SUCCEEDED *before* returning TaskResult and sleep 80ms to widen race window
            sink(
                TaskEvent(
                    "job",
                    EventKind.SUCCEEDED,
                    "task_done",
                    "Done",
                    details={
                        "outputs": (str(out_mp4),),
                        "segments": [{"id": 1, "text": "hello"}],
                        "transcript_options": {"utterances": [{"id": 1, "text": "hello"}]},
                        "asr_duration": 1.25,
                    },
                )
            )
            time.sleep(0.08)
            return TaskResult(
                job_id="job",
                status=TaskStatus.SUCCEEDED,
                output_dir=tmp_path,
                outputs=(out_mp4,),
                segments=({"id": 1, "text": "hello"},),
                asr_duration=1.25,
                transcript_options={"utterances": [{"id": 1, "text": "hello"}]},
            )

        mgr = JobManager(runner=racing_runner)
        app = create_app(
            upload_dir=tmp_path / "uploads",
            job_manager=mgr,
            media_probe=lambda _p: {"time": 1000, "width": 320, "height": 240, "video_streams": 1, "streams_audio": 0},
            asr_validator=lambda *_: None,
            translation_validator=lambda *_: None,
            gpu_initializer=lambda: None,
        )

        async def _run_race_check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                up = await client.post("/api/media", files={"file": ("src.mp4", b"src-content", "video/mp4")})
                assert up.status_code == 201
                media_id = up.json()["id"]

                sub = await client.post(
                    "/api/jobs",
                    json={
                        "mediaId": media_id,
                        "jobType": "render",
                        "options": {"sourceLanguage": "en", "targetLanguage": "vi", "subtitles": "1\n00:00:00,000 --> 00:00:01,000\nhi\n"},
                    },
                )
                assert sub.status_code == 202
                job_id = sub.json()["id"]

                async def _poll_loop():
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        r = await client.get(f"/api/jobs/{job_id}")
                        assert r.status_code == 200
                        snap = r.json()
                        if snap["status"] == "succeeded":
                            # Critical invariant: first observation of succeeded MUST already have outputs!
                            assert len(snap["outputs"]) == 1, f"Observed succeeded with empty outputs: {snap}"
                            assert snap["outputs"][0]["name"] == "rendered.mp4"
                            dl = await client.get(snap["outputs"][0]["url"])
                            assert dl.status_code == 200
                            assert dl.content == b"mp4-content"
                            segs = await client.get(f"/api/jobs/{job_id}/segments")
                            assert segs.status_code == 200
                            assert segs.json()["segments"] == [{"id": 1, "text": "hello"}]
                            assert segs.json()["transcriptOptions"] == {"utterances": [{"id": 1, "text": "hello"}]}
                            return snap
                        await asyncio.sleep(0.002)
                    raise AssertionError("Poller timed out")

                async def _sse_loop():
                    chunks: list[str] = []
                    async with client.stream("GET", f"/api/jobs/{job_id}/stream") as resp:
                        assert resp.status_code == 200
                        async for chunk in resp.aiter_text():
                            chunks.append(chunk)
                            if "event: done" in chunk:
                                break
                    body = "".join(chunks)
                    assert "event: done" in body
                    assert '"status": "succeeded"' in body
                    # Ensure no duplicate SSE event IDs
                    ids = [line for line in body.splitlines() if line.startswith("id: ")]
                    assert len(ids) == len(set(ids)), f"Duplicate SSE ids: {ids}"

                snap, _ = await asyncio.gather(_poll_loop(), _sse_loop())
                assert snap["asrDuration"] == 1.25

        asyncio.run(_run_race_check())
    finally:
        set_db_path(orig_db)


# ---------------------------------------------------------------------------
# Fix 4: SQLite WAL / busy_timeout / init_db / BEGIN IMMEDIATE contention
# ---------------------------------------------------------------------------


def test_sqlite_16_thread_contention_init_db_and_append_event(tmp_path):
    orig_db = get_db_path()
    db_file = tmp_path / "contended.db"
    set_db_path(db_file)
    try:
        init_db(db_file)
        proj = create_project(name="Contention Proj")
        shared_job = create_job("shared-job", type="render", project_id=proj["id"])
        assert shared_job["id"] == "shared-job"

        conn = get_db(db_file)
        try:
            assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
            assert int(conn.execute("PRAGMA busy_timeout").fetchone()[0]) == 5000
            assert int(conn.execute("PRAGMA synchronous").fetchone()[0]) == 1  # NORMAL
        finally:
            conn.close()

        barrier = threading.Barrier(16)
        errors: list[BaseException] = []

        def _worker(tid: int):
            try:
                barrier.wait(timeout=5.0)
                init_db(db_file)
                p = create_project(name=f"Proj-{tid}")
                j = create_job(f"job-{tid}", type="asr", project_id=p["id"])
                assert get_job(j["id"]) is not None
                for k in range(10):
                    append_event("shared-job", {"tid": tid, "k": k})
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        assert not errors, f"SQLite contention errors: {errors!r}"
        evts = events_since("shared-job", after_seq=0, limit=500)
        seqs = [e["seq"] for e in evts]
        assert seqs == list(range(1, 161))
    finally:
        set_db_path(orig_db)


# ---------------------------------------------------------------------------
# Fix 5: task_params null-safe camelCase/snake_case fallback
# ---------------------------------------------------------------------------


def test_task_params_null_camel_case_does_not_shadow_snake_case(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"vid")
    monkeypatch.setattr(
        task_params_module,
        "format_video",
        lambda _p: InputFile(
            name=source.as_posix(),
            dirname=tmp_path.as_posix(),
            basename="clip.mp4",
            noextname="clip",
            ext="mp4",
            uuid="null-shadow-uuid",
        ),
    )

    params = build_task_params(
        source,
        {
            "subtitleSource": "video_ocr",
            "ocrRoi": None,
            "ocr_roi": [0.1, 0.7, 0.8, 0.2],
            "ocrRoiConfirmed": None,
            "ocr_roi_confirmed": True,
            "ocrOptions": {
                "coarseIntervalMs": None,
                "coarse_interval_ms": 250,
                "useAngleCls": None,
                "use_angle_cls": True,
                "paragraphMode": None,
                "paragraph_mode": "unwrap",
            },
        },
        job_type="asr",
        temp_dir=str(tmp_path / "tmp"),
        output_dir=tmp_path / "out",
    )
    assert params["ocr_roi"] == (0.1, 0.7, 0.8, 0.2)
    assert params["ocr_roi_confirmed"] is True
    assert params["ocr_options"] == {
        "coarse_interval_ms": 250,
        "use_angle_cls": True,
        "paragraph_mode": "unwrap",
    }

    # Explicit False on camelCase must still win over snake_case True
    with pytest.raises(ValueError, match="ocrRoiConfirmed must be true"):
        build_task_params(
            source,
            {
                "subtitleSource": "video_ocr",
                "ocrRoi": [0.1, 0.7, 0.8, 0.2],
                "ocrRoiConfirmed": False,
                "ocr_roi_confirmed": True,
            },
            job_type="asr",
            temp_dir=str(tmp_path / "tmp"),
            output_dir=tmp_path / "out",
        )

    # None on camelCase coarseIntervalMs must NOT shadow boolean validation error on snake_case
    with pytest.raises(ValueError, match="ocrOptions.coarseIntervalMs must be an integer"):
        build_task_params(
            source,
            {
                "subtitleSource": "video_ocr",
                "ocrRoi": [0.1, 0.7, 0.8, 0.2],
                "ocrRoiConfirmed": True,
                "ocrOptions": {"coarseIntervalMs": None, "coarse_interval_ms": True},
            },
            job_type="asr",
            temp_dir=str(tmp_path / "tmp"),
            output_dir=tmp_path / "out",
        )


# ---------------------------------------------------------------------------
# Fix 6 (M7 Phase 1 Iter 2): is_novoice_mp4 / _split_novoice_byraw race
# ---------------------------------------------------------------------------


def test_orchestrator_mpeg4_soft_sub_succeeds_with_default_current_status_stop(tmp_path, monkeypatch):
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    src_video = tmp_path / "mpeg4_src.mp4"
    runffmpeg([
        "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=10:d=3.0",
        "-c:v", "mpeg4", "-pix_fmt", "yuv420p", src_video.as_posix(),
    ], force_cpu=True)

    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))
    assert app_cfg.current_status == "stop"

    req = TaskRequest({
        "name": str(src_video),
        "target_dir": str(out_dir),
        "source_language_code": "en",
        "target_language_code": "en",
        "voice_role": "No",
        "subtitle_type": 2,
        "subtitles": "1\n00:00:00,100 --> 00:00:01,000\nHello mpeg4\n",
    })
    res = orchestrator.run(req)
    assert res.status == TaskStatus.SUCCEEDED, f"Unexpected failure: {res.failure}"
    mp4_outputs = [p for p in res.outputs if p.suffix == ".mp4"]
    assert mp4_outputs, f"Missing rendered mp4 in {res.outputs}"
    assert get_video_duration(mp4_outputs[0].as_posix()) >= 2900


def test_novoice_wait_cancellation_error_propagation_and_contextvar(tmp_path, monkeypatch):
    # 1. ContextVar propagation & Thread return from run_in_threadpool
    seen_job_ids: list[str | None] = []
    set_current_job_id("ctx-job-42")
    try:
        t = _stage_prepare.run_in_threadpool(lambda: seen_job_ids.append(get_current_job_id()))
        assert isinstance(t, threading.Thread)
        t.join(timeout=2.0)
    finally:
        set_current_job_id(None)
    assert seen_job_ids == ["ctx-job-42"]

    # 2. Error propagation from queue_novice fails fast
    err_uuid = "novoice-err-prop"
    app_cfg.queue_novice[err_uuid] = "error:ffmpeg failed"
    try:
        with pytest.raises(RuntimeError, match="error:ffmpeg failed"):
            help_misc.is_novoice_mp4((tmp_path / "missing.mp4").as_posix(), err_uuid)
    finally:
        app_cfg.queue_novice.pop(err_uuid, None)

    # 3. Mid-prep cancellation in orchestrator returns TaskStatus.CANCELLED (not VideoTransError)
    temp_root = tmp_path / "temp_cancel"
    temp_root.mkdir()
    out_dir = tmp_path / "out_cancel"
    out_dir.mkdir()
    src_video = tmp_path / "mpeg4_cancel.mp4"
    runffmpeg([
        "-y", "-f", "lavfi", "-i", "color=c=red:s=320x240:r=10:d=3.0",
        "-c:v", "mpeg4", "-pix_fmt", "yuv420p", src_video.as_posix(),
    ], force_cpu=True)

    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))
    token = CancellationToken()

    def _cancel_on_wait(event):
        if event.kind == EventKind.STAGE_STARTED and event.stage == "assembling":
            token.cancel()

    # Slow down _split_novoice_byraw so assembling enters is_novoice_mp4 wait while active
    orig_split = _stage_prepare.PrepareMixin._split_novoice_byraw

    def _delayed_split(self):
        time.sleep(0.15)
        return orig_split(self)

    monkeypatch.setattr(_stage_prepare.PrepareMixin, "_split_novoice_byraw", _delayed_split)

    req = TaskRequest({
        "name": str(src_video),
        "target_dir": str(out_dir),
        "source_language_code": "en",
        "target_language_code": "en",
        "voice_role": "No",
        "subtitle_type": 2,
        "subtitles": "1\n00:00:00,100 --> 00:00:01,000\nCancel test\n",
    })
    res = orchestrator.run(req, event_sink=_cancel_on_wait, cancellation_token=token)
    assert res.status == TaskStatus.CANCELLED


def test_task_done_joins_alive_novoice_thread(tmp_path):
    joined = {"done": False}
    ready_evt = threading.Event()

    def _slow_worker():
        ready_evt.set()
        time.sleep(0.1)
        joined["done"] = True

    class _Task(AssembleMixin):
        def __init__(self):
            self.uuid = "join-novoice"
            self.hasend = False
            self.precent = 90
            self.cost_duration = time.time()
            self.is_audio_trans = False
            self.cfg = SimpleNamespace(
                name="src.mp4",
                app_mode="biaozhun",
                only_out_dubbed_audio=False,
                only_out_mp4=False,
                target_dir=str(tmp_path),
                targetdir_mp4=str(tmp_path / "out.mp4"),
                target_wav=str(tmp_path / "target.wav"),
            )
            self._novoice_thread = threading.Thread(target=_slow_worker, daemon=True)
            self._novoice_thread.start()
            ready_evt.wait(timeout=1.0)

        def _exit(self):
            return False

        def set_end(self, _ok):
            assert joined["done"] is True
            self.hasend = True

    task = _Task()
    task.task_done()
    assert task.hasend is True


# ---------------------------------------------------------------------------
# Fix 7 (M7 Phase 1 Iter 2): -shortest soft-sub, stale will_embed.m4a, thumbnail guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subtitle_type", [2, 4])
@pytest.mark.parametrize("should_dubbing", [False, True])
def test_assemble_soft_sub_with_audio_preserves_full_duration(tmp_path, subtitle_type, should_dubbing):
    cache_dir = tmp_path / "cache"
    target_dir = tmp_path / "out"
    cache_dir.mkdir()
    target_dir.mkdir()

    src_video = tmp_path / "src_with_audio.mp4"
    novoice_mp4 = cache_dir / "novoice.mp4"
    runffmpeg([
        "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10:d=3.0",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", "3.0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        src_video.as_posix(),
    ], force_cpu=True)
    runffmpeg(["-y", "-i", src_video.as_posix(), "-an", "-c:v", "copy", novoice_mp4.as_posix()], force_cpu=True)

    target_wav = cache_dir / "target.wav"
    if should_dubbing:
        runffmpeg([
            "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", "3.0", "-c:a", "pcm_s16le", target_wav.as_posix(),
        ], force_cpu=True)

    srt_path = target_dir / "target.srt"
    srt_path.write_text("1\n00:00:00,100 --> 00:00:01,000\nShort cue\n", encoding="utf-8")

    class _SoftSubAudioTask(AssembleMixin):
        def __init__(self):
            self.uuid = f"soft-sub-{subtitle_type}-{should_dubbing}"
            self.hasend = False
            self.precent = 90
            self.cost_duration = time.time()
            self.should_hebing = True
            self.should_dubbing = should_dubbing
            self.is_audio_trans = False
            self.is_copy_video = True
            self.video_codec_num = "264"
            self.queue_tts = []
            self.video_info = {"time": 3000, "streams_audio": 1, "video_streams": 1, "video_fps": 10}
            self.cfg = SimpleNamespace(
                name=src_video.as_posix(),
                cache_folder=cache_dir.as_posix(),
                target_dir=target_dir.as_posix(),
                novoice_mp4=novoice_mp4.as_posix(),
                target_wav=target_wav.as_posix(),
                source_wav_output=(target_dir / "source.m4a").as_posix(),
                target_wav_output=(target_dir / "target.m4a").as_posix(),
                targetdir_mp4=(target_dir / "final.mp4").as_posix(),
                source_sub=srt_path.as_posix(),
                target_sub=srt_path.as_posix(),
                source_language_code="en",
                target_language_code="en",
                subtitle_type=subtitle_type,
                video_autorate=False,
                app_mode="biaozhun",
                only_out_mp4=False,
                only_out_dubbed_audio=False,
                subtitle_style=None,
            )

        def _exit(self):
            return False

        def signal(self, **_kwargs):
            return None

        def _back_music(self):
            return None

        def _mix_original_audio(self):
            return None

        def _separate(self):
            return None

        def _process_subtitles(self):
            cached_srt = cache_dir / "end.srt"
            cached_srt.write_text(srt_path.read_text(encoding="utf-8"), encoding="utf-8")
            return "end.srt", "eng"

    task = _SoftSubAudioTask()
    task.assembling()
    task.hasend = True

    out_mp4 = Path(task.cfg.targetdir_mp4)
    assert out_mp4.is_file()
    dur_ms = get_video_duration(out_mp4.as_posix())
    assert dur_ms >= 2900, f"Soft-sub audio mux truncated output: {dur_ms}ms"
    probe = json.loads(runffprobe(["-v", "quiet", "-print_format", "json", "-show_streams", out_mp4.as_posix()]))
    sub_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "subtitle"]
    assert sub_streams, f"Expected subtitle stream for subtitle_type={subtitle_type}"


def test_assemble_unlinks_stale_will_embed_m4a_on_audioless_rerun(tmp_path):
    cache_dir = tmp_path / "cache"
    target_dir = tmp_path / "out"
    cache_dir.mkdir()
    target_dir.mkdir()

    src_video = tmp_path / "silent.mp4"
    novoice_mp4 = cache_dir / "novoice.mp4"
    runffmpeg([
        "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10:d=1.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", src_video.as_posix(),
    ], force_cpu=True)
    novoice_mp4.write_bytes(src_video.read_bytes())

    dub_wav = cache_dir / "target.wav"
    runffmpeg([
        "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", "1.5", "-c:a", "pcm_s16le", dub_wav.as_posix(),
    ], force_cpu=True)

    class _RerunTask(AssembleMixin):
        def __init__(self, should_dubbing: bool):
            self.uuid = f"stale-m4a-{should_dubbing}"
            self.hasend = False
            self.precent = 90
            self.cost_duration = time.time()
            self.should_hebing = True
            self.should_dubbing = should_dubbing
            self.is_audio_trans = False
            self.is_copy_video = True
            self.video_codec_num = "264"
            self.queue_tts = []
            self.video_info = {"time": 1500, "streams_audio": 0, "video_streams": 1, "video_fps": 10}
            self.cfg = SimpleNamespace(
                name=src_video.as_posix(),
                cache_folder=cache_dir.as_posix(),
                target_dir=target_dir.as_posix(),
                novoice_mp4=novoice_mp4.as_posix(),
                target_wav=dub_wav.as_posix(),
                source_wav_output=(target_dir / "source.m4a").as_posix(),
                target_wav_output=(target_dir / "target.m4a").as_posix(),
                targetdir_mp4=(target_dir / "final.mp4").as_posix(),
                source_sub=(target_dir / "source.srt").as_posix(),
                target_sub=(target_dir / "target.srt").as_posix(),
                source_language_code="en",
                target_language_code="en",
                subtitle_type=0,
                video_autorate=False,
                app_mode="biaozhun",
                only_out_mp4=False,
                only_out_dubbed_audio=False,
                subtitle_style=None,
            )

        def _exit(self):
            return False

        def signal(self, **_kwargs):
            return None

        def _back_music(self):
            return None

        def _mix_original_audio(self):
            return None

        def _separate(self):
            return None

    t1 = _RerunTask(should_dubbing=True)
    t1.assembling()
    t1.hasend = True
    assert (cache_dir / "will_embed.m4a").is_file()
    assert get_video_info(t1.cfg.targetdir_mp4)["streams_audio"] == 1

    # Rerun with voiceRole="No" (should_dubbing=False) on same cache_folder
    t2 = _RerunTask(should_dubbing=False)
    t2.assembling()
    t2.hasend = True
    assert not (cache_dir / "will_embed.m4a").exists()
    assert get_video_info(t2.cfg.targetdir_mp4)["streams_audio"] == 0


@pytest.mark.parametrize("ext", [".mkv", ".webm"])
def test_embed_thumbnail_skips_non_mp4_containers(tmp_path, monkeypatch, ext):
    video = tmp_path / f"video{ext}"
    video.write_bytes(b"container-bytes")
    thumb = tmp_path / "cover.png"
    thumb.write_bytes(b"png-bytes")
    called = []
    monkeypatch.setattr("videotrans.util.help_ffmpeg.runffmpeg", lambda cmd: called.append(cmd))
    orchestrator._embed_thumbnail(
        SimpleNamespace(only_out_dubbed_audio=False, thumbnail=str(thumb), targetdir_mp4=str(video))
    )
    assert called == []
    assert video.read_bytes() == b"container-bytes"


# ---------------------------------------------------------------------------
# Fix 8 (M7 Phase 1 Iter 2): SSE 100-round race, cancel monotonicity, rehydration
# ---------------------------------------------------------------------------


def test_sse_100_rounds_fast_completion_never_drops_terminal_event_or_metadata(tmp_path):
    orig_db = get_db_path()
    set_db_path(tmp_path / "sse_race.db")
    try:
        init_db()

        def fast_runner(req, sink, _token):
            params = req.params
            if params.get("should_fail"):
                sink(TaskEvent("j", EventKind.FAILED, "recogn", "Simulated ASR failure"))
                return TaskResult("j", TaskStatus.FAILED, tmp_path)
            sink(TaskEvent("j", EventKind.SUCCEEDED, "task_done", "Fast complete"))
            return TaskResult("j", TaskStatus.SUCCEEDED, tmp_path)

        mgr = JobManager(runner=fast_runner, asr_runner=fast_runner)
        app = create_app(
            upload_dir=tmp_path / "uploads",
            job_manager=mgr,
            media_probe=lambda _p: {"time": 1000, "width": 320, "height": 240, "video_streams": 1, "streams_audio": 0},
            asr_validator=lambda *_: None,
            translation_validator=lambda *_: None,
            gpu_initializer=lambda: None,
        )

        async def _run_100_rounds():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                up = await client.post("/api/media", files={"file": ("clip.mp4", b"vid", "video/mp4")})
                assert up.status_code == 201
                media_id = up.json()["id"]

                for i in range(100):
                    should_fail = (i % 2 == 1)
                    job = mgr.submit(
                        {"should_fail": should_fail},
                        media_id=f"{media_id}-{i}",
                        job_type="asr",
                    )
                    if i % 3 == 1:
                        await asyncio.sleep(0.001)
                    elif i % 3 == 2:
                        await asyncio.sleep(0.002)

                    chunks: list[str] = []
                    async with client.stream("GET", f"/api/jobs/{job.id}/stream") as resp:
                        assert resp.status_code == 200
                        async for chunk in resp.aiter_text():
                            chunks.append(chunk)
                            if "event: done" in chunk:
                                break
                    body = "".join(chunks)
                    expected_status = "failed" if should_fail else "succeeded"

                    # Parse SSE frames
                    frames = [f.strip() for f in body.strip().split("\n\n") if f.strip()]
                    msg_frames = []
                    done_frames = []
                    ids = []
                    for frame in frames:
                        ev_type = None
                        ev_id = None
                        ev_data = None
                        for line in frame.splitlines():
                            if line.startswith("id: "):
                                ev_id = int(line[4:])
                            elif line.startswith("event: "):
                                ev_type = line[7:]
                            elif line.startswith("data: "):
                                ev_data = json.loads(line[6:])
                        if ev_id is not None:
                            ids.append(ev_id)
                        if ev_type == "message" and ev_data is not None:
                            msg_frames.append((ev_id, ev_data))
                        elif ev_type == "done" and ev_data is not None:
                            done_frames.append((ev_id, ev_data))

                    assert ids == sorted(set(ids)), f"Round {i}: non-monotonic or duplicate SSE ids {ids}"
                    assert any(d.get("status") == expected_status for _, d in msg_frames), (
                        f"Round {i}: missed terminal event: message in {body}"
                    )
                    assert len(done_frames) == 1, f"Round {i}: missing event: done in {body}"
                    done_id, done_data = done_frames[0]
                    assert done_data["status"] == expected_status
                    assert done_data["terminal"] is True
                    last_msg_seq = max(seq for seq, _ in msg_frames)
                    assert done_data["seq"] == last_msg_seq
                    assert done_id == last_msg_seq + 1
                    if should_fail:
                        assert done_data.get("error") == "Simulated ASR failure"
                        assert done_data.get("message")

        asyncio.run(_run_100_rounds())
    finally:
        set_db_path(orig_db)


def test_cancel_terminal_monotonicity_all_4_orderings(tmp_path):
    orig_db = get_db_path()
    set_db_path(tmp_path / "cancel_mono.db")
    try:
        init_db()

        def _terminal_events(job_id: str) -> list[dict]:
            out = []
            for ev in events_since(job_id, after_seq=0, limit=500):
                p = json.loads(ev["payload"])
                if p.get("status") in {"succeeded", "failed", "cancelled"}:
                    out.append(p)
            return out

        def _wait_done(job):
            deadline = time.monotonic() + 5.0
            while job.snapshot()["status"] not in {"succeeded", "failed", "cancelled"} and time.monotonic() < deadline:
                time.sleep(0.005)

        # Case 1: cancel() before runner returns CANCELLED
        proj1 = create_project(name="Case 1")
        started1 = threading.Event()
        release1 = threading.Event()

        def runner1(_req, _sink, _token):
            started1.set()
            release1.wait(timeout=2.0)
            return TaskResult("j1", TaskStatus.CANCELLED, tmp_path)

        mgr1 = JobManager(runner=runner1, asr_runner=runner1)
        j1 = mgr1.submit({}, media_id="m1", job_type="asr", project_id=proj1["id"])
        assert started1.wait(timeout=2.0)
        mgr1.cancel(j1.id)
        release1.set()
        _wait_done(j1)
        assert j1.status == "cancelled"
        assert get_job(j1.id)["status"] == "cancelled"
        assert get_project(proj1["id"])["status"] == "paused"
        assert len(_terminal_events(j1.id)) == 1

        # Case 2: runner emits EventKind.CANCELLED & finishes CANCELLED, then cancel() is called
        proj2 = create_project(name="Case 2")

        def runner2(_req, sink, _token):
            sink(TaskEvent("j2", EventKind.CANCELLED, "recogn", "Cancelled inside runner"))
            return TaskResult("j2", TaskStatus.CANCELLED, tmp_path)

        mgr2 = JobManager(runner=runner2, asr_runner=runner2)
        j2 = mgr2.submit({}, media_id="m2", job_type="asr", project_id=proj2["id"])
        _wait_done(j2)
        mgr2.cancel(j2.id)
        assert j2.status == "cancelled"
        assert get_job(j2.id)["status"] == "cancelled"
        assert get_project(proj2["id"])["status"] == "paused"
        assert len(_terminal_events(j2.id)) == 1

        # Case 3: cancel() mid-run -> runner throws exception -> must stay cancelled, NOT failed
        proj3 = create_project(name="Case 3")
        started3 = threading.Event()
        release3 = threading.Event()

        def runner3(_req, _sink, _token):
            started3.set()
            release3.wait(timeout=2.0)
            raise RuntimeError("subprocess killed")

        mgr3 = JobManager(runner=runner3, asr_runner=runner3)
        j3 = mgr3.submit({}, media_id="m3", job_type="asr", project_id=proj3["id"])
        assert started3.wait(timeout=2.0)
        mgr3.cancel(j3.id)
        release3.set()
        time.sleep(0.05)
        assert j3.status == "cancelled"
        assert get_job(j3.id)["status"] == "cancelled"
        assert get_project(proj3["id"])["status"] == "paused"
        assert len(_terminal_events(j3.id)) == 1

        # Case 4: SUCCEEDED completes -> late cancel() must be a no-op
        proj4 = create_project(name="Case 4")

        def runner4(_req, sink, _token):
            sink(TaskEvent("j4", EventKind.SUCCEEDED, "task_done", "Done"))
            return TaskResult("j4", TaskStatus.SUCCEEDED, tmp_path)

        mgr4 = JobManager(runner=runner4, asr_runner=runner4)
        j4 = mgr4.submit({}, media_id="m4", job_type="asr", project_id=proj4["id"])
        _wait_done(j4)
        assert j4.status == "succeeded"
        assert get_project(proj4["id"])["status"] == "completed"
        mgr4.cancel(j4.id)
        assert j4.status == "succeeded"
        assert get_job(j4.id)["status"] == "succeeded"
        assert get_project(proj4["id"])["status"] == "completed"
        assert len(_terminal_events(j4.id)) == 1
    finally:
        set_db_path(orig_db)


def test_post_restart_job_manager_get_rehydrates_outputs_segments_and_media_id(tmp_path):
    orig_db = get_db_path()
    set_db_path(tmp_path / "restart_rehydrate.db")
    try:
        init_db()
        out_mp4 = tmp_path / "persisted.mp4"
        out_mp4.write_bytes(b"persisted-video")

        def ok_runner(_req, sink, _token):
            sink(TaskEvent("j", EventKind.LOG, "recogn", "timing", details={"source_type": "asr_timing", "duration": 2.75}))
            sink(
                TaskEvent(
                    "j",
                    EventKind.SUCCEEDED,
                    "task_done",
                    "Done",
                    details={
                        "outputs": [str(out_mp4)],
                        "segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "hi"}],
                        "transcript_options": {"utterances": [{"id": 1, "text": "hi"}]},
                        "asr_duration": 2.75,
                    },
                )
            )
            return TaskResult(
                "j",
                TaskStatus.SUCCEEDED,
                tmp_path,
                outputs=(out_mp4,),
                segments=({"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "hi"},),
                asr_duration=2.75,
                transcript_options={"utterances": [{"id": 1, "text": "hi"}]},
            )

        mgr1 = JobManager(runner=ok_runner)
        job = mgr1.submit({}, media_id="media-restart-1", job_type="render")
        deadline = time.monotonic() + 5.0
        while job.snapshot()["status"] != "succeeded" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.snapshot()["status"] == "succeeded"

        # Simulate server restart with a brand new JobManager
        mgr2 = JobManager(runner=ok_runner)
        rehydrated = mgr2.get(job.id)
        assert rehydrated is not None
        assert rehydrated.media_id == "media-restart-1"
        assert rehydrated.outputs == (out_mp4,)
        assert rehydrated.segments == ({"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "hi"},)
        assert rehydrated.transcript_options == {"utterances": [{"id": 1, "text": "hi"}]}
        assert rehydrated.asr_duration == 2.75
        assert len(rehydrated.events) >= 2

        app = create_app(
            upload_dir=tmp_path / "uploads",
            job_manager=mgr2,
            media_probe=lambda _p: {"time": 1000, "width": 320, "height": 240, "video_streams": 1, "streams_audio": 0},
            asr_validator=lambda *_: None,
            translation_validator=lambda *_: None,
            gpu_initializer=lambda: None,
        )

        async def _verify_routes():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                r_job = await client.get(f"/api/jobs/{job.id}")
                assert r_job.status_code == 200
                assert r_job.json()["outputs"][0]["name"] == "persisted.mp4"
                assert r_job.json()["asrDuration"] == 2.75

                r_tr = await client.get(f"/api/jobs/{job.id}/transcript")
                assert r_tr.status_code == 200
                assert r_tr.json()["segments"] == [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "hi"}]
                assert r_tr.json()["transcriptOptions"] == {"utterances": [{"id": 1, "text": "hi"}]}

                r_out = await client.get(f"/api/jobs/{job.id}/outputs/0")
                assert r_out.status_code == 200
                assert r_out.content == b"persisted-video"

        asyncio.run(_verify_routes())
    finally:
        set_db_path(orig_db)


# ---------------------------------------------------------------------------
# Fix 9 (M7 Phase 1 Iter 2): 16-thread atomic update_project_state
# ---------------------------------------------------------------------------


def test_update_project_state_16_thread_concurrent_merge_preserves_all_keys(tmp_path):
    orig_db = get_db_path()
    set_db_path(tmp_path / "atomic_proj_state.db")
    try:
        init_db()
        proj = create_project(name="Concurrent State", state={"initial": "keep"})
        pid = proj["id"]
        barrier = threading.Barrier(16)
        errors: list[BaseException] = []

        def _worker(idx: int):
            try:
                barrier.wait(timeout=5.0)
                res = update_project_state(pid, {f"key_{idx}": idx}, merge=True)
                assert res is not None
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Concurrent update_project_state errors: {errors!r}"
        final_proj = get_project(pid)
        assert final_proj is not None
        assert final_proj["state"]["initial"] == "keep"
        for i in range(16):
            assert final_proj["state"].get(f"key_{i}") == i, f"Lost key_{i} in {final_proj['state']}"
    finally:
        set_db_path(orig_db)


# ---------------------------------------------------------------------------
# Fix 10 (M7 Phase 1 Iter 3): _subprocess / _novoice / assembling cancellation
# ---------------------------------------------------------------------------


def _count_matching_ffmpeg_procs(marker: str, cwd_marker: str | None = None) -> int:
    import psutil

    norm_marker = marker.replace("\\", "/")
    norm_cwd = cwd_marker.replace("\\", "/") if cwd_marker is not None else None
    count = 0
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if "ffmpeg" not in name:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or []).replace("\\", "/")
            if norm_marker not in cmdline:
                continue
            if norm_cwd is not None:
                proc_cwd = (proc.cwd() or "").replace("\\", "/")
                if norm_cwd not in proc_cwd:
                    continue
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return count


@pytest.mark.parametrize("codec", ["libx264", "h264_nvenc"])
def test_novoice_cancel_force_lib_false_libx264_and_hw_paths(tmp_path, monkeypatch, codec):
    from videotrans.configure.config import settings
    from videotrans.core.proc_registry import has_active_procs, kill_job_procs
    from videotrans.task import _stage_assemble

    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    src_video = tmp_path / f"heavy_mpeg4_{codec}.mp4"
    runffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=30:d=28.0",
        "-c:v", "mpeg4", "-q:v", "5", "-pix_fmt", "yuv420p", src_video.as_posix(),
    ], force_cpu=True)

    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))
    monkeypatch.setitem(settings, "force_lib", False)
    monkeypatch.setattr(app_cfg, "video_codec", codec)
    if codec == "h264_nvenc":
        monkeypatch.setattr(
            _stage_assemble.AssembleMixin,
            "_get_hard_cfg",
            lambda self, subtitles_file=None, codec=None: ([], "", "libx264", []),
        )

    orig_prepare_runffmpeg = _stage_prepare.runffmpeg
    prepare_runffmpeg_calls: list[list[str]] = []

    def _spy_prepare_runffmpeg(arg, *args, **kwargs):
        prepare_runffmpeg_calls.append(list(arg))
        return orig_prepare_runffmpeg(arg, *args, **kwargs)

    monkeypatch.setattr(_stage_prepare, "runffmpeg", _spy_prepare_runffmpeg)

    job_id = f"novoice-cancel-{codec}-{time.monotonic_ns()}"
    token = CancellationToken()
    assembling_started = threading.Event()
    result_box: list[TaskResult] = []
    errors: list[BaseException] = []

    def _sink(event: TaskEvent):
        if event.kind == EventKind.STAGE_STARTED and event.stage == "assembling":
            assembling_started.set()

    req = TaskRequest({
        "name": str(src_video),
        "target_dir": str(out_dir),
        "source_language_code": "en",
        "target_language_code": "en",
        "voice_role": "No",
        "subtitle_type": 2,
        "subtitles": "1\n00:00:00,100 --> 00:00:01,000\nCancel novoice\n",
    })

    def _worker():
        set_current_job_id(job_id)
        try:
            result_box.append(orchestrator.run(req, event_sink=_sink, cancellation_token=token))
        except BaseException as exc:
            errors.append(exc)
        finally:
            set_current_job_id(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        assert assembling_started.wait(timeout=10.0), "assembling stage never started"
        deadline = time.monotonic() + 5.0
        while not has_active_procs(job_id) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert has_active_procs(job_id) is True, f"Expected active proc for {codec}"
        assert _count_matching_ffmpeg_procs(src_video.as_posix()) >= 1

        token.cancel()
        kill_job_procs(job_id)
        t.join(timeout=10.0)
        assert not t.is_alive()
        assert not errors, f"Worker raised: {errors!r}"
        assert len(result_box) == 1
        res = result_box[0]
        assert res.status == TaskStatus.CANCELLED
        assert not has_active_procs(job_id)
        assert _count_matching_ffmpeg_procs(src_video.as_posix()) == 0
        assert app_cfg.queue_novice.get(res.job_id) != "ing"
        assert app_cfg.queue_novice.get(res.job_id) == "cancelled"
        if codec == "h264_nvenc":
            assert prepare_runffmpeg_calls == []
    finally:
        token.cancel()
        kill_job_procs(job_id)
        t.join(timeout=5.0)
        if result_box:
            app_cfg.queue_novice.pop(result_box[0].job_id, None)


@pytest.mark.parametrize("subtitle_type,video_codec_num", [(2, 265), (1, 264)])
def test_assembling_subprocess_cancellation_no_cpu_fallback(tmp_path, monkeypatch, subtitle_type, video_codec_num):
    from videotrans.configure.config import settings
    from videotrans.core.proc_registry import has_active_procs, kill_job_procs
    from videotrans.task import _stage_assemble

    cache_dir = tmp_path / "cache"
    target_dir = tmp_path / "out"
    cache_dir.mkdir()
    target_dir.mkdir()

    novoice_mp4 = cache_dir / "novoice.mp4"
    runffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=15.0",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", novoice_mp4.as_posix(),
    ], force_cpu=True)

    srt_path = target_dir / "target.srt"
    srt_path.write_text("1\n00:00:00,100 --> 00:00:02,000\nHardware cancel\n", encoding="utf-8")
    (cache_dir / "end.srt").write_text(srt_path.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setitem(settings, "force_lib", False)
    monkeypatch.setattr(app_cfg, "video_codec", "h264_nvenc")
    monkeypatch.setattr(
        _stage_assemble.AssembleMixin,
        "_get_hard_cfg",
        lambda self, subtitles_file=None, codec=None: (
            [],
            f"[0:v]subtitles=filename='{subtitles_file}'[v_out]" if subtitles_file else "",
            "libx264",
            ["-preset", "slow"],
        ),
    )

    fallback_calls: list[list[str]] = []
    orig_assemble_runffmpeg = _stage_assemble.runffmpeg

    def _spy_runffmpeg(arg, *args, **kwargs):
        fallback_calls.append(list(arg))
        return orig_assemble_runffmpeg(arg, *args, **kwargs)

    monkeypatch.setattr(_stage_assemble, "runffmpeg", _spy_runffmpeg)

    token = CancellationToken()
    job_id = f"assemble-subproc-{subtitle_type}-{time.monotonic_ns()}"

    class _HwCancelTask(AssembleMixin):
        def __init__(self):
            self.uuid = f"hw-cancel-{subtitle_type}"
            self.hasend = False
            self.precent = 90
            self.cost_duration = time.time()
            self.should_hebing = True
            self.should_dubbing = False
            self.is_audio_trans = False
            self.is_copy_video = False
            self.video_codec_num = str(video_codec_num)
            self.queue_tts = []
            self.cancellation_token = token
            self.event_sink = None
            self.video_info = {"time": 15000, "streams_audio": 0, "video_streams": 1, "video_fps": 30}
            self.cfg = SimpleNamespace(
                name=novoice_mp4.as_posix(),
                cache_folder=cache_dir.as_posix(),
                target_dir=target_dir.as_posix(),
                novoice_mp4=novoice_mp4.as_posix(),
                target_wav=(cache_dir / "target.wav").as_posix(),
                source_wav_output=(target_dir / "source.m4a").as_posix(),
                target_wav_output=(target_dir / "target.m4a").as_posix(),
                targetdir_mp4=(target_dir / "final.mp4").as_posix(),
                source_sub=srt_path.as_posix(),
                target_sub=srt_path.as_posix(),
                source_language_code="en",
                target_language_code="en",
                subtitle_type=subtitle_type,
                video_autorate=False,
                app_mode="biaozhun",
                only_out_mp4=False,
                only_out_dubbed_audio=False,
                subtitle_style=None,
            )

        def _exit(self):
            return self.cancellation_token.is_cancelled()

        def signal(self, **_kwargs):
            return None

        def _process_subtitles(self):
            return "end.srt", "eng"

    task = _HwCancelTask()
    errors: list[BaseException] = []

    def _worker():
        set_current_job_id(job_id)
        try:
            task.assembling()
        except BaseException as exc:
            errors.append(exc)
        finally:
            task.hasend = True
            set_current_job_id(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 8.0
        while not has_active_procs(job_id) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert has_active_procs(job_id) is True

        token.cancel()
        kill_job_procs(job_id)
        t.join(timeout=10.0)
        assert not t.is_alive()
        assert not errors, f"assembling raised: {errors!r}"
        assert not has_active_procs(job_id)
        assert _count_matching_ffmpeg_procs("laste_target", cwd_marker=cache_dir.as_posix()) == 0
        assert fallback_calls == []
        assert not Path(task.cfg.targetdir_mp4).exists()
    finally:
        token.cancel()
        kill_job_procs(job_id)
        t.join(timeout=5.0)


def test_subprocess_hw_error_falls_back_when_not_cancelled(tmp_path, monkeypatch):
    from videotrans.configure.config import settings
    from videotrans.configure.excepts import FFmpegError
    from videotrans.task import _stage_assemble

    cache_dir = tmp_path / "cache"
    target_dir = tmp_path / "out"
    cache_dir.mkdir()
    target_dir.mkdir()

    src_video = tmp_path / "src_mpeg4.mp4"
    runffmpeg([
        "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10:d=1.0",
        "-c:v", "mpeg4", "-pix_fmt", "yuv420p", src_video.as_posix(),
    ], force_cpu=True)

    monkeypatch.setitem(settings, "force_lib", False)
    monkeypatch.setattr(app_cfg, "video_codec", "h264_nvenc")

    class _FallbackTask(_stage_prepare.PrepareMixin, AssembleMixin):
        def __init__(self):
            self.uuid = f"hw-fallback-{time.monotonic_ns()}"
            self.hasend = False
            self.precent = 50
            self.cost_duration = time.time()
            self.should_hebing = True
            self.should_dubbing = False
            self.is_audio_trans = False
            self.is_copy_video = False
            self.video_codec_num = "265"
            self.queue_tts = []
            self.cancellation_token = CancellationToken()
            self.event_sink = None
            self.video_info = {"time": 1000, "streams_audio": 0, "video_streams": 1, "video_fps": 10}
            self.cfg = SimpleNamespace(
                name=src_video.as_posix(),
                cache_folder=cache_dir.as_posix(),
                target_dir=target_dir.as_posix(),
                novoice_mp4=(cache_dir / "novoice.mp4").as_posix(),
                target_wav=(cache_dir / "target.wav").as_posix(),
                source_wav_output=(target_dir / "source.m4a").as_posix(),
                target_wav_output=(target_dir / "target.m4a").as_posix(),
                targetdir_mp4=(target_dir / "final.mp4").as_posix(),
                source_sub=(target_dir / "source.srt").as_posix(),
                target_sub=(target_dir / "target.srt").as_posix(),
                source_language_code="en",
                target_language_code="en",
                subtitle_type=0,
                video_autorate=False,
                app_mode="biaozhun",
                only_out_mp4=False,
                only_out_dubbed_audio=False,
                subtitle_style=None,
            )

        def _exit(self):
            return self.cancellation_token.is_cancelled()

        def signal(self, **_kwargs):
            return None

        def _get_hard_cfg(self, subtitles_file=None, codec=None):
            return [], "", "nonexistent_hw_codec_xyz", []

    task = _FallbackTask()
    with pytest.raises(FFmpegError, match=r"尝试使用硬件执行命令出错\[CalledProcessError\]:"):
        task._subprocess(["-y", "-i", src_video.as_posix(), "-c:v", "nonexistent_hw_codec_xyz", (cache_dir / "fail.mp4").as_posix()])

    app_cfg.queue_novice[task.uuid] = "ing"
    try:
        task._split_novoice_byraw()
        assert Path(task.cfg.novoice_mp4).is_file()
        assert app_cfg.queue_novice.get(task.uuid) == "end"

        task.assembling()
        task.hasend = True
        assert Path(task.cfg.targetdir_mp4).is_file()
    finally:
        app_cfg.queue_novice.pop(task.uuid, None)


def test_exit_soft_aligned_between_base_exit_and_is_novoice_mp4(tmp_path, monkeypatch):
    from videotrans.configure.base import BaseCon

    class _ExitAlignedTask(BaseCon, AssembleMixin):
        def __init__(self):
            super().__init__(cancellation_token=CancellationToken())
            self.uuid = "exit-soft-aligned"
            self.should_hebing = True
            self.should_dubbing = False
            self.queue_tts = []
            self.cfg = SimpleNamespace(
                novoice_mp4=(tmp_path / "missing_novoice.mp4").as_posix(),
                target_sub=(tmp_path / "target.srt").as_posix(),
            )

    monkeypatch.setattr(app_cfg, "exit_soft", True)
    task = _ExitAlignedTask()
    assert task._exit() is True
    assert help_misc.is_novoice_mp4(
        task.cfg.novoice_mp4,
        task.uuid,
        cancellation_token=task.cancellation_token,
    ) is False
    task._join_video_audio_srt()


def test_only_out_dubbed_audio_video_autorate_waits_for_novoice_before_export(tmp_path):
    cache_dir = tmp_path / "cache"
    target_dir = tmp_path / "out"
    cache_dir.mkdir()
    target_dir.mkdir()
    target_wav = cache_dir / "target.wav"
    target_wav.write_bytes(b"dubbed audio")
    novoice_mp4 = cache_dir / "novoice.mp4"

    def _delayed_novoice():
        time.sleep(2.3)
        novoice_mp4.write_bytes(b"slow silent video after 2.3s")

    class _SlowNovoiceExportTask(AssembleMixin):
        def __init__(self):
            self.uuid = "slow-novoice-export"
            self.hasend = False
            self.precent = 90
            self.cost_duration = time.time()
            self.is_audio_trans = False
            self.should_hebing = True
            self.done = False
            self.cfg = SimpleNamespace(
                name="input.mp4",
                app_mode="biaozhun",
                only_out_mp4=False,
                only_out_dubbed_audio=True,
                video_autorate=True,
                source_sub=target_dir / "source.srt",
                target_sub=target_dir / "target.srt",
                target_wav=target_wav,
                source_wav_output=target_dir / "source.m4a",
                target_wav_output=target_dir / "target.m4a",
                target_dir=target_dir,
                target_language_code="vi",
                targetdir_mp4=target_dir / "result.mp4",
                novoice_mp4=novoice_mp4,
            )
            self._novoice_thread = threading.Thread(target=_delayed_novoice, daemon=True)
            self._novoice_thread.start()

        def _exit(self):
            return False

        def set_end(self, success):
            self.done = success

        def signal(self, **_kwargs):
            return None

    task = _SlowNovoiceExportTask()
    task.task_done()
    assert task.done is True
    assert (target_dir / "vi-dubbing.wav").read_bytes() == b"dubbed audio"
    assert (target_dir / "vi-slowed.mp4").read_bytes() == b"slow silent video after 2.3s"


# ---------------------------------------------------------------------------
# Fix 11 (M7 Phase 1 Iter 3): Terminal monotonicity, project sync & >500 events
# ---------------------------------------------------------------------------


def _terminal_events_for_job(job_id: str) -> list[dict]:
    out = []
    for ev in events_since(job_id, after_seq=0, limit=1000):
        p = json.loads(ev["payload"])
        if p.get("status") in {"succeeded", "failed", "cancelled"}:
            out.append(p)
    return out


def _wait_terminal(job, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while job.snapshot()["status"] not in {"succeeded", "failed", "cancelled"} and time.monotonic() < deadline:
        time.sleep(0.005)


def test_case2_slow_runner_return_syncs_project_status_before_wait_done_returns(tmp_path):
    orig_db = get_db_path()
    set_db_path(tmp_path / "case2_slow.db")
    try:
        init_db()
        proj = create_project(name="Case 2 Slow")
        emitted_cancel = threading.Event()
        release_runner = threading.Event()

        def runner(_req, sink, _token):
            sink(TaskEvent("j", EventKind.CANCELLED, "recogn", "Cancelled inside runner"))
            emitted_cancel.set()
            release_runner.wait(timeout=5.0)
            return TaskResult("j", TaskStatus.CANCELLED, tmp_path)

        mgr = JobManager(runner=runner, asr_runner=runner)
        job = mgr.submit({}, media_id="m-case2-slow", job_type="asr", project_id=proj["id"])
        assert emitted_cancel.wait(timeout=2.0)
        _wait_terminal(job)

        # While runner is STILL blocked before returning TaskResult:
        assert job.status == "cancelled"
        assert get_job(job.id)["status"] == "cancelled"
        assert get_project(proj["id"])["status"] == "paused"

        mgr.cancel(job.id)
        assert job.status == "cancelled"
        assert get_project(proj["id"])["status"] == "paused"

        release_runner.set()
        deadline = time.monotonic() + 5.0
        while mgr._active_by_media and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(_terminal_events_for_job(job.id)) == 1
    finally:
        set_db_path(orig_db)


def test_event_cancelled_then_runtime_error_preserves_cancelled_and_paused(tmp_path):
    orig_db = get_db_path()
    set_db_path(tmp_path / "cancel_then_err.db")
    try:
        init_db()
        proj = create_project(name="Cancel Then Error")

        def runner(_req, sink, _token):
            sink(TaskEvent("j", EventKind.CANCELLED, "recogn", "Cancelled before error"))
            raise RuntimeError("cleanup threw after EventKind.CANCELLED")

        mgr = JobManager(runner=runner, asr_runner=runner)
        job = mgr.submit({}, media_id="m-cancel-err", job_type="asr", project_id=proj["id"])
        deadline = time.monotonic() + 5.0
        while mgr._active_by_media and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not mgr._active_by_media
        assert job.status == "cancelled"
        assert get_job(job.id)["status"] == "cancelled"
        assert get_project(proj["id"])["status"] == "paused"
        assert len(_terminal_events_for_job(job.id)) == 1
    finally:
        set_db_path(orig_db)


@pytest.mark.parametrize("late_outcome", ["none", "succeeded"])
def test_event_cancelled_then_none_or_succeeded_preserves_cancelled_and_paused(tmp_path, late_outcome):
    orig_db = get_db_path()
    set_db_path(tmp_path / f"cancel_then_{late_outcome}.db")
    try:
        init_db()
        proj = create_project(name=f"Cancel Then {late_outcome}")

        def runner(_req, sink, _token):
            sink(TaskEvent("j", EventKind.CANCELLED, "recogn", "Cancelled early"))
            if late_outcome == "none":
                return None
            return TaskResult("j", TaskStatus.SUCCEEDED, tmp_path)

        mgr = JobManager(runner=runner, asr_runner=runner)
        job = mgr.submit({}, media_id=f"m-{late_outcome}", job_type="asr", project_id=proj["id"])
        deadline = time.monotonic() + 5.0
        while mgr._active_by_media and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not mgr._active_by_media
        assert job.status == "cancelled"
        assert get_job(job.id)["status"] == "cancelled"
        assert get_project(proj["id"])["status"] == "paused"
        assert len(_terminal_events_for_job(job.id)) == 1
    finally:
        set_db_path(orig_db)


@pytest.mark.parametrize("late_status", [TaskStatus.SUCCEEDED, TaskStatus.CANCELLED])
def test_event_failed_then_succeeded_or_cancelled_preserves_failed_status(tmp_path, late_status):
    orig_db = get_db_path()
    set_db_path(tmp_path / f"fail_then_{late_status.value}.db")
    try:
        init_db()
        proj = create_project(name=f"Fail Then {late_status.value}")

        def runner(_req, sink, _token):
            sink(TaskEvent("j", EventKind.FAILED, "recogn", "Primary failure"))
            return TaskResult("j", late_status, tmp_path)

        mgr = JobManager(runner=runner, asr_runner=runner)
        job = mgr.submit({}, media_id=f"m-fail-{late_status.value}", job_type="asr", project_id=proj["id"])
        deadline = time.monotonic() + 5.0
        while mgr._active_by_media and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not mgr._active_by_media
        assert job.status == "failed"
        assert get_job(job.id)["status"] == "failed"
        assert get_project(proj["id"])["status"] == "failed"
        assert len(_terminal_events_for_job(job.id)) == 1
    finally:
        set_db_path(orig_db)


def test_immediate_cancel_before_mark_running_never_reverts_to_running_or_processing(tmp_path, monkeypatch):
    from videotrans.core import job_executor

    orig_db = get_db_path()
    set_db_path(tmp_path / "immediate_cancel.db")
    try:
        init_db()
        proj = create_project(name="Immediate Cancel")

        entered_execute = threading.Event()
        release_execute = threading.Event()
        orig_execute_job = job_executor.execute_job

        def _barrier_execute_job(job, params, runners):
            entered_execute.set()
            release_execute.wait(timeout=5.0)
            return orig_execute_job(job, params, runners)

        monkeypatch.setattr("videotrans.core.job_manager.execute_job", _barrier_execute_job)

        runner_called = []

        def runner(_req, _sink, _token):
            runner_called.append(True)
            return TaskResult("j", TaskStatus.SUCCEEDED, tmp_path)

        mgr = JobManager(runner=runner, asr_runner=runner)
        job = mgr.submit({}, media_id="m-immediate-cancel", job_type="asr", project_id=proj["id"])
        assert entered_execute.wait(timeout=2.0)

        mgr.cancel(job.id)
        release_execute.set()

        deadline = time.monotonic() + 5.0
        while mgr._active_by_media and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not mgr._active_by_media
        assert runner_called == []
        assert job.status == "cancelled"
        assert get_job(job.id)["status"] == "cancelled"
        assert get_project(proj["id"])["status"] == "paused"
        assert len(_terminal_events_for_job(job.id)) == 1
    finally:
        set_db_path(orig_db)


@pytest.mark.parametrize("emit_succeeded_event", [True, False])
def test_rehydration_over_500_events_preserves_all_metadata(tmp_path, emit_succeeded_event):
    from videotrans.core.db import db_conn

    orig_db = get_db_path()
    set_db_path(tmp_path / f"rehydrate_500_{emit_succeeded_event}.db")
    submitted = threading.Event()
    job_id_box: list[str] = []
    mgr1 = None
    try:
        init_db()
        out_mp4 = tmp_path / "out_500.mp4"
        out_mp4.write_bytes(b"video-500")
        segs = ({"id": 1, "startSec": 0.0, "endSec": 1.2, "sourceText": "Line 1"},)
        opts = {"utterances": [{"id": 1, "text": "Line 1"}]}

        def runner(_req, sink, _token):
            assert submitted.wait(timeout=5.0), "job_id was not populated in time"
            real_job_id = job_id_box[0]
            sink(
                TaskEvent(
                    real_job_id,
                    EventKind.LOG,
                    "recogn",
                    "timing",
                    details={"source_type": "asr_timing", "duration": 6.25},
                )
            )
            now = time.time()
            with db_conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS s FROM job_events WHERE job_id = ?",
                    (real_job_id,),
                ).fetchone()
                base_seq = int(row["s"])
                rows = [
                    (
                        real_job_id,
                        base_seq + i + 1,
                        now,
                        json.dumps({
                            "kind": EventKind.PROGRESS.value,
                            "stage": "recogn",
                            "message": f"seeded step {i}",
                            "progress": float(i % 100),
                            "status": "running",
                            "error": None,
                            "details": {},
                        }),
                    )
                    for i in range(505)
                ]
                conn.executemany(
                    "INSERT INTO job_events (job_id, seq, created_at, payload) VALUES (?, ?, ?, ?)",
                    rows,
                )
            for i in range(5):
                sink(
                    TaskEvent(
                        real_job_id,
                        EventKind.PROGRESS,
                        "recogn",
                        f"live step {i}",
                        progress=float(90 + i),
                    )
                )
            if emit_succeeded_event:
                sink(
                    TaskEvent(
                        real_job_id,
                        EventKind.SUCCEEDED,
                        "task_done",
                        "Complete",
                        details={
                            "outputs": [str(out_mp4)],
                            "segments": list(segs),
                            "transcript_options": opts,
                            "asr_duration": 6.25,
                        },
                    )
                )
            return TaskResult(
                real_job_id,
                TaskStatus.SUCCEEDED,
                tmp_path,
                outputs=(out_mp4,),
                segments=segs,
                asr_duration=6.25,
                transcript_options=opts,
            )

        mgr1 = JobManager(runner=runner)
        job = mgr1.submit({}, media_id=f"m-500-{emit_succeeded_event}", job_type="full")
        job_id_box.append(job.id)
        submitted.set()

        _wait_terminal(job, timeout=15.0)
        deadline = time.monotonic() + 15.0
        while mgr1._active_by_media and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not mgr1._active_by_media
        assert job.status == "succeeded"

        persisted = events_since(job.id, after_seq=0, limit=1000)
        assert len(persisted) == 500
        assert persisted[0]["seq"] > 1
        assert all(
            json.loads(ev["payload"]).get("details", {}).get("source_type") != "asr_timing"
            for ev in persisted
        )

        mgr2 = JobManager(runner=runner)
        rehydrated = mgr2.get(job.id)
        assert rehydrated is not None
        assert rehydrated.status == "succeeded"
        assert rehydrated.outputs == (out_mp4,)
        assert rehydrated.segments == segs
        assert rehydrated.transcript_options == opts
        assert rehydrated.asr_duration == 6.25
    finally:
        submitted.set()
        if mgr1 is not None:
            cleanup_deadline = time.monotonic() + 10.0
            while mgr1._active_by_media and time.monotonic() < cleanup_deadline:
                time.sleep(0.01)
        set_db_path(orig_db)


def test_snapshot_non_blocking_during_slow_sqlite_write(tmp_path, monkeypatch):
    from videotrans.core import job_manager

    orig_db = get_db_path()
    set_db_path(tmp_path / "nonblocking_snap.db")
    try:
        init_db()
        proj = create_project(name="Slow SQLite Snap")
        job = job_manager.JobRecord(
            id="job-slow-sqlite",
            token=CancellationToken(),
            status="running",
            project_id=proj["id"],
            job_type="asr",
        )
        create_job(job.id, type="asr", project_id=proj["id"])

        entered_sync = threading.Event()
        release_sync = threading.Event()
        orig_sync = job_manager.sync_project_for_terminal_status

        def _slow_sync(project_id, status, job_type="full"):
            entered_sync.set()
            release_sync.wait(timeout=5.0)
            return orig_sync(project_id, status, job_type)

        monkeypatch.setattr(job_manager, "sync_project_for_terminal_status", _slow_sync)

        t = threading.Thread(
            target=lambda: job.accept(TaskEvent(job.id, EventKind.SUCCEEDED, "task_done", "Done")),
            daemon=True,
        )
        t.start()
        assert entered_sync.wait(timeout=2.0)

        t0 = time.monotonic()
        snap_during = job.snapshot()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.2, f"snapshot() blocked on SQLite persistence ({elapsed:.3f}s)"
        assert snap_during["status"] == "running"

        release_sync.set()
        t.join(timeout=5.0)
        assert not t.is_alive()
        assert job.snapshot()["status"] == "succeeded"
        assert get_project(proj["id"])["status"] == "completed"
    finally:
        release_sync.set()
        set_db_path(orig_db)


# ---------------------------------------------------------------------------
# Fix 13 (M7 Phase 1 Iter 4): run_staged_translation cancel + exception &
# JobManager.submit() post-lock update_project race
# ---------------------------------------------------------------------------


def test_run_staged_translation_cancel_with_exception_returns_cancelled(tmp_path, monkeypatch):
    from videotrans import translator

    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    monkeypatch.setattr(orchestrator, "TEMP_DIR", str(temp_root))
    dummy_mp4 = tmp_path / "trans_cancel.mp4"
    dummy_mp4.write_bytes(b"mp4")

    events: list[TaskEvent] = []
    token = CancellationToken()

    def _abort_run(*args, **kwargs):
        token.cancel()
        raise RuntimeError("aborted mid-translation")

    monkeypatch.setattr(translator, "run", _abort_run)

    req = TaskRequest({
        "name": dummy_mp4.as_posix(),
        "target_dir": str(tmp_path / "out"),
        "source_language_code": "en",
        "target_language_code": "es",
        "translate_type": 0,
        "segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Hello world"}],
    })

    res = orchestrator.run_staged_translation(req, event_sink=events.append, cancellation_token=token)
    assert res.status == TaskStatus.CANCELLED
    assert any(ev.kind == EventKind.CANCELLED for ev in events)
    assert not any(ev.kind == EventKind.FAILED for ev in events)


def test_job_manager_submit_vs_concurrent_cancel_before_update_project(tmp_path, monkeypatch):
    from videotrans.core import job_manager

    orig_db = get_db_path()
    set_db_path(tmp_path / "submit_cancel_race.db")
    try:
        init_db()
        orig_job_record = job_manager.JobRecord

        for idx in range(20):
            proj = create_project(name=f"Submit Cancel Race {idx}")
            pid = proj["id"]
            release_runner = threading.Event()
            cancel_ready = threading.Event()
            cancel_done = threading.Event()
            created_job_id: list[str] = []

            class _TxLockProxy:
                def __init__(self, real_lock: threading.Lock, job_id: str, submit_tid: int) -> None:
                    self._real = real_lock
                    self._job_id = job_id
                    self._submit_tid = submit_tid
                    self._tripped = False

                def __enter__(self):
                    if not self._tripped and threading.get_ident() == self._submit_tid:
                        self._tripped = True
                        created_job_id.append(self._job_id)
                        cancel_ready.set()
                        cancel_done.wait(timeout=5.0)
                    self._real.acquire()
                    return self

                def __exit__(self, exc_type, exc, tb):
                    self._real.release()

            def _wrapped_job_record(*args, **kwargs):
                rec = orig_job_record(*args, **kwargs)
                rec._tx_lock = _TxLockProxy(rec._tx_lock, rec.id, threading.get_ident())  # type: ignore[assignment]
                return rec

            monkeypatch.setattr(job_manager, "JobRecord", _wrapped_job_record)

            def _slow_runner(_req, _sink, _token):
                release_runner.wait(timeout=5.0)
                return TaskResult("j", TaskStatus.CANCELLED, tmp_path)

            mgr = JobManager(runner=_slow_runner)

            def _do_cancel():
                assert cancel_ready.wait(timeout=5.0)
                try:
                    mgr.cancel(created_job_id[0])
                finally:
                    cancel_done.set()

            t_cancel = threading.Thread(target=_do_cancel, daemon=True)
            t_cancel.start()
            try:
                job = mgr.submit({}, media_id=f"m-race-{idx}", job_type="full", project_id=pid)
                t_cancel.join(timeout=5.0)
                assert not t_cancel.is_alive()
            finally:
                cancel_ready.set()
                cancel_done.set()
                release_runner.set()

            deadline = time.monotonic() + 5.0
            while mgr._active_by_media and time.monotonic() < deadline:
                time.sleep(0.01)

            assert job.status == "cancelled"
            assert get_job(job.id)["status"] == "cancelled"
            assert get_project(pid)["status"] == "paused", f"Round {idx} overwrote paused -> processing"
    finally:
        set_db_path(orig_db)
