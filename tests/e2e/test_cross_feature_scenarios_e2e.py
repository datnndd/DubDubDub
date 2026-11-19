# -*- coding: utf-8 -*-
"""Tier 3 (Pairwise Cross-Feature >= 12) & Tier 4 (Real-World Multi-Stage E2E >= 5) Tests.

All tests execute 100% offline under HF_HUB_OFFLINE=1 using real PaddleOcrProvider,
real FFmpeg media processing, real FastAPI endpoints, and an in-process HTTP
OpenAI-compatible translation server.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime
import functools
import io
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any
import wave
import zipfile

from fastapi.testclient import TestClient
import pytest

from tests.e2e.media_synth import (
    LocalOpenAITranslationServer,
    build_segmented_video,
    isolate_runtime_env,
    make_paragraph_bgr,
    make_wav,
    normalized_text_similarity,
    require_ocr_env,
    restore_runtime_config,
)

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _check_env():
    require_ocr_env()


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last.get("status") in {"succeeded", "failed", "cancelled"}:
            return last
        time.sleep(0.15)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s; last={last}")


def _install_fake_elevenlabs_wav(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, sample_rate: int = 24000, delay_sec: float = 0.0):
    """Patch external ElevenLabs network SDK to emit real 24kHz WAV audio bytes offline."""
    import elevenlabs

    wav_file = make_wav(tmp_path / f"el_synth_{sample_rate}.wav", duration_sec=0.8, sample_rate=sample_rate)
    wav_bytes = wav_file.read_bytes()

    class _FakeTTS:
        def convert(self, **_kwargs):
            if delay_sec > 0:
                time.sleep(delay_sec)
            return [wav_bytes]

    class _FakeElevenLabs:
        def __init__(self, **_kwargs):
            self.text_to_speech = _FakeTTS()

    monkeypatch.setattr(elevenlabs, "ElevenLabs", _FakeElevenLabs)
    try:
        from videotrans.tts import _elevenlabs as mod_el
        monkeypatch.setattr(mod_el, "ElevenLabs", _FakeElevenLabs)
    except Exception:
        pass


# ===========================================================================
# Tier 3: Pairwise Cross-Feature Combinations (>= 12)
# ===========================================================================


def test_t3_pair01_batch_date_parent_and_video_ocr_subproject(tmp_path, monkeypatch):
    """F1 (Batch Date-Parent) x F12 (Video OCR subproject job + state merge)."""
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.core.job_manager import JobManager, run_prepare_review
    from videotrans.core.project_store import get_project

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        folder = tmp_path / "batch_in"
        folder.mkdir()
        roi = [0.05, 0.74, 0.90, 0.22]
        vid1 = build_segmented_video(folder / "clip_a.mp4", [(2.5, ["Batch video A"], "en")], with_audio=False, roi_box=tuple(roi))
        vid2 = build_segmented_video(folder / "clip_b.mp4", [(2.5, ["Batch video B"], "en")], with_audio=False, roi_box=tuple(roi))

        app = create_app(
            job_manager=JobManager(runner=run_prepare_review),
            upload_dir=iso["uploads_dir"],
            task_params_builder=functools.partial(
                build_task_params,
                output_dir=iso["output_dir"],
                temp_dir=iso["temp_dir"],
            ),
        )
        with TestClient(app) as client:
            scan = client.post("/api/media/scan-folders", json={"paths": [str(folder)]})
            assert scan.status_code == 200 and scan.json()["count"] == 2

            reg = client.post("/api/media/register-local", json={"paths": [str(vid1), str(vid2)]})
            assert reg.status_code == 201
            media_items = reg.json()["media"]

            today_local = datetime.date.today().isoformat()
            batch = client.post(
                "/api/projects/batch-create",
                json={
                    "date": today_local,
                    "items": [
                        {"name": m["filename"], "mediaId": m["id"]}
                        for m in media_items
                    ],
                },
            )
            assert batch.status_code == 201, batch.text
            parent = batch.json()["parent"]
            subs = batch.json()["projects"]
            assert parent["id"] == f"parent_{today_local}"
            assert len(subs) == 2

            # Run Video OCR on first subproject
            r_job = client.post(
                "/api/jobs",
                json={
                    "mediaId": subs[0]["media_id"],
                    "projectId": subs[0]["id"],
                    "jobType": "asr",
                    "options": {
                        "subtitleSource": "video_ocr",
                        "ocrRoi": roi,
                        "ocrRoiConfirmed": True,
                        "sourceLanguage": "en",
                        "targetLanguage": "zh-cn",
                    },
                },
            )
            assert r_job.status_code == 202, r_job.text
            done = _wait_for_job(client, r_job.json()["id"])
            assert done["status"] == "succeeded", done

            # Partial state update must preserve parent_id
            put_res = client.put(
                f"/api/projects/{subs[0]['id']}/state",
                json={"stage": 2, "state": {"currentStep": 2, "segments": done["segments"]}},
            )
            assert put_res.status_code == 200
            persisted = get_project(subs[0]["id"])
            assert persisted is not None
            assert persisted["state"].get("parent_id") == parent["id"]
            assert "Batch video A" in persisted["state"]["segments"][0]["sourceText"]
    finally:
        restore_runtime_config(iso)


def test_t3_pair02_video_ocr_to_ai_translation_srt_cache_and_context(tmp_path, monkeypatch):
    """F11 (Video OCR SRT output) x F7 (AI Translation SRT cache hit & sliding context)."""
    from videotrans import translator
    from videotrans.configure import config as runtime_config
    from videotrans.ocr import scan_video_to_srt
    from videotrans.translator._chatgpt import ChatGPT

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        roi = (0.05, 0.74, 0.90, 0.22)
        vid = build_segmented_video(
            tmp_path / "ocr_trans_cache.mp4",
            [
                (2.5, ["Hello first cue"], "en"),
                (2.5, ["Hello second cue"], "en"),
            ],
            roi_box=roi,
        )
        srt_items, _ = scan_video_to_srt(str(vid), roi=roi, language="en", coarse_interval_ms=500)
        assert len(srt_items) == 2

        with LocalOpenAITranslationServer(prefix="[ZH]") as srv:
            runtime_config.params["chatgpt_key"] = "local-key"
            runtime_config.params["chatgpt_api"] = srv.base_url
            runtime_config.params["chatgpt_model"] = "gpt-4o-mini"
            runtime_config.settings["aitrans_context"] = True
            runtime_config.settings["aitrans_thread"] = 1

            text_list_1 = [
                {
                    "line": it["line"],
                    "time": it["time"],
                    "text": it["text"],
                    "start_time": it["start_time"],
                    "end_time": it["end_time"],
                    "startraw": it["startraw"],
                    "endraw": it["endraw"],
                }
                for it in srt_items
            ]
            t1 = ChatGPT(
                translate_type=translator.CHATGPT_INDEX,
                text_list=text_list_1,
                source_code="en",
                target_code="zh-cn",
                target_language_name="Simplified Chinese",
                aisendsrt=True,
            )
            out1 = t1.run()
            assert len(out1) == 2
            assert "[ZH]" in out1[0]["text"]
            req_count_after_first = len(srv.requests)
            assert req_count_after_first >= 1

            # Second identical run MUST hit SRT translation cache (0 new HTTP requests)
            text_list_2 = [dict(x) for x in text_list_1]
            t2 = ChatGPT(
                translate_type=translator.CHATGPT_INDEX,
                text_list=text_list_2,
                source_code="en",
                target_code="zh-cn",
                target_language_name="Simplified Chinese",
                aisendsrt=True,
            )
            out2 = t2.run()
            assert len(out2) == 2
            assert out2[0]["text"] == out1[0]["text"]
            assert len(srv.requests) == req_count_after_first, (
                f"Expected SRT cache hit with 0 extra HTTP requests, got {len(srv.requests)} vs {req_count_after_first}"
            )
    finally:
        restore_runtime_config(iso)


def test_t3_pair03_video_ocr_stage2_extract_repair_and_stage2_reset(tmp_path, monkeypatch):
    """F12 (Video OCR baseline) x F4 (Stage 2 /api/ocr/extract repair) x F2 (Stage 2 reset)."""
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.core.job_manager import JobManager, run_prepare_review

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        roi = [0.05, 0.72, 0.90, 0.24]
        vid = build_segmented_video(
            tmp_path / "repair_reset.mp4",
            [(3.0, ["Original OCR Line One", "Original OCR Line Two"], "en")],
            with_audio=False,
            roi_box=tuple(roi),
            font_size=30,
        )
        app = create_app(
            job_manager=JobManager(runner=run_prepare_review),
            upload_dir=iso["uploads_dir"],
            task_params_builder=functools.partial(
                build_task_params,
                output_dir=iso["output_dir"],
                temp_dir=iso["temp_dir"],
            ),
        )
        with TestClient(app) as client:
            with vid.open("rb") as f:
                media = client.post("/api/media", files={"file": ("repair_reset.mp4", f, "video/mp4")}).json()
            proj = client.post(
                "/api/projects",
                json={"name": "Repair Reset", "mediaId": media["id"]},
            ).json()

            r_job = client.post(
                "/api/jobs",
                json={
                    "mediaId": media["id"],
                    "projectId": proj["id"],
                    "jobType": "asr",
                    "options": {
                        "subtitleSource": "video_ocr",
                        "ocrRoi": roi,
                        "ocrRoiConfirmed": True,
                        "sourceLanguage": "en",
                    },
                },
            )
            assert r_job.status_code == 202, r_job.text
            done = _wait_for_job(client, r_job.json()["id"])
            assert done["status"] == "succeeded"
            orig_text = done["segments"][0]["sourceText"]

            # Stage 2 OCR extract
            ext = client.post(
                "/api/ocr/extract",
                json={"mediaId": media["id"], "startSec": 0.0, "endSec": 3.0, "roi": roi, "language": "en"},
            )
            assert ext.status_code == 200
            assert "\n" in ext.json()["text"]

            # Edit & save ocrCrop
            edited_segs = [dict(done["segments"][0], sourceText="Manually Overwritten Text")]
            client.put(
                f"/api/projects/{proj['id']}/state",
                json={"stage": 2, "state": {"segments": edited_segs, "ocrCrop": {"x": 0.1, "y": 0.7, "w": 0.8, "h": 0.2}}},
            )

            # Reset Stage 2 -> restores baseline and clears ocrCrop
            reset = client.post(f"/api/projects/{proj['id']}/stages/2/reset")
            assert reset.status_code == 200, reset.text
            reset_state = reset.json()["state"]
            assert reset_state["segments"][0]["sourceText"] == orig_text
            assert reset_state.get("ocrCrop") is None

            # Subsequent partial save must NOT resurrect ocrCrop
            after_save = client.put(
                f"/api/projects/{proj['id']}/state",
                json={"stage": 2, "state": {"currentStep": 2}},
            ).json()
            assert after_save["state"].get("ocrCrop") is None
    finally:
        restore_runtime_config(iso)


def test_t3_pair04_stage2_translation_voice_overrides_to_dubbing_assemble_24k_to_48k(tmp_path, monkeypatch):
    """F7/F8 (/api/translate) x F5/F9 (segmentVoiceOverrides + TTS preview + 24kHz->48kHz assembly)."""
    from videotrans import translator
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.configure import config as runtime_config
    from videotrans.core import secret_store

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    _install_fake_elevenlabs_wav(monkeypatch, tmp_path, sample_rate=24000)
    try:
        with LocalOpenAITranslationServer(prefix="[VI]") as srv:
            secret_store.set_secret("chatgpt_key", "local-key")
            runtime_config.params["chatgpt_key"] = "local-key"
            runtime_config.params["chatgpt_api"] = srv.base_url
            runtime_config.params["chatgpt_model"] = "gpt-4o-mini"
            runtime_config.params["elevenlabstts_key"] = "local-el-key"

            app = create_app(
                upload_dir=iso["uploads_dir"],
                task_params_builder=functools.partial(
                    build_task_params,
                    output_dir=iso["output_dir"],
                    temp_dir=iso["temp_dir"],
                ),
            )
            with TestClient(app) as client:
                proj = client.post("/api/projects", json={"name": "Assemble 24k", "duration": 5.0}).json()
                segs = [
                    {"id": 1, "startSec": 0.2, "endSec": 2.0, "sourceText": "Line one", "voiceOverride": "Rachel"},
                    {"id": 2, "startSec": 2.5, "endSec": 4.5, "sourceText": "Line two", "voiceOverride": "Domi"},
                ]
                tr = client.post(
                    "/api/translate",
                    json={
                        "projectId": proj["id"],
                        "segments": segs,
                        "sourceLanguage": "en",
                        "targetLanguage": "vi",
                        "translateType": translator.CHATGPT_INDEX,
                        "translationMode": "srt",
                    },
                )
                assert tr.status_code == 200, tr.text
                t_segs = tr.json()["segments"]
                assert all("[VI]" in s["targetText"] for s in t_segs)

                for s in t_segs:
                    pv = client.post(
                        "/api/tts/preview",
                        json={
                            "provider": "elevenlabs",
                            "voice": s.get("voiceOverride") or "Rachel",
                            "text": s["targetText"],
                            "language": "vi",
                            "segmentId": s["id"],
                        },
                    )
                    assert pv.status_code == 200, pv.text
                    s["previewAudioId"] = pv.json()["preview_id"]

                asm = client.post(f"/api/projects/{proj['id']}/dubbing/assemble", json={"segments": t_segs})
                assert asm.status_code == 200, asm.text
                audio_resp = client.get(asm.json()["audio_url"])
                assert audio_resp.status_code == 200
                with wave.open(io.BytesIO(audio_resp.content), "rb") as wf:
                    assert wf.getframerate() == 48000
                    assert wf.getnchannels() == 1
                    assert wf.getnframes() > 48000 * 3
    finally:
        restore_runtime_config(iso)


def test_t3_pair05_audio_asr_stage2_multiline_ocr_and_capcut_export(tmp_path, monkeypatch):
    """F6 (Stage 1 ASR baseline) x F4/F10 (Stage 2 multi-line OCR) x F5 (CapCut targetText export)."""
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        roi = [0.05, 0.72, 0.90, 0.24]
        expected_lines = ["CapCut line one", "CapCut line two"]
        vid = build_segmented_video(
            tmp_path / "capcut_ocr.mp4",
            [(2.5, expected_lines, "en")],
            with_audio=True,
            roi_box=tuple(roi),
        )
        app = create_app(
            upload_dir=iso["uploads_dir"],
            task_params_builder=functools.partial(
                build_task_params,
                output_dir=iso["output_dir"],
                temp_dir=iso["temp_dir"],
            ),
        )
        with TestClient(app) as client:
            with vid.open("rb") as f:
                media = client.post("/api/media", files={"file": ("capcut_ocr.mp4", f, "video/mp4")}).json()
            proj = client.post(
                "/api/projects",
                json={"name": "CapCut OCR", "mediaId": media["id"], "duration": 2.5},
            ).json()

            ext = client.post(
                "/api/ocr/extract",
                json={"mediaId": media["id"], "startSec": 0.0, "endSec": 2.5, "roi": roi, "language": "en"},
            )
            assert ext.status_code == 200, ext.text
            ocr_text = ext.json()["text"]
            assert "\n" in ocr_text

            segs = [
                {
                    "id": 1,
                    "startSec": 0.0,
                    "endSec": 2.5,
                    "sourceText": ocr_text,
                    "rawTranslation": "Dong mot\nDong hai",
                    "targetText": "Dong mot\nDong hai (edited)",
                }
            ]
            client.put(f"/api/projects/{proj['id']}/state", json={"stage": 4, "state": {"segments": segs}})

            zip_resp = client.get(f"/api/projects/{proj['id']}/export-capcut/bundle.zip")
            assert zip_resp.status_code == 200, zip_resp.text
            with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
                names = zf.namelist()
                target_name = next(n for n in names if n.endswith("subtitles_target.srt"))
                edited_name = next(n for n in names if n.endswith("subtitles_edited.srt"))
                target_srt = zf.read(target_name).decode("utf-8")
                edited_srt = zf.read(edited_name).decode("utf-8")
                assert "Dong mot" in target_srt
                assert "CapCut line one" not in target_srt
                assert "(edited)" in edited_srt
    finally:
        restore_runtime_config(iso)


def test_t3_pair06_video_ocr_is_separate_false_audioless_render(tmp_path, monkeypatch):
    """F11/F12 (audio-less video_ocr) x F6 (is_separate=False) x Stage 4 render."""
    from videotrans import translator
    from videotrans.configure import config as runtime_config
    from videotrans.task.orchestrator import TaskRequest, TaskStatus, run
    from videotrans.util._ffprobe import get_video_info

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        roi = (0.05, 0.74, 0.90, 0.22)
        vid = build_segmented_video(
            tmp_path / "audioless_render.mp4",
            [(2.5, ["Render audio-less cue"], "en")],
            with_audio=False,
            roi_box=roi,
        )
        with LocalOpenAITranslationServer(prefix="[ZH]") as srv:
            runtime_config.params["chatgpt_key"] = "local-key"
            runtime_config.params["chatgpt_api"] = srv.base_url
            runtime_config.params["chatgpt_model"] = "gpt-4o-mini"

            out_dir = iso["output_dir"] / "audioless_render_out"
            cache_dir = iso["temp_dir"] / "audioless_render_cache"
            res = run(
                TaskRequest({
                    "name": vid.as_posix(),
                    "target_dir": out_dir.as_posix(),
                    "cache_folder": cache_dir.as_posix(),
                    "source_language_code": "en",
                    "target_language_code": "zh-cn",
                    "translate_type": translator.CHATGPT_INDEX,
                    "voice_role": "No",
                    "is_separate": False,
                    "subtitle_type": 1,
                    "subtitle_source": "video_ocr",
                    "ocr_roi": roi,
                    "ocr_roi_confirmed": True,
                })
            )
            assert res.status == TaskStatus.SUCCEEDED, res.failure
            mp4s = [p for p in res.outputs if p.suffix.lower() == ".mp4"]
            assert mp4s and mp4s[0].is_file()
            info = get_video_info(mp4s[0].as_posix())
            assert info.get("video_streams", 0) >= 1
    finally:
        restore_runtime_config(iso)


def test_t3_pair07_language_switch_with_use_angle_cls_false_and_unwrap():
    """F10 (use_angle_cls=False + paragraph_mode='unwrap' + EN->ZH->EN switch)."""
    from videotrans.ocr._paddle import PaddleOcrProvider

    provider = PaddleOcrProvider(device="cpu", use_angle_cls=False, paragraph_mode="unwrap")
    en_lines = ["First unwrapped sentence", "continues onto line two."]
    zh_lines = ["第一行中文无空格拼接", "第二行中文继续衔接。"]
    en_img = make_paragraph_bgr(en_lines, lang="en", size=(960, 180))
    zh_img = make_paragraph_bgr(zh_lines, lang="zh-cn", size=(960, 180))

    r1 = provider.recognize(en_img, language="en")
    assert "\n" not in r1.text
    assert normalized_text_similarity(r1.text, " ".join(en_lines)) >= 0.92

    r2 = provider.recognize(zh_img, language="zh-cn")
    assert "\n" not in r2.text
    assert " " not in r2.text
    assert normalized_text_similarity(r2.text, "".join(zh_lines)) >= 0.92

    r3 = provider.recognize(en_img, language="en")
    assert normalized_text_similarity(r3.text, " ".join(en_lines)) >= 0.92


def test_t3_pair08_stage2_translate_and_stage3_tts_preview_concurrency(tmp_path, monkeypatch):
    """F8 (Stage 2 /api/translate) x F9 (Stage 3 /api/tts/preview non-blocking concurrency)."""
    from videotrans import translator
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.configure import config as runtime_config
    from videotrans.core import secret_store

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    _install_fake_elevenlabs_wav(monkeypatch, tmp_path, sample_rate=24000, delay_sec=0.8)
    try:
        with LocalOpenAITranslationServer(prefix="[ES]") as srv:
            secret_store.set_secret("chatgpt_key", "local-key")
            runtime_config.params["chatgpt_key"] = "local-key"
            runtime_config.params["chatgpt_api"] = srv.base_url
            runtime_config.params["chatgpt_model"] = "gpt-4o-mini"
            runtime_config.params["elevenlabstts_key"] = "local-el-key"

            app = create_app(
                upload_dir=iso["uploads_dir"],
                task_params_builder=functools.partial(
                    build_task_params,
                    output_dir=iso["output_dir"],
                    temp_dir=iso["temp_dir"],
                ),
            )
            with TestClient(app) as client:
                def call_translate():
                    return client.post(
                        "/api/translate",
                        json={
                            "segments": [
                                {"id": 1, "startSec": 0.0, "endSec": 1.5, "sourceText": "Concurrent line A"},
                                {"id": 2, "startSec": 1.5, "endSec": 3.0, "sourceText": "Concurrent line B"},
                            ],
                            "sourceLanguage": "en",
                            "targetLanguage": "es",
                            "translateType": translator.CHATGPT_INDEX,
                            "translationMode": "srt",
                        },
                    )

                def call_preview(idx: int):
                    return client.post(
                        "/api/tts/preview",
                        json={
                            "provider": "elevenlabs",
                            "voice": "Rachel",
                            "text": f"Preview concurrent clip {idx}",
                            "language": "es",
                            "forceRefresh": True,
                        },
                    )

                # Wait for lifespan's background vieneu-warmup thread (transformers/vieneu)
                # and warm cold translation SDK imports before timing concurrent execution.
                for th in threading.enumerate():
                    if th.name == "vieneu-warmup" and th.is_alive():
                        th.join(timeout=20.0)
                warmup_tr = client.post(
                    "/api/translate",
                    json={
                        "segments": [
                            {"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Warmup translation line"},
                        ],
                        "sourceLanguage": "en",
                        "targetLanguage": "es",
                        "translateType": translator.CHATGPT_INDEX,
                        "translationMode": "srt",
                    },
                )
                assert warmup_tr.status_code == 200, warmup_tr.text
                reqs_before = len(srv.requests)

                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=4) as pool:
                    f_tr = pool.submit(call_translate)
                    f_p1 = pool.submit(call_preview, 1)
                    f_p2 = pool.submit(call_preview, 2)
                    r_tr = f_tr.result(timeout=15)
                    r_p1 = f_p1.result(timeout=15)
                    r_p2 = f_p2.result(timeout=15)
                elapsed = time.perf_counter() - t0

                assert r_tr.status_code == 200 and "[ES]" in r_tr.json()["segments"][0]["targetText"]
                assert len(srv.requests) > reqs_before
                assert r_p1.status_code == 200 and r_p1.json()["ok"] is True
                assert r_p2.status_code == 200 and r_p2.json()["ok"] is True
                # Concurrent cloud TTS + translation must overlap rather than serialize (>= 1.6s)
                assert elapsed < 1.3, f"Expected concurrent execution (< 1.3s), got {elapsed:.2f}s"
    finally:
        restore_runtime_config(iso)


def test_t3_pair09_settings_provider_update_and_translation_429_retry(tmp_path, monkeypatch):
    """F3/F4 (Settings API update) x F8 (Runtime 429 retry + multiline <think> stripping)."""
    from videotrans import translator
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.configure import config as runtime_config

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        # Keep default retry_nums=1 to prove runtime retry still handles transient 429
        runtime_config.settings["retry_nums"] = 1

        with LocalOpenAITranslationServer(fail_first_n_with_429=2, wrap_think=True, prefix="[FR]") as srv:
            app = create_app(
                upload_dir=iso["uploads_dir"],
                task_params_builder=functools.partial(
                    build_task_params,
                    output_dir=iso["output_dir"],
                    temp_dir=iso["temp_dir"],
                ),
            )
            with TestClient(app) as client:
                save_res = client.post(
                    "/api/translation-settings/openai",
                    json={
                        "apiKey": "local-test-key",
                        "baseUrl": srv.base_url,
                        "model": "gpt-4o-mini",
                    },
                )
                assert save_res.status_code == 200, save_res.text

                tr = client.post(
                    "/api/translate",
                    json={
                        "segments": [{"id": 1, "startSec": 0.0, "endSec": 2.0, "sourceText": "Retry after rate limit"}],
                        "sourceLanguage": "en",
                        "targetLanguage": "fr",
                        "translateType": translator.CHATGPT_INDEX,
                        "translationMode": "srt",
                    },
                )
                assert tr.status_code == 200, tr.text
                out_txt = tr.json()["segments"][0]["targetText"]
                assert "<think>" not in out_txt
                assert "[FR]" in out_txt
                assert len(srv.requests) >= 3
    finally:
        restore_runtime_config(iso)


def test_t3_pair10_deepgram_detect_language_none_merge_sub_and_stage2_reset(tmp_path, monkeypatch):
    """F6 (Deepgram detect_language=None + _merge_sub) x F2 (Stage 2 reset)."""
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.configure import config as runtime_config
    from videotrans.core.project_store import init_project_dirs
    from videotrans.recognition._deepgram import DeepgramRecogn

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        wav = make_wav(tmp_path / "dg_test.wav", duration_sec=2.0, sample_rate=16000)
        runtime_config.params["deepgram_apikey"] = "dg-local-key"

        # Fake only the remote Deepgram network call
        class _Utt:
            def __init__(self, start, end, transcript, speaker=0):
                self.start = start
                self.end = end
                self.transcript = transcript
                self.speaker = speaker

        class _FakeDGRes:
            def __init__(self):
                self.results = type(
                    "Results",
                    (),
                    {
                        "utterances": [
                            _Utt(0.0, 1.0, "Hello world.", 0),
                            _Utt(1.0, 2.0, "Second line.", 0),
                        ],
                        "channels": [type("Chan", (), {"detected_language": "en", "alternatives": []})()],
                    },
                )()

            def to_json(self, indent=4):
                return json.dumps({
                    "results": {
                        "utterances": [
                            {"start": 0.0, "end": 1.0, "transcript": "Hello world.", "speaker": 0},
                            {"start": 1.0, "end": 2.0, "transcript": "Second line.", "speaker": 0},
                        ],
                        "channels": [{
                            "detected_language": "en",
                            "alternatives": [{
                                "transcript": "Hello world. Second line.",
                                "words": [
                                    {"word": "Hello", "start": 0.0, "end": 0.4, "punctuated_word": "Hello"},
                                    {"word": "world", "start": 0.5, "end": 0.9, "punctuated_word": "world."},
                                ],
                            }],
                        }],
                    }
                })

        class _FakeRestV1:
            def transcribe_file(self, *_args, **_kwargs):
                return _FakeDGRes()

        class _FakeDGClient:
            def __init__(self, *_args, **_kwargs):
                self.listen = type("Listen", (), {"rest": type("Rest", (), {"v": lambda _self, _v: _FakeRestV1()})()})()

        from videotrans.recognition import _deepgram as dg_mod
        monkeypatch.setattr(dg_mod, "DeepgramClient", _FakeDGClient)

        dg = DeepgramRecogn(
            audio_file=str(wav),
            detect_language=None,
            cache_folder=str(iso["temp_dir"] / "dg_cache"),
            model_name="nova-3",
        )
        Path(dg.cache_folder).mkdir(parents=True, exist_ok=True)
        items = dg._exec()
        assert items and len(items) >= 1

        # _merge_sub with empty-text middle item must not crash with IndexError
        merged = dg._merge_sub([
            {"line": 1, "start_time": 0, "end_time": 1000, "startraw": "00:00:00,000", "endraw": "00:00:01,000", "time": "00:00:00,000 --> 00:00:01,000", "text": "Hello,"},
            {"line": 2, "start_time": 1000, "end_time": 1200, "startraw": "00:00:01,000", "endraw": "00:00:01,200", "time": "00:00:01,000 --> 00:00:01,200", "text": ""},
            {"line": 3, "start_time": 1200, "end_time": 2500, "startraw": "00:00:01,200", "endraw": "00:00:02,500", "time": "00:00:01,200 --> 00:00:02,500", "text": "world."},
        ])
        assert merged

        app = create_app(
            upload_dir=iso["uploads_dir"],
            task_params_builder=functools.partial(
                build_task_params,
                output_dir=iso["output_dir"],
                temp_dir=iso["temp_dir"],
            ),
        )
        with TestClient(app) as client:
            proj = client.post("/api/projects", json={"name": "DG Reset"}).json()
            dirs = init_project_dirs(proj["id"])
            baseline_segs = [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Hello world.", "text": "Hello world."}]
            (dirs["transcripts"] / "asr_baseline.json").write_text(json.dumps({"segments": baseline_segs}), encoding="utf-8")

            client.put(f"/api/projects/{proj['id']}/state", json={"stage": 2, "state": {"segments": [{"id": 1, "startSec": 0.0, "endSec": 1.0, "sourceText": "Edited"}]}})
            rst = client.post(f"/api/projects/{proj['id']}/stages/2/reset")
            assert rst.status_code == 200
            assert rst.json()["state"]["segments"][0]["sourceText"] == "Hello world."
    finally:
        restore_runtime_config(iso)


def test_t3_pair11_cli_vtv_video_ocr_with_local_llm_server(tmp_path, monkeypatch):
    """F12 (cli.py --task vtv --subtitle_source video_ocr) x F7/F8 (Local OpenAI server)."""
    import cli
    from videotrans.configure import config as runtime_config

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        vid = build_segmented_video(
            tmp_path / "cli_vtv.mp4",
            [(2.5, ["CLI VTV Subtitle"], "en")],
            with_audio=False,
            roi_box=(0.05, 0.74, 0.90, 0.22),
        )
        out_dir = iso["output_dir"] / "cli_vtv_out"
        with LocalOpenAITranslationServer(prefix="[ZH]") as srv:
            runtime_config.params["chatgpt_key"] = "local-key"
            runtime_config.params["chatgpt_api"] = srv.base_url
            runtime_config.params["chatgpt_model"] = "gpt-4o-mini"

            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "cli.py",
                    "--task", "vtv",
                    "--name", str(vid),
                    "--subtitle_source", "video_ocr",
                    "--ocr_roi", "0.05,0.74,0.90,0.22",
                    "--source_language_code", "en",
                    "--target_language_code", "zh-cn",
                    "--translate_type", "1",
                    "--voice_role", "No",
                    "--subtitle_type", "1",
                    "--output-dir", str(out_dir),
                ],
            )
            rc = cli.main()
            assert rc == 0
            assert (out_dir / "en.srt").is_file()
            assert (out_dir / "zh-cn.srt").is_file()
            assert "[ZH]" in (out_dir / "zh-cn.srt").read_text(encoding="utf-8")
            assert list(out_dir.glob("*.mp4"))
    finally:
        restore_runtime_config(iso)


def test_t3_pair12_ocr_checkpoint_resume_and_stage2_cached_provider_reuse(tmp_path):
    """F11 (OcrCheckpoint stop/resume) x F10/F4 (Stage 2 extract_ocr_segment_text cached provider)."""
    from videotrans.api.ocr_helpers import extract_ocr_segment_text
    from videotrans.ocr import scan_video_to_srt

    roi = (0.05, 0.74, 0.90, 0.22)
    vid = build_segmented_video(
        tmp_path / "ckpt_stage2.mp4",
        [
            (2.5, ["Part One Text"], "en"),
            (2.5, ["Part Two Text"], "en"),
            (2.5, ["Part Three Text"], "en"),
        ],
        roi_box=roi,
    )
    ckpt = tmp_path / "ckpt.json"
    stop = False

    def on_prog(ev: dict[str, Any]) -> None:
        nonlocal stop
        if ev.get("timestamp_ms", 0) >= 3500:
            stop = True

    scan_video_to_srt(
        str(vid),
        roi=roi,
        language="en",
        checkpoint_path=str(ckpt),
        stop_event=lambda: stop,
        on_progress=on_prog,
    )
    assert ckpt.is_file()

    # Interleaved Stage 2 inspection on same video
    s2 = extract_ocr_segment_text(vid, 2.5, 5.0, roi, language="en")
    assert s2["ok"] is True
    assert "Part Two" in s2["text"]

    # Resume checkpoint
    _, final_segs = scan_video_to_srt(str(vid), roi=roi, language="en", checkpoint_path=str(ckpt))
    assert len(final_segs) == 3
    assert "Part Three" in final_segs[2].text


# ===========================================================================
# Tier 4: Full End-to-End Real-World Scenarios (>= 5)
# ===========================================================================


def test_t4_scenario1_audioless_video_ocr_translate_dubbing_capcut(tmp_path, monkeypatch):
    """Scenario 1: Audio-less 6s 720p Video OCR -> Local HTTP AI Translation -> TTS Dubbing -> CapCut ZIP."""
    from videotrans import translator
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.configure import config as runtime_config
    from videotrans.core import secret_store
    from videotrans.core.job_manager import JobManager, run_prepare_review

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    _install_fake_elevenlabs_wav(monkeypatch, tmp_path, sample_rate=24000)
    try:
        roi = [0.05, 0.72, 0.90, 0.24]
        vid = build_segmented_video(
            tmp_path / "s1_audioless.mp4",
            [
                (2.0, ["Welcome to the demo", "Subtitle line two"], "en"),
                (2.0, ["Second scene caption"], "en"),
                (2.0, ["Final closing line"], "en"),
            ],
            with_audio=False,
            roi_box=tuple(roi),
            font_size=30,
        )

        with LocalOpenAITranslationServer(prefix="[VI]") as srv:
            secret_store.set_secret("chatgpt_key", "local-key")
            runtime_config.params["chatgpt_key"] = "local-key"
            runtime_config.params["chatgpt_api"] = srv.base_url
            runtime_config.params["chatgpt_model"] = "gpt-4o-mini"
            runtime_config.params["elevenlabstts_key"] = "local-el-key"

            app = create_app(
                job_manager=JobManager(runner=run_prepare_review),
                upload_dir=iso["uploads_dir"],
                task_params_builder=functools.partial(
                    build_task_params,
                    output_dir=iso["output_dir"],
                    temp_dir=iso["temp_dir"],
                ),
            )
            with TestClient(app) as client:
                with vid.open("rb") as f:
                    media = client.post("/api/media", files={"file": ("s1_audioless.mp4", f, "video/mp4")}).json()
                proj = client.post(
                    "/api/projects",
                    json={"name": "Scenario 1", "mediaId": media["id"], "duration": 6.0},
                ).json()

                # Stage 1: Video OCR
                r_j = client.post(
                    "/api/jobs",
                    json={
                        "mediaId": media["id"],
                        "projectId": proj["id"],
                        "jobType": "asr",
                        "options": {
                            "subtitleSource": "video_ocr",
                            "ocrRoi": roi,
                            "ocrRoiConfirmed": True,
                            "sourceLanguage": "en",
                            "targetLanguage": "vi",
                        },
                    },
                )
                assert r_j.status_code == 202, r_j.text
                done = _wait_for_job(client, r_j.json()["id"])
                assert done["status"] == "succeeded", done
                assert len(done["segments"]) == 3
                assert "\n" in done["segments"][0]["sourceText"]

                # Stage 2: AI Translation
                tr = client.post(
                    "/api/translate",
                    json={
                        "projectId": proj["id"],
                        "segments": done["segments"],
                        "sourceLanguage": "en",
                        "targetLanguage": "vi",
                        "translateType": translator.CHATGPT_INDEX,
                        "translationMode": "srt",
                    },
                )
                assert tr.status_code == 200, tr.text
                t_segs = tr.json()["segments"]
                for s in t_segs:
                    s["rawTranslation"] = s["targetText"]

                # Stage 3: TTS preview + Dubbing assembly
                for s in t_segs:
                    pv = client.post(
                        "/api/tts/preview",
                        json={"provider": "elevenlabs", "voice": "Rachel", "text": s["targetText"], "language": "vi"},
                    )
                    assert pv.status_code == 200, pv.text
                    s["previewAudioId"] = pv.json()["preview_id"]

                asm = client.post(f"/api/projects/{proj['id']}/dubbing/assemble", json={"segments": t_segs})
                assert asm.status_code == 200, asm.text
                client.put(f"/api/projects/{proj['id']}/state", json={"stage": 4, "state": {"segments": t_segs}})

                # Stage 4: CapCut bundle.zip export
                bundle = client.get(f"/api/projects/{proj['id']}/export-capcut/bundle.zip")
                assert bundle.status_code == 200
                with zipfile.ZipFile(io.BytesIO(bundle.content)) as zf:
                    names = set(zf.namelist())
                    assert any(n.endswith("subtitles_edited.srt") for n in names)
                    assert any(n.endswith("subtitles_target.srt") for n in names)
                    assert any(n.endswith("voiceover_merged.wav") for n in names)
    finally:
        restore_runtime_config(iso)


def test_t4_scenario2_batch_folder_ingest_ocr_repair_translate_assemble_render(tmp_path, monkeypatch):
    """Scenario 2: Batch folder ingest -> Video OCR -> Stage 2 repair -> Translate -> 24k->48k assemble -> Stage 4 MP4 render."""
    from videotrans import translator
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.configure import config as runtime_config
    from videotrans.core import secret_store
    from videotrans.core.job_manager import JobManager
    from videotrans.task.orchestrator import run
    from videotrans.util._ffprobe import get_video_info

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    _install_fake_elevenlabs_wav(monkeypatch, tmp_path, sample_rate=24000)
    try:
        folder = tmp_path / "batch_s2"
        folder.mkdir()
        roi = [0.05, 0.72, 0.90, 0.24]
        v1 = build_segmented_video(folder / "ep1.mp4", [(2.5, ["Episode one line A", "Episode one line B"], "en")], with_audio=True, roi_box=tuple(roi), font_size=30)
        v2 = build_segmented_video(folder / "ep2.mp4", [(2.5, ["Episode two line A", "Episode two line B"], "en")], with_audio=True, roi_box=tuple(roi), font_size=30)

        with LocalOpenAITranslationServer(prefix="[ZH]") as srv:
            secret_store.set_secret("chatgpt_key", "local-key")
            runtime_config.params["chatgpt_key"] = "local-key"
            runtime_config.params["chatgpt_api"] = srv.base_url
            runtime_config.params["chatgpt_model"] = "gpt-4o-mini"
            runtime_config.params["elevenlabstts_key"] = "local-el-key"

            # Use full orchestrator.run in JobManager so /api/render produces real MP4
            app = create_app(
                job_manager=JobManager(runner=run),
                upload_dir=iso["uploads_dir"],
                task_params_builder=functools.partial(
                    build_task_params,
                    output_dir=iso["output_dir"],
                    temp_dir=iso["temp_dir"],
                ),
            )
            with TestClient(app) as client:
                scan = client.post("/api/media/scan-folders", json={"paths": [str(folder)]}).json()
                assert scan["count"] == 2
                reg = client.post("/api/media/register-local", json={"paths": [str(v1), str(v2)]}).json()
                batch = client.post(
                    "/api/projects/batch-create",
                    json={"items": [{"name": m["filename"], "mediaId": m["id"]} for m in reg["media"]]},
                ).json()
                subs = batch["projects"]

                for sub in subs:
                    r_asr = client.post(
                        "/api/jobs",
                        json={
                            "mediaId": sub["media_id"],
                            "projectId": sub["id"],
                            "jobType": "asr",
                            "options": {
                                "subtitleSource": "video_ocr",
                                "ocrRoi": roi,
                                "ocrRoiConfirmed": True,
                                "sourceLanguage": "en",
                                "targetLanguage": "zh-cn",
                            },
                        },
                    )
                    assert r_asr.status_code == 202, r_asr.text
                    done_asr = _wait_for_job(client, r_asr.json()["id"])
                    assert done_asr["status"] == "succeeded"

                    # Stage 2 multi-line OCR repair
                    rep = client.post(
                        "/api/ocr/extract",
                        json={"mediaId": sub["media_id"], "startSec": 0.0, "endSec": 2.5, "roi": roi, "language": "en"},
                    ).json()
                    assert "\n" in rep["text"]
                    segs = [dict(done_asr["segments"][0], sourceText=rep["text"])]

                    # Batch translate
                    tr = client.post(
                        "/api/translate",
                        json={
                            "projectId": sub["id"],
                            "segments": segs,
                            "sourceLanguage": "en",
                            "targetLanguage": "zh-cn",
                            "translateType": translator.CHATGPT_INDEX,
                        },
                    ).json()
                    t_segs = tr["segments"]

                    # Dubbing preview + 24kHz->48kHz assembly
                    pv = client.post(
                        "/api/tts/preview",
                        json={"provider": "elevenlabs", "voice": "Rachel", "text": t_segs[0]["targetText"], "language": "zh-cn"},
                    ).json()
                    t_segs[0]["previewAudioId"] = pv["preview_id"]
                    asm = client.post(f"/api/projects/{sub['id']}/dubbing/assemble", json={"segments": t_segs})
                    assert asm.status_code == 200

                # Stage 4 video render on first subproject
                s0 = client.get(f"/api/projects/{subs[0]['id']}").json()
                t_segs_0 = s0["state"]["segments"]
                srt_str = f"1\n00:00:00,000 --> 00:00:02,500\n{t_segs_0[0]['targetText']}\n"
                r_render = client.post(
                    "/api/render",
                    json={
                        "mediaId": subs[0]["media_id"],
                        "projectId": subs[0]["id"],
                        "jobType": "render",
                        "options": {
                            "sourceLanguage": "en",
                            "targetLanguage": "zh-cn",
                            "subtitles": srt_str,
                            "voiceRole": "No",
                        },
                    },
                )
                assert r_render.status_code == 202, r_render.text
                done_render = _wait_for_job(client, r_render.json()["id"], timeout=60.0)
                assert done_render["status"] == "succeeded", done_render
                mp4_out = next(o for o in done_render["outputs"] if o["name"].endswith(".mp4"))
                dl = client.get(mp4_out["url"])
                assert dl.status_code == 200
                rendered_file = tmp_path / "rendered_ep1.mp4"
                rendered_file.write_bytes(dl.content)
                vinfo = get_video_info(rendered_file.as_posix())
                assert vinfo.get("video_streams", 0) >= 1
    finally:
        restore_runtime_config(iso)


def test_t4_scenario3_cli_vtv_video_ocr_full_pipeline(tmp_path, monkeypatch):
    """Scenario 3: CLI --task vtv --subtitle_source video_ocr full pipeline producing .srt, .wav, .mp4."""
    import cli
    from videotrans import tts
    from videotrans.configure import config as runtime_config
    from videotrans.util._ffprobe import get_video_info

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    _install_fake_elevenlabs_wav(monkeypatch, tmp_path, sample_rate=24000)
    try:
        vid = build_segmented_video(
            tmp_path / "s3_cli_full.mp4",
            [(2.5, ["CLI Full Pipeline Cue"], "en")],
            with_audio=False,
            roi_box=(0.05, 0.74, 0.90, 0.22),
        )
        out_dir = iso["output_dir"] / "s3_cli_out"
        with LocalOpenAITranslationServer(prefix="[ZH]") as srv:
            runtime_config.params["chatgpt_key"] = "local-key"
            runtime_config.params["chatgpt_api"] = srv.base_url
            runtime_config.params["chatgpt_model"] = "gpt-4o-mini"
            runtime_config.params["elevenlabstts_key"] = "local-el-key"

            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "cli.py",
                    "--task", "vtv",
                    "--name", str(vid),
                    "--subtitle_source", "video_ocr",
                    "--ocr_roi", "0.05,0.74,0.90,0.22",
                    "--source_language_code", "en",
                    "--target_language_code", "zh-cn",
                    "--translate_type", "1",
                    "--tts_type", str(tts.ELEVENLABS_TTS),
                    "--voice_role", "Rachel",
                    "--subtitle_type", "1",
                    "--output-dir", str(out_dir),
                ],
            )
            rc = cli.main()
            assert rc == 0
            assert (out_dir / "en.srt").is_file()
            assert (out_dir / "zh-cn.srt").is_file()
            mp4s = list(out_dir.glob("*.mp4"))
            assert mp4s, f"Expected output .mp4 in {out_dir}"
            info = get_video_info(mp4s[0].as_posix())
            assert info.get("video_streams", 0) >= 1
            assert info.get("streams_audio", 0) >= 1
            assert (out_dir / "zh-cn.m4a").is_file() or (out_dir / "en.m4a").is_file() or info.get("streams_audio", 0) >= 1
    finally:
        restore_runtime_config(iso)


def test_t4_scenario4_stage_reset_and_recovery_workflow(tmp_path, monkeypatch):
    """Scenario 4: ASR -> Stage 2 edit & OCR -> Reset Stage 2 -> asr_baseline.json restored & ocrCrop cleared."""
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.core.job_manager import JobManager, run_prepare_review
    from videotrans.core.project_store import get_project, get_project_dir

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        roi = [0.05, 0.74, 0.90, 0.22]
        vid = build_segmented_video(
            tmp_path / "s4_recovery.mp4",
            [(2.5, ["Baseline Recovery Cue"], "en")],
            with_audio=False,
            roi_box=tuple(roi),
        )
        app = create_app(
            job_manager=JobManager(runner=run_prepare_review),
            upload_dir=iso["uploads_dir"],
            task_params_builder=functools.partial(
                build_task_params,
                output_dir=iso["output_dir"],
                temp_dir=iso["temp_dir"],
            ),
        )
        with TestClient(app) as client:
            with vid.open("rb") as f:
                media = client.post("/api/media", files={"file": ("s4_recovery.mp4", f, "video/mp4")}).json()
            proj = client.post(
                "/api/projects",
                json={
                    "name": "Scenario 4",
                    "mediaId": media["id"],
                    "state": {"languages": {"source": {"code": "en"}, "target": {"code": "vi"}}},
                },
            ).json()

            r_j = client.post(
                "/api/jobs",
                json={
                    "mediaId": media["id"],
                    "projectId": proj["id"],
                    "jobType": "asr",
                    "options": {"subtitleSource": "video_ocr", "ocrRoi": roi, "ocrRoiConfirmed": True, "sourceLanguage": "en"},
                },
            )
            assert r_j.status_code == 202, r_j.text
            done = _wait_for_job(client, r_j.json()["id"])
            assert done["status"] == "succeeded"

            baseline_file = get_project_dir(proj["id"]) / "transcripts" / "asr_baseline.json"
            assert baseline_file.is_file()
            baseline_data = json.loads(baseline_file.read_text(encoding="utf-8"))

            # Stage 2 OCR & edit + ocrCrop
            client.post("/api/ocr/extract", json={"mediaId": media["id"], "startSec": 0.0, "endSec": 2.5, "roi": roi, "language": "en"})
            client.put(
                f"/api/projects/{proj['id']}/state",
                json={
                    "stage": 2,
                    "state": {
                        "segments": [{"id": 1, "startSec": 0.0, "endSec": 2.5, "sourceText": "Corrupted Edit"}],
                        "ocrCrop": {"x": 0.05, "y": 0.74, "w": 0.9, "h": 0.22},
                    },
                },
            )

            # Reset Stage 2
            rst = client.post(f"/api/projects/{proj['id']}/stages/2/reset")
            assert rst.status_code == 200
            st = rst.json()["state"]
            assert st["segments"] == baseline_data["segments"]
            assert "ocrCrop" not in st or st["ocrCrop"] is None

            # Subsequent partial autosave preserves parent_id & languages and keeps ocrCrop cleared
            client.put(f"/api/projects/{proj['id']}/state", json={"stage": 2, "state": {"currentStep": 2}})
            final_proj = get_project(proj["id"])
            assert final_proj is not None
            assert final_proj["state"].get("ocrCrop") is None
            assert final_proj["state"].get("parent_id") == proj["parent_id"]
            assert final_proj["state"].get("languages", {}).get("target", {}).get("code") == "vi"
    finally:
        restore_runtime_config(iso)


def test_t4_scenario5_multilingual_en_zh_video_ocr_and_concurrent_stage2_inspection(tmp_path, monkeypatch):
    """Scenario 5: Multi-language EN+ZH Video OCR + concurrent Stage 2 OCR extraction with thread safety."""
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.ocr import scan_video_to_srt

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        roi = (0.05, 0.70, 0.90, 0.26)
        en_lines = ["Concurrent English line one", "Concurrent English line two"]
        zh_lines = ["并发中文第一行字幕", "并发中文第二行字幕"]
        vid = build_segmented_video(
            tmp_path / "s5_bilingual_concurrent.mp4",
            [
                (3.0, en_lines, "en"),
                (3.0, zh_lines, "zh-cn"),
            ],
            roi_box=roi,
            font_size=30,
        )

        app = create_app(
            upload_dir=iso["uploads_dir"],
            task_params_builder=functools.partial(
                build_task_params,
                output_dir=iso["output_dir"],
                temp_dir=iso["temp_dir"],
            ),
        )
        with TestClient(app) as client:
            with vid.open("rb") as f:
                media_id = client.post("/api/media", files={"file": ("s5_bilingual_concurrent.mp4", f, "video/mp4")}).json()["id"]

            def run_full_scan():
                return scan_video_to_srt(str(vid), roi=roi, language="zh-cn", coarse_interval_ms=500)

            def run_stage2_en():
                return client.post(
                    "/api/ocr/extract",
                    json={"mediaId": media_id, "startSec": 0.0, "endSec": 3.0, "roi": list(roi), "language": "en"},
                )

            def run_stage2_zh():
                return client.post(
                    "/api/ocr/extract",
                    json={"mediaId": media_id, "startSec": 3.0, "endSec": 6.0, "roi": list(roi), "language": "zh-cn"},
                )

            with ThreadPoolExecutor(max_workers=3) as pool:
                f_scan = pool.submit(run_full_scan)
                f_en = pool.submit(run_stage2_en)
                f_zh = pool.submit(run_stage2_zh)

                _, scan_segs = f_scan.result(timeout=60)
                r_en = f_en.result(timeout=60)
                r_zh = f_zh.result(timeout=60)

            assert len(scan_segs) == 2
            assert normalized_text_similarity(scan_segs[0].text, "\n".join(en_lines)) >= 0.88
            assert normalized_text_similarity(scan_segs[1].text, "\n".join(zh_lines)) >= 0.88

            assert r_en.status_code == 200
            assert r_zh.status_code == 200
            assert normalized_text_similarity(r_en.json()["text"], "\n".join(en_lines)) >= 0.88
            assert normalized_text_similarity(r_zh.json()["text"], "\n".join(zh_lines)) >= 0.88
            assert " " not in r_zh.json()["text"]
    finally:
        restore_runtime_config(iso)
