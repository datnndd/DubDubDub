# -*- coding: utf-8 -*-
"""E2E Opaque-Box Tests for F11 & F12: R5 Resource-Efficient Video OCR & Pipeline Integration.

Uses real FFmpeg-encoded 720p H.264 clips and real offline PaddleOcrProvider.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.e2e.media_synth import (
    build_segmented_video,
    isolate_runtime_env,
    make_paragraph_bgr,
    normalized_text_similarity,
    require_ocr_env,
    restore_runtime_config,
)

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _check_env():
    require_ocr_env()


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 45.0) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Tier 1: Primary Feature Coverage (F11 Scanner & F12 WebUI/API/CLI)
# ---------------------------------------------------------------------------


def test_t1_scan_video_to_srt_multisegment_and_single_token(tmp_path):
    from videotrans.ocr import scan_video_to_srt

    roi = (0.05, 0.72, 0.90, 0.24)
    segments_spec = [
        (1.0, [], "en"),  # 0.0..1.0s blank
        (2.5, ["First English cue line one", "Second English cue line two"], "en"),  # 1.0..3.5s
        (2.5, ["第三段中文字幕第一行", "第三段中文字幕第二行"], "zh-cn"),  # 3.5..6.0s
        (1.0, [], "en"),  # 6.0..7.0s blank
        (1.5, ["Wait 10s"], "en"),  # 7.0..8.5s
        (1.0, ["Wait 20s"], "en"),  # 8.5..9.5s (single-token change)
    ]
    vid = build_segmented_video(tmp_path / "multi_9_5s.mp4", segments_spec, roi_box=roi, font_size=30)

    progress_events: list[dict[str, Any]] = []
    srt_items, ocr_segments = scan_video_to_srt(
        str(vid),
        roi=roi,
        language="zh-cn",
        device="auto",
        coarse_interval_ms=500,
        on_progress=progress_events.append,
    )

    expected_texts = [
        "First English cue line one\nSecond English cue line two",
        "第三段中文字幕第一行\n第三段中文字幕第二行",
        "Wait 10s",
        "Wait 20s",
    ]
    assert len(ocr_segments) == 4, (
        f"Expected 4 segments (single-token Wait 10s->Wait 20s must NOT merge), got {len(ocr_segments)}: {ocr_segments}"
    )
    assert len(srt_items) == 4

    for seg, exp in zip(ocr_segments, expected_texts):
        assert normalized_text_similarity(seg.text, exp) >= 0.90, (
            f"Segment text mismatch: got {seg.text!r}, expected {exp!r}"
        )

    assert "10" in ocr_segments[2].text and "20" in ocr_segments[3].text, (
        f"Single-token change Wait 10s -> Wait 20s was lost: {[s.text for s in ocr_segments]!r}"
    )

    assert progress_events, "Expected progress events during scan_video_to_srt"
    last_ev = progress_events[-1]
    assert last_ev.get("device") == "cpu", f"Expected accurate device='cpu', got {last_ev.get('device')!r}"
    total_coarse_frames = 19  # 9500ms / 500ms
    skipped = last_ev.get("skipped_frames")
    if skipped is not None:
        assert skipped > 0, f"Expected dedup to skip static coarse frames: {last_ev}"
        assert (total_coarse_frames - skipped) < total_coarse_frames
    else:
        assert last_ev.get("ocr_calls", 999) <= total_coarse_frames + 5, (
            f"Expected dedup to limit OCR calls: ocr_calls={last_ev.get('ocr_calls')} vs {total_coarse_frames}"
        )


def test_t1_scan_video_to_srt_boundary_accuracy_250ms(tmp_path):
    from videotrans.ocr import scan_video_to_srt

    roi = (0.05, 0.74, 0.90, 0.22)
    segments_spec = [
        (1.0, [], "en"),
        (2.5, ["Boundary test alpha"], "en"),   # 1000..3500 (EMPTY->ACTIVE, ACTIVE->ACTIVE)
        (2.5, ["Boundary test beta"], "en"),    # 3500..6000 (ACTIVE->ACTIVE, ACTIVE->EMPTY)
        (1.0, [], "en"),
        (1.5, ["Wait 10s"], "en"),              # 7000..8500
        (1.0, ["Wait 20s"], "en"),              # 8500..9500
    ]
    expected_bounds = [
        (1000, 3500, "Boundary test alpha"),
        (3500, 6000, "Boundary test beta"),
        (7000, 8500, "Wait 10s"),
        (8500, 9500, "Wait 20s"),
    ]
    vid = build_segmented_video(tmp_path / "bounds_9_5s.mp4", segments_spec, roi_box=roi, font_size=32)

    _, ocr_segments = scan_video_to_srt(
        str(vid),
        roi=roi,
        language="en",
        device="auto",
        coarse_interval_ms=500,
    )
    assert len(ocr_segments) == len(expected_bounds), (
        f"Expected {len(expected_bounds)} segments, got {len(ocr_segments)}: {ocr_segments}"
    )

    for seg, (exp_start, exp_end, exp_txt) in zip(ocr_segments, expected_bounds):
        assert normalized_text_similarity(seg.text, exp_txt) >= 0.90
        assert abs(seg.start_ms - exp_start) <= 250, (
            f"Start boundary drift >250ms for {exp_txt!r}: got {seg.start_ms}ms vs {exp_start}ms"
        )
        assert abs(seg.end_ms - exp_end) <= 250, (
            f"End boundary drift >250ms for {exp_txt!r}: got {seg.end_ms}ms vs {exp_end}ms"
        )


def test_t1_api_jobs_audioless_video_ocr_asr(tmp_path, monkeypatch):
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.configure import config as runtime_config
    from videotrans.core import secret_store
    from videotrans.core.job_manager import JobManager, run_prepare_review

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        # Ensure NO ASR API keys are configured
        for k in ("deepgram_apikey", "gemini_key", "elevenlabstts_key", "chatgpt_key"):
            secret_store.delete_secret(k)
            runtime_config.params[k] = ""

        roi = [0.05, 0.74, 0.90, 0.22]
        vid = build_segmented_video(
            tmp_path / "audioless_ocr_asr.mp4",
            [
                (2.5, ["Audio-less opening subtitle"], "en"),
                (2.5, ["Silent stream second caption"], "en"),
            ],
            with_audio=False,
            roi_box=tuple(roi),
        )

        jm = JobManager(runner=run_prepare_review)
        app = create_app(
            job_manager=jm,
            upload_dir=iso["uploads_dir"],
            task_params_builder=functools.partial(
                build_task_params,
                output_dir=iso["output_dir"],
                temp_dir=iso["temp_dir"],
            ),
        )
        with TestClient(app) as client:
            with vid.open("rb") as f:
                up = client.post("/api/media", files={"file": ("audioless_ocr_asr.mp4", f, "video/mp4")})
            assert up.status_code == 201, up.text
            media_id = up.json()["id"]

            r_asr = client.post(
                "/api/jobs",
                json={
                    "mediaId": media_id,
                    "jobType": "asr",
                    "options": {
                        "subtitleSource": "video_ocr",
                        "ocrRoi": roi,
                        "ocrRoiConfirmed": True,
                        "sourceLanguage": "en",
                        "targetLanguage": "zh-cn",
                        "recognType": 1,
                    },
                },
            )
            assert r_asr.status_code == 202, r_asr.text
            job_id_asr = r_asr.json()["id"]
            done_asr = _wait_for_job(client, job_id_asr, timeout=60.0)
            assert done_asr["status"] == "succeeded", f"ASR video_ocr job failed: {done_asr}"
            assert (done_asr.get("asrDuration") or 0) > 0, f"Expected asrDuration > 0: {done_asr}"
            assert len(done_asr.get("segments") or []) >= 2
            combined = " ".join(s.get("sourceText", "") for s in done_asr["segments"])
            assert "Audio-less" in combined
            assert any(
                (ev.get("details") or {}).get("source_type") == "ocr_progress"
                or ev.get("source_type") == "ocr_progress"
                for ev in done_asr.get("events", [])
            ), f"Expected ocr_progress events in job telemetry: {done_asr.get('events')}"
    finally:
        restore_runtime_config(iso)


def test_t1_api_jobs_audioless_video_ocr_vtv(tmp_path, monkeypatch):
    from videotrans import translator
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.configure import config as runtime_config
    from videotrans.core import secret_store
    from videotrans.core.job_manager import JobManager, run_prepare_review

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        # Isolate .env lookups/writes and ensure NO ASR or translation API keys are configured initially
        monkeypatch.setenv("VIDEOTRANS_ENV_FILE", str(tmp_path / "e2e.env"))
        for env_k in ("OPENAI_API_KEY", "CHATGPT_KEY", "DEEPGRAM_API_KEY", "GEMINI_API_KEY", "ELEVENLABS_API_KEY"):
            monkeypatch.delenv(env_k, raising=False)
        for k in ("deepgram_apikey", "gemini_key", "elevenlabstts_key", "chatgpt_key"):
            secret_store.delete_secret(k, persist_to_env=False)
            runtime_config.params[k] = ""

        roi = [0.05, 0.74, 0.90, 0.22]
        vid = build_segmented_video(
            tmp_path / "audioless_ocr_vtv.mp4",
            [
                (2.5, ["Audio-less opening subtitle"], "en"),
                (2.5, ["Silent stream second caption"], "en"),
            ],
            with_audio=False,
            roi_box=tuple(roi),
        )

        jm = JobManager(runner=run_prepare_review)
        app = create_app(
            job_manager=jm,
            upload_dir=iso["uploads_dir"],
            task_params_builder=functools.partial(
                build_task_params,
                output_dir=iso["output_dir"],
                temp_dir=iso["temp_dir"],
            ),
        )
        with TestClient(app) as client:
            with vid.open("rb") as f:
                up = client.post("/api/media", files={"file": ("audioless_ocr_vtv.mp4", f, "video/mp4")})
            assert up.status_code == 201, up.text
            media_id = up.json()["id"]

            vtv_payload = {
                "mediaId": media_id,
                "jobType": "vtv",
                "options": {
                    "subtitleSource": "video_ocr",
                    "ocrRoi": roi,
                    "ocrRoiConfirmed": True,
                    "sourceLanguage": "en",
                    "targetLanguage": "zh-cn",
                    "recognType": 1,
                    "translateType": translator.CHATGPT_INDEX,
                    "voiceRole": "No",
                },
            }

            # translation_validator must still reject missing chatgpt_key on vtv even when asr_validator is bypassed
            r_missing_key = client.post("/api/jobs", json=vtv_payload)
            assert r_missing_key.status_code == 400, (
                f"Expected 400 when chatgpt_key is missing on vtv video_ocr: {r_missing_key.status_code} {r_missing_key.text}"
            )

            # Configure translation settings (ASR keys remain absent) and verify Stage 1 vtv prepare succeeds
            secret_store.set_secret("chatgpt_key", "local-test-key", persist_to_env=False)
            runtime_config.params["chatgpt_key"] = "local-test-key"
            runtime_config.params["chatgpt_api"] = "http://127.0.0.1:9999/v1"
            runtime_config.params["chatgpt_model"] = "gpt-4o-mini"

            r_vtv = client.post("/api/jobs", json=vtv_payload)
            assert r_vtv.status_code == 202, r_vtv.text
            done_vtv = _wait_for_job(client, r_vtv.json()["id"], timeout=60.0)
            assert done_vtv["status"] == "succeeded", f"VTV video_ocr job failed: {done_vtv}"
            assert (done_vtv.get("asrDuration") or 0) > 0, f"Expected asrDuration > 0: {done_vtv}"
            assert len(done_vtv.get("segments") or []) >= 2
            combined = " ".join((s.get("text") or s.get("sourceText") or "") for s in done_vtv["segments"])
            assert "Audio-less" in combined
            events = done_vtv.get("events", [])
            assert any(
                (ev.get("details") or {}).get("source_type") == "ocr_progress"
                or ev.get("source_type") == "ocr_progress"
                or ev.get("kind") == "ocr_progress"
                for ev in events
            ), f"Expected ocr_progress events in job telemetry: {events}"
    finally:
        restore_runtime_config(iso)


def test_t1_cli_video_ocr_parser_params_and_stt_execution(tmp_path, monkeypatch):
    import cli

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        roi = (0.05, 0.74, 0.90, 0.22)
        vid = build_segmented_video(
            tmp_path / "cli_ocr.mp4",
            [(2.5, ["CLI video OCR subtitle"], "en")],
            with_audio=False,
            roi_box=roi,
        )
        out_dir = iso["output_dir"] / "cli_out"
        parser = cli.build_parser()
        args = parser.parse_args([
            "--task", "stt",
            "--name", str(vid),
            "--subtitle_source", "video_ocr",
            "--ocr_roi", "0.05,0.74,0.90,0.22",
            "--detect_language", "en",
            "--output-dir", str(out_dir),
        ])

        stt_params = cli.build_stt_params(args)
        assert stt_params.get("subtitle_source") == "video_ocr"
        assert tuple(stt_params.get("ocr_roi") or ()) == pytest.approx(roi)
        assert stt_params.get("ocr_roi_confirmed") is True

        vtv_args = parser.parse_args([
            "--task", "vtv",
            "--name", str(vid),
            "--subtitle_source", "video_ocr",
            "--ocr_roi", "0.05,0.74,0.90,0.22",
            "--source_language_code", "en",
            "--target_language_code", "zh-cn",
        ])
        vtv_params = cli.build_vtv_params(vtv_args)
        assert vtv_params.get("subtitle_source") == "video_ocr"
        assert tuple(vtv_params.get("ocr_roi") or ()) == pytest.approx(roi)
        assert vtv_params.get("ocr_roi_confirmed") is True

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cli.py",
                "--task", "stt",
                "--name", str(vid),
                "--subtitle_source", "video_ocr",
                "--ocr_roi", "0.05,0.74,0.90,0.22",
                "--detect_language", "en",
                "--output-dir", str(out_dir),
            ],
        )
        rc = cli.main()
        assert rc == 0
        srts = list(out_dir.glob("*.srt"))
        assert srts, f"Expected .srt output in {out_dir}"
        srt_text = srts[0].read_text(encoding="utf-8")
        assert normalized_text_similarity(srt_text, "1\n00:00:00,000 --> 00:00:02,500\nCLI video OCR subtitle") >= 0.60
        assert "CLI video OCR" in srt_text
    finally:
        restore_runtime_config(iso)


def test_t1_stage2_extract_ocr_segment_text_deduplicates_static_video(tmp_path):
    import inspect
    from videotrans.api.ocr_helpers import extract_ocr_segment_text
    from videotrans.ocr import PADDLE_OCR, get_cached_provider

    # Warm up cached provider and "en" predictor matching extract_ocr_segment_text's cache key
    warm_kwargs: dict[str, Any] = {
        "provider": PADDLE_OCR,
        "device": "cpu",
        "use_angle_cls": False,
        "paragraph_mode": "preserve_lines",
        "cpu_threads": None,
    }
    sig = inspect.signature(get_cached_provider)
    if not any(par.kind == inspect.Parameter.VAR_KEYWORD for par in sig.parameters.values()):
        warm_kwargs = {k: v for k, v in warm_kwargs.items() if k in sig.parameters}
    try:
        p_warm = get_cached_provider(**warm_kwargs)
    except TypeError:
        p_warm = get_cached_provider(device="cpu")
    p_warm.recognize(make_paragraph_bgr(["Warmup text"], lang="en", size=(640, 120)), "en")

    roi = (0.05, 0.72, 0.90, 0.24)
    expected = [
        "Static segment line one",
        "Static segment line two",
    ]
    vid = build_segmented_video(
        tmp_path / "static_4s.mp4",
        [(4.0, expected, "en")],
        roi_box=roi,
        font_size=30,
    )

    t0 = time.perf_counter()
    res = extract_ocr_segment_text(vid, 0.0, 4.0, roi, language="en")
    elapsed = time.perf_counter() - t0

    assert res["ok"] is True
    assert normalized_text_similarity(res["text"], "\n".join(expected)) >= 0.90
    assert "\n" in res["text"], f"Expected preserved newline in Stage 2 output: {res['text']!r}"
    assert res["samples"] < 10, f"Expected dedup to avoid OCRing all 16 frames, got samples={res['samples']}"
    assert elapsed < 8.0, f"Stage 2 static 4s extract took too long: {elapsed:.2f}s"


def test_t1_f11_high_res_roi_width_clamping_and_coarse_interval(tmp_path):
    from videotrans.ocr import OcrConfig, scan_video_to_srt

    cfg = OcrConfig()
    assert cfg.max_roi_width == 1280
    assert cfg.coarse_interval_ms == 500
    assert cfg.use_angle_cls is False

    # 1080p clip where ROI width is 0.90 * 1920 = 1728px (> max_roi_width=1280)
    roi = (0.05, 0.74, 0.90, 0.22)
    vid = build_segmented_video(
        tmp_path / "hd1080_roi.mp4",
        [
            (2.0, ["Full HD 1080p opening subtitle"], "en"),
            (2.0, ["Second distinct 1080p caption text"], "en"),
        ],
        width=1920,
        height=1080,
        roi_box=roi,
        font_size=40,
    )

    events_fine: list[dict[str, Any]] = []
    srt_fine, segs_fine = scan_video_to_srt(
        str(vid),
        roi=roi,
        language="en",
        coarse_interval_ms=400,
        on_progress=events_fine.append,
    )
    assert len(segs_fine) == 2, f"Expected 2 segments on 1080p clamped ROI, got {segs_fine}"
    assert len(srt_fine) == 2
    assert normalized_text_similarity(segs_fine[0].text, "Full HD 1080p opening subtitle") >= 0.90
    assert normalized_text_similarity(segs_fine[1].text, "Second distinct 1080p caption text") >= 0.90
    assert events_fine

    # Larger coarse_interval_ms=1000 yields fewer OCR calls than fine sampling while preserving cues
    events_coarse: list[dict[str, Any]] = []
    _, segs_coarse = scan_video_to_srt(
        str(vid),
        roi=roi,
        language="en",
        coarse_interval_ms=1000,
        on_progress=events_coarse.append,
    )
    assert len(segs_coarse) == 2
    assert events_coarse[-1]["ocr_calls"] <= 8


def test_t1_f11_static_video_dedup_and_transition_types(tmp_path):
    from videotrans.ocr import scan_video_to_srt

    roi = (0.05, 0.74, 0.90, 0.22)
    # Clip A: 6s static cue -> 12 coarse frames at 500ms, dedup must invoke OCR strictly < 12 times
    static_vid = build_segmented_video(
        tmp_path / "static_6s.mp4",
        [(6.0, ["Static persistent subtitle line"], "en")],
        roi_box=roi,
    )
    ev_static: list[dict[str, Any]] = []
    _, segs_static = scan_video_to_srt(
        str(static_vid),
        roi=roi,
        language="en",
        coarse_interval_ms=500,
        on_progress=ev_static.append,
    )
    assert len(segs_static) == 1
    assert normalized_text_similarity(segs_static[0].text, "Static persistent subtitle line") >= 0.90
    assert segs_static[0].start_ms <= 250
    assert segs_static[0].end_ms >= 5250
    assert ev_static and ev_static[-1]["ocr_calls"] < 12, (
        f"Expected static 6s clip dedup ocr_calls < 12, got {ev_static[-1]}"
    )

    # Clip B: direct ACTIVE -> ACTIVE -> EMPTY transitions with <=250ms accuracy
    trans_vid = build_segmented_video(
        tmp_path / "active_active_empty.mp4",
        [
            (2.0, ["Cue alpha active"], "en"),
            (2.0, ["Cue beta active"], "en"),
            (1.5, [], "en"),
        ],
        roi_box=roi,
    )
    _, segs_trans = scan_video_to_srt(str(trans_vid), roi=roi, language="en", coarse_interval_ms=500)
    assert len(segs_trans) == 2, f"Expected 2 segments for ACTIVE->ACTIVE->EMPTY, got {segs_trans}"
    assert abs(segs_trans[0].end_ms - 2000) <= 250
    assert abs(segs_trans[1].start_ms - 2000) <= 250
    assert abs(segs_trans[1].end_ms - 4000) <= 250


# ---------------------------------------------------------------------------
# Tier 2: Boundary & Corner Cases (Odd ROI, Blank/Tiny, Watermark, Resume, Cancel)
# ---------------------------------------------------------------------------


def test_t2_odd_height_roi_no_shearing_scan_and_stage2(tmp_path, monkeypatch):
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.ocr import scan_video_to_srt

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        # On 1280x720: round(0.72*720)=518, round((0.72+0.24)*720)=691 -> 173px (ODD height!)
        odd_roi = (0.05, 0.72, 0.90, 0.24)
        expected = ["Odd height ROI 173px test"]
        vid = build_segmented_video(
            tmp_path / "odd_roi.mp4",
            [(2.5, expected, "en")],
            roi_box=odd_roi,
            font_size=30,
        )

        _, segs = scan_video_to_srt(str(vid), roi=odd_roi, language="en", coarse_interval_ms=500)
        assert len(segs) == 1
        assert normalized_text_similarity(segs[0].text, expected[0]) >= 0.90

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
                media_id = client.post("/api/media", files={"file": ("odd_roi.mp4", f, "video/mp4")}).json()["id"]
            r = client.post(
                "/api/ocr/extract",
                json={"mediaId": media_id, "startSec": 0.0, "endSec": 2.5, "roi": list(odd_roi), "language": "en"},
            )
            assert r.status_code == 200, r.text
            assert normalized_text_similarity(r.json()["text"], expected[0]) >= 0.90
    finally:
        restore_runtime_config(iso)


def test_t2_blank_video_returns_empty_and_tiny_roi_extracts_text(tmp_path):
    from videotrans.ocr import scan_video_to_srt

    roi = (0.05, 0.74, 0.90, 0.22)
    blank_vid = build_segmented_video(tmp_path / "blank_3s.mp4", [(3.0, [], "en")], roi_box=roi)
    srt_items, segs = scan_video_to_srt(str(blank_vid), roi=roi, language="en")
    assert srt_items == []
    assert segs == []

    # Tiny ROI: 0.055 * 720 = 40px (< 48px height)
    tiny_roi = (0.10, 0.85, 0.80, 0.055)
    tiny_vid = build_segmented_video(
        tmp_path / "tiny_roi.mp4",
        [(2.5, ["Tiny strip 99"], "en")],
        roi_box=tiny_roi,
        font_size=22,
    )
    _, tiny_segs = scan_video_to_srt(str(tiny_vid), roi=tiny_roi, language="en")
    assert len(tiny_segs) == 1
    assert normalized_text_similarity(tiny_segs[0].text, "Tiny strip 99") >= 0.85


def test_t2_persistent_static_watermark_does_not_swallow_cues(tmp_path):
    from videotrans.ocr import scan_video_to_srt

    roi = (0.05, 0.72, 0.90, 0.24)
    vid = build_segmented_video(
        tmp_path / "watermark.mp4",
        [
            (2.5, ["First subtitle alpha line"], "en"),
            (2.5, ["Second subtitle beta line"], "en"),
        ],
        roi_box=roi,
        watermark="LOGO HD",
        font_size=30,
    )
    _, segs = scan_video_to_srt(str(vid), roi=roi, language="en", coarse_interval_ms=500)
    assert len(segs) == 2, f"Static watermark swallowed or fragmented cue transitions: {segs}"
    assert "First" in segs[0].text and "alpha" in segs[0].text
    assert "Second" in segs[1].text and "beta" in segs[1].text
    assert abs(segs[1].start_ms - 2500) <= 250


def test_t2_checkpoint_resume_matches_uninterrupted_within_250ms(tmp_path):
    from videotrans.ocr import load_checkpoint, scan_video_to_srt, is_compatible, video_fingerprint

    roi = (0.05, 0.74, 0.90, 0.22)
    vid = build_segmented_video(
        tmp_path / "resume_8s.mp4",
        [
            (0.5, [], "en"),
            (2.5, ["Segment Alpha"], "en"),
            (2.5, ["Segment Beta"], "en"),
            (2.0, ["Segment Gamma"], "en"),
            (0.5, [], "en"),
        ],
        roi_box=roi,
    )

    _, baseline_segs = scan_video_to_srt(str(vid), roi=roi, language="en", coarse_interval_ms=500)
    assert len(baseline_segs) == 3

    ckpt_path = tmp_path / "ocr_ckpt.json"
    stop_flag = False

    def on_prog(ev: dict[str, Any]) -> None:
        nonlocal stop_flag
        if ev.get("timestamp_ms", 0) >= 4500:
            stop_flag = True

    _, partial_segs = scan_video_to_srt(
        str(vid),
        roi=roi,
        language="en",
        coarse_interval_ms=500,
        checkpoint_path=str(ckpt_path),
        stop_event=lambda: stop_flag,
        on_progress=on_prog,
    )
    assert ckpt_path.is_file(), "Checkpoint file was not persisted on stop"
    cp = load_checkpoint(str(ckpt_path))
    assert cp is not None
    assert cp.last_processed_ms >= 3500
    assert len(partial_segs) < 3 or partial_segs[-1].end_ms < baseline_segs[-1].end_ms

    # Resume from same checkpoint without stop_event
    _, resumed_segs = scan_video_to_srt(
        str(vid),
        roi=roi,
        language="en",
        coarse_interval_ms=500,
        checkpoint_path=str(ckpt_path),
    )
    assert len(resumed_segs) == len(baseline_segs), (
        f"Expected {len(baseline_segs)} segments after resume (no duplicates), got {resumed_segs}"
    )
    for res_s, base_s in zip(resumed_segs, baseline_segs):
        assert normalized_text_similarity(res_s.text, base_s.text) >= 0.90
        assert abs(res_s.start_ms - base_s.start_ms) <= 250
        assert abs(res_s.end_ms - base_s.end_ms) <= 250


def test_t2_mid_scan_cancel_via_api_kills_ffmpeg_promptly(tmp_path, monkeypatch):
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.core.job_manager import JobManager, run_prepare_review
    from videotrans.core.proc_registry import has_active_procs

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        roi = [0.05, 0.74, 0.90, 0.22]
        vid = build_segmented_video(
            tmp_path / "cancel_12s.mp4",
            [
                (3.0, ["Cancel test cue one"], "en"),
                (3.0, ["Cancel test cue two"], "en"),
                (3.0, ["Cancel test cue three"], "en"),
                (3.0, ["Cancel test cue four"], "en"),
            ],
            with_audio=False,
            roi_box=tuple(roi),
        )

        jm = JobManager(runner=run_prepare_review)
        app = create_app(
            job_manager=jm,
            upload_dir=iso["uploads_dir"],
            task_params_builder=functools.partial(
                build_task_params,
                output_dir=iso["output_dir"],
                temp_dir=iso["temp_dir"],
            ),
        )
        with TestClient(app) as client:
            with vid.open("rb") as f:
                media_id = client.post("/api/media", files={"file": ("cancel_12s.mp4", f, "video/mp4")}).json()["id"]

            resp = client.post(
                "/api/jobs",
                json={
                    "mediaId": media_id,
                    "jobType": "asr",
                    "options": {
                        "subtitleSource": "video_ocr",
                        "ocrRoi": roi,
                        "ocrRoiConfirmed": True,
                        "sourceLanguage": "en",
                        "targetLanguage": "zh-cn",
                        "recognType": 1,
                    },
                },
            )
            assert resp.status_code == 202, resp.text
            job_id = resp.json()["id"]

            # Wait for OCR recognition stage / ocr_progress (short fallback if prepare stalls)
            deadline = time.monotonic() + 10.0
            prep_fallback_deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                snap = client.get(f"/api/jobs/{job_id}").json()
                if snap["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                has_ocr_prog = any(
                    (ev.get("details") or {}).get("source_type") == "ocr_progress"
                    or ev.get("source_type") == "ocr_progress"
                    for ev in snap.get("events", [])
                )
                if has_ocr_prog or snap.get("stage") == "recogn":
                    break
                if (
                    snap["status"] == "running"
                    and len(snap.get("events", [])) >= 2
                    and time.monotonic() >= prep_fallback_deadline
                ):
                    break
                time.sleep(0.05)

            t_cancel = time.monotonic()
            c_resp = client.post(f"/api/jobs/{job_id}/cancel")
            assert c_resp.status_code == 200, c_resp.text
            final = _wait_for_job(client, job_id, timeout=5.0)
            cancel_elapsed = time.monotonic() - t_cancel

            assert final["status"] == "cancelled", f"Expected cancelled, got {final}"
            assert cancel_elapsed < 5.0
            assert has_active_procs(job_id) is False
    finally:
        restore_runtime_config(iso)


def test_t2_reject_unconfirmed_or_invalid_roi_and_bypass_audio_hash_dedup(tmp_path, monkeypatch):
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params
    from videotrans.core.content_hash import compute_content_hash
    from videotrans.core.job_manager import JobManager, run_prepare_review
    from videotrans.core.project_store import create_project
    from videotrans.task.orchestrator import TaskRequest, TaskStatus, run_staged_asr

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        vid = build_segmented_video(
            tmp_path / "dedup_bypass.mp4",
            [(2.5, ["Actual Video OCR Text"], "en")],
            with_audio=True,
            roi_box=(0.05, 0.74, 0.90, 0.22),
        )

        # 1. API rejects unconfirmed or out-of-bounds ROI for video_ocr
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
                media_id = client.post("/api/media", files={"file": ("dedup_bypass.mp4", f, "video/mp4")}).json()["id"]

            r_unconf = client.post(
                "/api/jobs",
                json={
                    "mediaId": media_id,
                    "jobType": "asr",
                    "options": {
                        "subtitleSource": "video_ocr",
                        "ocrRoi": [0.05, 0.74, 0.90, 0.22],
                        "ocrRoiConfirmed": False,
                        "sourceLanguage": "en",
                    },
                },
            )
            assert r_unconf.status_code == 400, f"Unconfirmed ROI should be rejected with 400: {r_unconf.text}"

            r_bad_roi = client.post(
                "/api/jobs",
                json={
                    "mediaId": media_id,
                    "jobType": "asr",
                    "options": {
                        "subtitleSource": "video_ocr",
                        "ocrRoi": [0.8, 0.8, 0.5, 0.5],
                        "ocrRoiConfirmed": True,
                        "sourceLanguage": "en",
                    },
                },
            )
            assert r_bad_roi.status_code == 400, f"Out-of-bounds ROI should be rejected with 400: {r_bad_roi.text}"

        # 2. Audio-hash dedup MUST NOT reuse audio_asr transcript for video_ocr
        content_hash = compute_content_hash(vid)
        create_project(
            project_id="proj_audio_asr",
            name="Audio Asr Prior",
            audio_hash=content_hash,
            stage=2,
            status="completed",
            state={
                "segments": [
                    {
                        "id": 1,
                        "startTime": "00:00:00,000",
                        "endTime": "00:00:02,500",
                        "sourceText": "STALE AUDIO ASR TRANSCRIPT",
                        "text": "STALE AUDIO ASR TRANSCRIPT",
                    }
                ]
            },
        )
        proj_ocr = create_project(project_id="proj_video_ocr", name="Video Ocr Target")

        cache_dir = iso["temp_dir"] / "ocr_dedup_cache"
        target_dir = iso["output_dir"] / "ocr_dedup_out"
        req = TaskRequest({
            "name": vid.as_posix(),
            "target_dir": target_dir.as_posix(),
            "cache_folder": cache_dir.as_posix(),
            "source_language_code": "en",
            "target_language_code": "zh-cn",
            "recogn_type": 1,
            "subtitle_source": "video_ocr",
            "ocr_roi": (0.05, 0.74, 0.90, 0.22),
            "ocr_roi_confirmed": True,
            "project_id": proj_ocr["id"],
        })
        res = run_staged_asr(req)
        assert res.status == TaskStatus.SUCCEEDED, res.failure
        texts = " ".join(s.get("sourceText", "") for s in res.segments)
        assert "STALE AUDIO ASR" not in texts
        assert "actual video ocr" in texts.lower()
    finally:
        restore_runtime_config(iso)


def test_t2_f11_single_char_transition_vs_identical_repeat_after_gap(tmp_path):
    from videotrans.ocr import scan_video_to_srt

    roi = (0.05, 0.74, 0.90, 0.22)
    # 1. Single-character change ("Part A" -> "Part B") must NOT be swallowed by visual dedup
    single_char_vid = build_segmented_video(
        tmp_path / "part_ab.mp4",
        [
            (2.0, ["Part A"], "en"),
            (2.0, ["Part B"], "en"),
        ],
        roi_box=roi,
    )
    _, segs_ab = scan_video_to_srt(str(single_char_vid), roi=roi, language="en", coarse_interval_ms=500)
    assert len(segs_ab) == 2, f"Single-character transition Part A -> Part B was dropped: {segs_ab}"
    assert "A" in segs_ab[0].text and "B" in segs_ab[1].text
    assert abs(segs_ab[1].start_ms - 2000) <= 250

    # 2. Identical text separated by > grace_gap_ms empty gap must stay 2 distinct segments
    repeat_vid = build_segmented_video(
        tmp_path / "repeat_gap.mp4",
        [
            (2.0, ["Repeated subtitle cue"], "en"),
            (1.5, [], "en"),
            (2.0, ["Repeated subtitle cue"], "en"),
        ],
        roi_box=roi,
    )
    _, segs_rep = scan_video_to_srt(str(repeat_vid), roi=roi, language="en", coarse_interval_ms=500)
    assert len(segs_rep) == 2, f"Repeated text separated by 1.5s gap merged into 1 segment: {segs_rep}"
    assert segs_rep[0].end_ms <= 2250
    assert segs_rep[1].start_ms >= 3250


# ---------------------------------------------------------------------------
# Tier 4: Video OCR Workload & Resource Efficiency Benchmark
# ---------------------------------------------------------------------------


def test_t4_video_ocr_resource_benchmark_memory_speed_dedup(tmp_path):
    """Run 10s multi-segment 720p scan in a fresh subprocess after full video-OCR warm-up.

    Measures incremental PeakWorkingSetSize (peak_after - peak_before) < 50.0 MB,
    tracemalloc peak Python allocations < 20.0 MB, wall-clock <= 15.0s (<= 1.5x real-time),
    and deduplication savings across static coarse frames.
    """
    roi = (0.05, 0.74, 0.90, 0.22)
    warmup_vid = build_segmented_video(
        tmp_path / "warmup_2_5s.mp4",
        [
            (0.5, [], "en"),
            (1.0, ["Warmup longest chapter caption text"], "en"),
            (1.0, ["Warmup second cue line", "Warmup line two"], "en"),
        ],
        roi_box=roi,
    )
    vid = build_segmented_video(
        tmp_path / "bench_10s.mp4",
        [
            (1.0, [], "en"),
            (2.5, ["Opening scene morning greeting"], "en"),
            (2.5, ["Second chapter mountain journey"], "en"),
            (1.0, [], "en"),
            (1.5, ["Third part ocean breeze"], "en"),
            (1.5, ["Final credits roll tonight"], "en"),
        ],
        roi_box=roi,
    )

    script = (
        "import gc, json, os, time, tracemalloc, psutil\n"
        "from videotrans.ocr import scan_video_to_srt\n"
        f"scan_video_to_srt({str(warmup_vid)!r}, roi={roi!r}, language='en', device='cpu', coarse_interval_ms=500, use_angle_cls=False, paragraph_mode='preserve_lines')\n"
        "gc.collect()\n"
        "proc = psutil.Process(os.getpid())\n"
        "mi_before = proc.memory_info()\n"
        "rss_warm = mi_before.rss\n"
        "peak_before = getattr(mi_before, 'peak_wset', rss_warm)\n"
        "tracemalloc.start()\n"
        "events = []\n"
        "t0 = time.perf_counter()\n"
        f"srt, segs = scan_video_to_srt({str(vid)!r}, roi={roi!r}, language='en', device='cpu', coarse_interval_ms=500, on_progress=events.append)\n"
        "elapsed = time.perf_counter() - t0\n"
        "_, tm_peak = tracemalloc.get_traced_memory()\n"
        "tracemalloc.stop()\n"
        "mi_after = proc.memory_info()\n"
        "rss_after = mi_after.rss\n"
        "peak_after = getattr(mi_after, 'peak_wset', rss_after)\n"
        "print(json.dumps({\n"
        "  'elapsed': elapsed,\n"
        "  'rss_warm_mb': rss_warm / (1024**2),\n"
        "  'rss_after_mb': rss_after / (1024**2),\n"
        "  'delta_rss_mb': (rss_after - rss_warm) / (1024**2),\n"
        "  'peak_before_mb': peak_before / (1024**2),\n"
        "  'peak_after_mb': peak_after / (1024**2),\n"
        "  'delta_peak_mb': (peak_after - peak_before) / (1024**2),\n"
        "  'tracemalloc_peak_mb': tm_peak / (1024**2),\n"
        "  'ocr_calls': events[-1]['ocr_calls'] if events else -1,\n"
        "  'skipped_frames': events[-1].get('skipped_frames') if events else None,\n"
        "  'segments': len(segs),\n"
        "}))\n"
    )
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=105,
        check=True,
    )
    metrics = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"[test_t4] metrics={json.dumps(metrics)}")

    assert metrics["segments"] == 4, f"Expected 4 segments in benchmark clip: {metrics}"
    if metrics.get("skipped_frames") is not None:
        assert metrics["skipped_frames"] > 0, f"Expected dedup to skip static coarse frames: {metrics}"
    assert metrics["ocr_calls"] <= 25, f"Expected dedup to keep total OCR calls bounded (<= 25): {metrics}"
    assert metrics["elapsed"] <= 15.0, (
        f"10s video OCR exceeded 1.5x real-time budget (15.0s): {metrics['elapsed']:.2f}s | {metrics}"
    )
    assert metrics["delta_peak_mb"] < 50.0, (
        f"Incremental scan peak_wset delta exceeded 50 MB: {metrics['delta_peak_mb']:.2f} MB | {metrics}"
    )
    assert metrics["tracemalloc_peak_mb"] < 20.0, (
        f"Peak Python allocations exceeded 20 MB: {metrics['tracemalloc_peak_mb']:.2f} MB | {metrics}"
    )
