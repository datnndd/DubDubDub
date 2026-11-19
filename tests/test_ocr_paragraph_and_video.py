# -*- coding: utf-8 -*-
"""Comprehensive R4 paragraph OCR & R5 video OCR tests + resource benchmark.

Covers:
- Pure-logic unit tests for row-band clustering, 2-column gutter detection,
  CJK/Latin joining (`preserve_lines` & `unwrap`), `is_same_subtitle_text`,
  bounded `SegmentBuilder` observations (`<= 16`), `on_dedup_skip`, watermark
  stripping, and `get_cached_provider` / `release_cached_providers`.
- Real `PaddleOCR` + real `ffmpeg` tests (no engine mocks) for 4-line EN
  paragraphs, 3-line ZH paragraphs without spurious ASCII spaces, jittered
  same-line reading order, 2-column layout, `en -> zh-cn -> en` language
  switching, `select_device("cuda")` before `recognize()` without `shm.dll`
  crash, odd-height ROI H.264 multi-segment video OCR with `<= 250ms` boundary
  accuracy, and a subprocess resource benchmark (elapsed time, CPU time, and
  `< 50 MB` RSS / PeakWorkingSet delta).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from videotrans.ocr import (
    OcrConfig,
    OcrLine,
    OcrResult,
    SegmentBuilder,
    get_cached_provider,
    join_ordered_lines,
    order_lines,
    release_cached_providers,
)
from videotrans.ocr._text import is_same_subtitle_text


# ---------------------------------------------------------------------------
# Pure-logic Unit Tests (Row Clustering, Columns, SegmentBuilder, Provider Cache)
# ---------------------------------------------------------------------------


def test_order_and_join_jittered_sameline_and_two_column():
    # 2 rows with 4px vertical jitter + wide horizontal gap must stay row-by-row
    jittered = [
        OcrLine(text="Right 1", confidence=0.9, box=(480, 36, 100, 28)),
        OcrLine(text="Left 1", confidence=0.9, box=(40, 40, 100, 28)),
        OcrLine(text="Right 2", confidence=0.9, box=(480, 96, 100, 28)),
        OcrLine(text="Left 2", confidence=0.9, box=(40, 100, 100, 28)),
    ]
    assert join_ordered_lines(jittered, paragraph_mode="preserve_lines") == "Left 1 Right 1\nLeft 2 Right 2"

    # 3+ rows with consistent vertical gutter must read column-by-column
    two_col = [
        OcrLine(text="Col2 A", confidence=0.9, box=(520, 30, 120, 28)),
        OcrLine(text="Col1 A", confidence=0.9, box=(40, 30, 120, 28)),
        OcrLine(text="Col2 B", confidence=0.9, box=(520, 86, 120, 28)),
        OcrLine(text="Col1 B", confidence=0.9, box=(40, 86, 120, 28)),
        OcrLine(text="Col2 C", confidence=0.9, box=(520, 142, 120, 28)),
        OcrLine(text="Col1 C", confidence=0.9, box=(40, 142, 120, 28)),
    ]
    ordered = order_lines(two_col)
    assert [ln.text for ln in ordered] == [
        "Col1 A",
        "Col1 B",
        "Col1 C",
        "Col2 A",
        "Col2 B",
        "Col2 C",
    ]
    assert join_ordered_lines(two_col, paragraph_mode="preserve_lines") == (
        "Col1 A\nCol1 B\nCol1 C\nCol2 A\nCol2 B\nCol2 C"
    )


def test_join_ordered_lines_unwrap_cjk_and_hyphenation():
    en_hyphen = [
        OcrLine(text="This is a hyphen-", confidence=0.9, box=(10, 10, 200, 24)),
        OcrLine(text="ated paragraph.", confidence=0.9, box=(10, 42, 200, 24)),
    ]
    assert join_ordered_lines(en_hyphen, paragraph_mode="unwrap") == "This is a hyphenated paragraph."

    zh_lines = [
        OcrLine(text="第一行中文", confidence=0.9, box=(10, 10, 200, 28)),
        OcrLine(text="第二行中文", confidence=0.9, box=(10, 48, 200, 28)),
    ]
    assert join_ordered_lines(zh_lines, paragraph_mode="preserve_lines") == "第一行中文\n第二行中文"
    assert join_ordered_lines(zh_lines, paragraph_mode="unwrap") == "第一行中文第二行中文"


def test_is_same_subtitle_text_guards_numbers_and_single_word_swaps():
    assert is_same_subtitle_text("Wait 10s", "Wait 20s", 0.85) is False
    assert is_same_subtitle_text("Part A", "Part B", 0.85) is False
    assert is_same_subtitle_text("Full HD 1080p line one", "Full HD 1080p line two", 0.85) is False
    assert is_same_subtitle_text("Benchmark subtitle alpha line", "Benchmark subtitle beta line", 0.85) is False
    assert is_same_subtitle_text("Hello world!", "Hello world", 0.85) is True


def test_segment_builder_bounded_observations_dedup_skip_and_watermark():
    cfg = OcrConfig(min_stable_samples=2, grace_gap_ms=400)
    sb = SegmentBuilder(cfg)

    # Low-confidence (<0.90) candidate followed by dedup skip must open segment
    sb.observe(1000, OcrResult(text="Sub one", confidence=0.80))
    sb.on_dedup_skip(1500)
    assert sb._state == "ACTIVE"

    # Feed 30 more samples -> observations list must stay bounded <= 16 while sample_count tracks all
    for i in range(2, 32):
        sb.observe(1000 + i * 200, OcrResult(text="Sub one", confidence=0.95 if i == 2 else 0.85))
    assert len(sb._open["observations"]) <= 16
    assert sb._open["sample_count"] == 32

    # Grace-gap expiry via on_dedup_skip
    sb.observe(7600, OcrResult(text="", confidence=0.0))
    sb.on_dedup_skip(7900)  # 7900 - 7200 = 700 > 400 -> closes at 7200
    assert sb._state == "EMPTY"
    segs = sb.finalize()
    assert len(segs) == 1
    assert segs[0].samples == 32
    assert segs[0].end_ms == 7200

    # Persistent watermark (>85% observations) stripped when multiple cues exist
    sb_wm = SegmentBuilder(OcrConfig(min_stable_samples=1, grace_gap_ms=300))
    for ts in (0, 500, 1000):
        sb_wm.observe(ts, OcrResult(text="LOGO HD\nFirst alpha cue line", confidence=0.95))
    for ts in (1500, 2000, 2500):
        sb_wm.observe(ts, OcrResult(text="LOGO HD\nSecond beta cue line", confidence=0.95))
    wm_segs = sb_wm.finalize()
    assert len(wm_segs) == 2
    assert wm_segs[0].text == "First alpha cue line"
    assert wm_segs[1].text == "Second beta cue line"

    # Single-cue static video must NOT be wiped out by watermark filter
    sb_single = SegmentBuilder(OcrConfig(min_stable_samples=1))
    for ts in (0, 500, 1000, 1500, 2000):
        sb_single.observe(ts, OcrResult(text="Single persistent subtitle", confidence=0.95))
    single_segs = sb_single.finalize()
    assert len(single_segs) == 1
    assert single_segs[0].text == "Single persistent subtitle"


def test_get_cached_provider_bounded_and_monkeypatch_bypass(monkeypatch):
    import videotrans.ocr as ocr_pkg
    from videotrans.ocr._paddle import PaddleOcrProvider

    release_cached_providers()
    try:
        expected_auto_dev = PaddleOcrProvider(device="auto").select_device("auto")
        p1 = get_cached_provider(device="cpu")
        p2 = get_cached_provider(device="cpu")
        p3 = get_cached_provider(
            provider=ocr_pkg.PADDLE_OCR,
            device="CPU",
            use_angle_cls=False,
            paragraph_mode="preserve_lines",
            cpu_threads=None,
        )
        p_auto = get_cached_provider(device="auto")
        assert p1 is p2
        assert p1 is p3
        if expected_auto_dev == "cpu":
            assert p_auto is p1
        else:
            assert p_auto is get_cached_provider(device=expected_auto_dev)

        # Monkeypatch bypass
        monkeypatch.setattr(ocr_pkg, "get_provider", lambda **kw: ("mocked", kw))
        assert get_cached_provider(device="cpu")[0] == "mocked"
    finally:
        release_cached_providers()


def test_scanner_forward_brackets_when_coarse_frame_shifted_ahead():
    import numpy as np
    from videotrans.ocr import OcrScanner

    def make_roi(gray):
        return np.full((32, 128, 3), gray, dtype=np.uint8)

    class GrayProvider:
        def recognize(self, crop, language=None):
            val = int(round(float(np.mean(crop))))
            if val == 180:
                return OcrResult(text="Unaligned cue two", confidence=0.95)
            return OcrResult(text="", confidence=0.0)

    # True timeline: cue two active in [1200..6240)
    def true_frame_at(ts_ms):
        return make_roi(180 if 1200 <= ts_ms < 6240 else 20)

    # Simulated shifted coarse stream (6000 delivers the 6240 blank frame!)
    coarse_frames = [
        (0, make_roi(20)),
        (500, make_roi(20)),
        (1000, make_roi(20)),
        (1500, make_roi(180)),
        (5500, make_roi(180)),
        (6000, make_roi(20)),
        (6500, make_roi(20)),
        (7000, make_roi(20)),
    ]
    cfg = OcrConfig(
        coarse_interval_ms=500,
        boundary_interval_ms=150,
        min_stable_samples=1,
        grace_gap_ms=400,
    )
    scanner = OcrScanner(
        GrayProvider(),
        cfg,
        frames_are_cropped=True,
        frame_fetcher=true_frame_at,
    )
    segs = scanner.scan(coarse_frames, duration_ms=7240)
    assert len(segs) == 1
    assert segs[0].text == "Unaligned cue two"
    assert abs(segs[0].start_ms - 1200) <= 250
    assert abs(segs[0].end_ms - 6240) <= 250


def test_scanner_finalize_tail_refinement_and_monotonic_timestamps():
    import numpy as np
    from videotrans.ocr import OcrScanner

    def make_roi(gray):
        return np.full((32, 128, 3), gray, dtype=np.uint8)

    class GrayProvider:
        def recognize(self, crop, language=None):
            val = int(round(float(np.mean(crop))))
            if val == 180:
                return OcrResult(text="Tail first line", confidence=0.95)
            if val == 230:
                return OcrResult(text="Tail second line", confidence=0.95)
            return OcrResult(text="", confidence=0.0)

    cfg = OcrConfig(
        coarse_interval_ms=500,
        boundary_interval_ms=150,
        min_stable_samples=2,
        grace_gap_ms=400,
    )

    # 1. Tail blank transition at 5840ms in 6000ms clip
    def tail_blank_at(ts_ms):
        return make_roi(180 if 1000 <= ts_ms < 5840 else 20)

    coarse_blank = [(ts, tail_blank_at(ts)) for ts in range(0, 6000, 500)]
    scanner_blank = OcrScanner(
        GrayProvider(),
        cfg,
        frames_are_cropped=True,
        frame_fetcher=tail_blank_at,
    )
    observed_blank = []
    orig_obs = scanner_blank._builder.observe

    def track_obs(ts_ms, res):
        observed_blank.append(int(ts_ms))
        return orig_obs(ts_ms, res)

    scanner_blank._builder.observe = track_obs
    segs_blank = scanner_blank.scan(coarse_blank, duration_ms=6000)
    assert observed_blank == sorted(observed_blank)
    assert len(observed_blank) == len(set(observed_blank))
    assert len(segs_blank) == 1
    assert abs(segs_blank[0].start_ms - 1000) <= 250
    assert abs(segs_blank[0].end_ms - 5840) <= 250

    # 2. Tail switch [1000..5600 "Tail first line", 5600..6000 "Tail second line"]
    def tail_switch_at(ts_ms):
        if ts_ms < 1000:
            return make_roi(20)
        if ts_ms < 5600:
            return make_roi(180)
        return make_roi(230)

    coarse_switch = [(ts, tail_switch_at(ts)) for ts in range(0, 6000, 500)]
    scanner_switch = OcrScanner(
        GrayProvider(),
        cfg,
        frames_are_cropped=True,
        frame_fetcher=tail_switch_at,
    )
    observed_switch = []
    orig_obs_sw = scanner_switch._builder.observe

    def track_obs_sw(ts_ms, res):
        observed_switch.append(int(ts_ms))
        return orig_obs_sw(ts_ms, res)

    scanner_switch._builder.observe = track_obs_sw
    segs_switch = scanner_switch.scan(coarse_switch, duration_ms=6000)
    assert observed_switch == sorted(observed_switch)
    assert len(observed_switch) == len(set(observed_switch))
    assert len(segs_switch) == 2
    assert segs_switch[0].text == "Tail first line"
    assert segs_switch[1].text == "Tail second line"
    assert abs(segs_switch[0].end_ms - 5600) <= 250
    assert abs(segs_switch[1].start_ms - 5600) <= 250
    assert segs_switch[1].end_ms == 6000
    assert segs_switch[1].end_ms > segs_switch[1].start_ms

    # 3. frame_fetcher=None preserves tail extension to duration_ms
    scanner_no_fetch = OcrScanner(
        GrayProvider(),
        cfg,
        frames_are_cropped=True,
        frame_fetcher=None,
    )
    segs_no_fetch = scanner_no_fetch.scan(
        [(ts, make_roi(180 if ts >= 1000 else 20)) for ts in range(0, 6000, 500)],
        duration_ms=6000,
    )
    assert len(segs_no_fetch) == 1
    assert segs_no_fetch[0].end_ms == 6000


# ---------------------------------------------------------------------------
# Real PaddleOCR + Real FFmpeg Integration & Resource Benchmark Tests
# ---------------------------------------------------------------------------


def _require_real_ocr():
    from tests.e2e.media_synth import require_ocr_env
    return require_ocr_env()


def test_real_paddle_paragraphs_reading_order_lang_switch_and_device_safety():
    _require_real_ocr()
    import paddle
    from tests.e2e.media_synth import (
        make_jittered_sameline_bgr,
        make_paragraph_bgr,
        make_two_column_bgr,
        normalized_text_similarity,
    )
    from videotrans.ocr._paddle import PaddleOcrProvider

    provider = PaddleOcrProvider(device="auto")
    expected_dev = "gpu" if paddle.is_compiled_with_cuda() else "cpu"

    # select_device("cuda") BEFORE recognize() must not raise WinError 127 (shm.dll)
    assert provider.select_device("cuda") == expected_dev
    assert provider.select_device("auto") == expected_dev

    # 1. 4-line EN paragraph (0 dropped lines)
    en_lines = [
        "Artificial intelligence is transforming how creators localize video content",
        "across multiple languages. Modern speech recognition and optical character",
        "recognition pipelines must handle full paragraphs accurately without dropping",
        "lines or scrambling the natural reading order of the text.",
    ]
    en_img = make_paragraph_bgr(en_lines, lang="en", font_size=32, size=(1160, 280), line_spacing=54)
    res_en = provider.recognize(en_img, language="en")
    assert res_en.device == expected_dev
    assert len([ln for ln in res_en.text.splitlines() if ln.strip()]) == 4
    assert normalized_text_similarity(res_en.text, "\n".join(en_lines)) >= 0.95

    # 2. Language switch en -> zh-cn (3-line ZH paragraph, 0 dropped, no ASCII space)
    zh_lines = [
        "人工智能正在改变创作者将视频内容本地化到多种语言的方式。",
        "现代语音识别和光学字符识别流程必须准确处理完整段落，",
        "而不能丢失行内容、破坏阅读顺序或在换行处插入多余空格。",
    ]
    zh_img = make_paragraph_bgr(zh_lines, lang="zh-cn", font_size=32, size=(1120, 240), line_spacing=56)
    res_zh = provider.recognize(zh_img, language="zh-cn")
    assert len([ln for ln in res_zh.text.splitlines() if ln.strip()]) == 3
    assert " " not in res_zh.text
    assert normalized_text_similarity(res_zh.text, "\n".join(zh_lines)) >= 0.95

    # 3. Switch back zh-cn -> en
    res_en2 = provider.recognize(en_img, language="en")
    assert normalized_text_similarity(res_en2.text, "\n".join(en_lines)) >= 0.95

    # 4. Jittered same-line reading order
    jit_img, jit_rows = make_jittered_sameline_bgr()
    res_jit = provider.recognize(jit_img, language="en")
    jit_out = [ln.strip() for ln in res_jit.text.splitlines() if ln.strip()]
    assert len(jit_out) == 2
    for row_txt, (l_exp, r_exp) in zip(jit_out, jit_rows):
        assert row_txt.lower().find("left") < row_txt.lower().find("right")
        assert normalized_text_similarity(row_txt, f"{l_exp} {r_exp}") >= 0.90

    # 5. Two-column reading order
    col_img, _, _ = make_two_column_bgr()
    res_col = provider.recognize(col_img, language="en")
    col_lines = [ln.strip() for ln in res_col.text.splitlines() if ln.strip()]
    assert len(col_lines) == 6
    col1_idx = [i for i, ln in enumerate(col_lines) if "one" in ln.lower()]
    col2_idx = [i for i, ln in enumerate(col_lines) if "two" in ln.lower()]
    assert len(col1_idx) == 3 and len(col2_idx) == 3
    assert max(col1_idx) < min(col2_idx)


def test_real_video_ocr_odd_roi_and_resource_benchmark(tmp_path):
    _require_real_ocr()
    from tests.e2e.media_synth import build_segmented_video, normalized_text_similarity

    # Odd ROI height on 1280x720: round(0.72 * 720)=518, round((0.72+0.24)*720)=691 -> 173px
    odd_roi = (0.05, 0.72, 0.90, 0.24)
    vid = build_segmented_video(
        tmp_path / "bench_odd_roi.mp4",
        [
            (1.0, [], "en"),
            (2.5, ["Odd ROI line one", "Odd ROI line two"], "en"),
            (2.5, ["Second cue paragraph line"], "en"),
            (1.0, [], "en"),
            (1.5, ["Wait 10s"], "en"),
            (1.5, ["Wait 20s"], "en"),
        ],
        roi_box=odd_roi,
        font_size=30,
    )

    script = (
        "import gc, json, os, time, tracemalloc, psutil\n"
        "from tests.e2e.media_synth import make_paragraph_bgr\n"
        "from videotrans.ocr import get_cached_provider, scan_video_to_srt\n"
        "p = get_cached_provider(device='cpu')\n"
        "warm1 = make_paragraph_bgr(['Warmup paragraph line alpha', 'Warmup paragraph line beta'], lang='en', size=(1152, 173))\n"
        "warm2 = make_paragraph_bgr(['Wait 10s'], lang='en', size=(1152, 173))\n"
        "p.recognize(warm1, 'en')\n"
        "p.recognize(warm2, 'en')\n"
        "gc.collect()\n"
        "proc = psutil.Process(os.getpid())\n"
        "mi0 = proc.memory_info()\n"
        "ct0 = proc.cpu_times()\n"
        "peak0 = getattr(mi0, 'peak_wset', mi0.rss)\n"
        "tracemalloc.start()\n"
        "events = []\n"
        "t0 = time.perf_counter()\n"
        f"srt, segs = scan_video_to_srt({str(vid)!r}, roi={odd_roi!r}, language='en', device='cpu', coarse_interval_ms=500, on_progress=events.append)\n"
        "elapsed = time.perf_counter() - t0\n"
        "_, tm_peak = tracemalloc.get_traced_memory()\n"
        "tracemalloc.stop()\n"
        "mi1 = proc.memory_info()\n"
        "ct1 = proc.cpu_times()\n"
        "peak1 = getattr(mi1, 'peak_wset', mi1.rss)\n"
        "cpu_time = (ct1.user - ct0.user) + (ct1.system - ct0.system)\n"
        "print(json.dumps({\n"
        "  'elapsed': elapsed,\n"
        "  'cpu_time': cpu_time,\n"
        "  'delta_rss_mb': (mi1.rss - mi0.rss) / (1024**2),\n"
        "  'delta_peak_mb': (peak1 - peak0) / (1024**2),\n"
        "  'tracemalloc_peak_mb': tm_peak / (1024**2),\n"
        "  'ocr_calls': events[-1]['ocr_calls'] if events else -1,\n"
        "  'skipped_frames': events[-1].get('skipped_frames') if events else 0,\n"
        "  'device': events[-1].get('device') if events else None,\n"
        "  'segments': [{'start_ms': s.start_ms, 'end_ms': s.end_ms, 'text': s.text} for s in segs],\n"
        "  'srt_count': len(srt),\n"
        "}))\n"
    )
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
        check=True,
    )
    metrics = json.loads(proc.stdout.strip().splitlines()[-1])

    segs = metrics["segments"]
    assert len(segs) == 4, f"Expected 4 segments, got {segs}"
    assert metrics["srt_count"] == 4
    assert metrics["device"] == "cpu"
    assert metrics["skipped_frames"] > 0
    assert metrics["ocr_calls"] <= 25
    assert metrics["elapsed"] <= 12.0, f"Elapsed too high: {metrics}"
    assert metrics["cpu_time"] >= 0.0
    assert metrics["delta_peak_mb"] < 50.0, f"Peak memory delta >= 50MB: {metrics}"
    assert metrics["delta_rss_mb"] < 50.0, f"RSS delta >= 50MB: {metrics}"
    assert metrics["tracemalloc_peak_mb"] < 20.0, f"Tracemalloc peak >= 20MB: {metrics}"

    expected = [
        (1000, 3500, "Odd ROI line one\nOdd ROI line two"),
        (3500, 6000, "Second cue paragraph line"),
        (7000, 8500, "Wait 10s"),
        (8500, 10000, "Wait 20s"),
    ]
    for seg, (exp_s, exp_e, exp_txt) in zip(segs, expected):
        assert normalized_text_similarity(seg["text"], exp_txt) >= 0.90
        assert abs(seg["start_ms"] - exp_s) <= 250
        assert abs(seg["end_ms"] - exp_e) <= 250


@pytest.mark.parametrize(
    "timeline,expected",
    [
        (
            [
                (1.2, [], "en"),
                (2.52, ["Unaligned cue one"], "en"),
                (2.52, ["Unaligned cue two"], "en"),
                (1.0, [], "en"),
            ],
            [
                (1200, 3720, "Unaligned cue one"),
                (3720, 6240, "Unaligned cue two"),
            ],
        ),
        (
            [
                (1.2, [], "en"),
                (2.48, ["Boundary cue alpha"], "en"),
                (2.52, ["Boundary cue beta"], "en"),
                (1.0, [], "en"),
            ],
            [
                (1200, 3680, "Boundary cue alpha"),
                (3680, 6200, "Boundary cue beta"),
            ],
        ),
    ],
)
def test_real_video_ocr_non_500ms_aligned_25fps_boundaries(tmp_path, timeline, expected):
    _require_real_ocr()
    from tests.e2e.media_synth import build_segmented_video, normalized_text_similarity
    from videotrans.ocr import scan_video_to_srt

    odd_roi = (0.05, 0.72, 0.90, 0.24)
    vid = build_segmented_video(
        tmp_path / "non_aligned_25fps.mp4",
        timeline,
        roi_box=odd_roi,
        fps=25,
        font_size=30,
    )
    _srt, segs = scan_video_to_srt(
        str(vid),
        roi=odd_roi,
        language="en",
        device="cpu",
        coarse_interval_ms=500,
    )
    assert len(segs) == len(expected), f"Expected {len(expected)} segments, got {segs}"
    for seg, (exp_s, exp_e, exp_txt) in zip(segs, expected):
        assert normalized_text_similarity(seg.text, exp_txt) >= 0.90
        assert abs(seg.start_ms - exp_s) <= 250, f"Start drift > 250ms: {seg} vs {exp_s}"
        assert abs(seg.end_ms - exp_e) <= 250, f"End drift > 250ms: {seg} vs {exp_e}"


@pytest.mark.parametrize(
    "timeline,expected",
    [
        (
            [
                (1.48, [], "en"),
                (2.08, ["Wait 10s"], "en"),
                (2.32, ["Wait 20s"], "en"),
                (0.12, [], "en"),
            ],
            [
                (1480, 3560, "Wait 10s"),
                (3560, 5880, "Wait 20s"),
            ],
        ),
        (
            [
                (1.0, [], "en"),
                (4.84, ["Tail blank test"], "en"),
                (0.16, [], "en"),
            ],
            [
                (1000, 5840, "Tail blank test"),
            ],
        ),
        (
            [
                (1.0, [], "en"),
                (4.6, ["Tail first line"], "en"),
                (0.4, ["Tail second line"], "en"),
            ],
            [
                (1000, 5600, "Tail first line"),
                (5600, 6000, "Tail second line"),
            ],
        ),
    ],
)
def test_real_video_ocr_tail_boundaries_25fps_odd_roi(tmp_path, timeline, expected):
    _require_real_ocr()
    from tests.e2e.media_synth import build_segmented_video, normalized_text_similarity
    from videotrans.ocr import scan_video_to_srt

    # Exact odd ROI (65, 519, 1151, 173) on 1280x720 H.264 25fps (dur=6000ms)
    odd_roi = (65 / 1280, 519 / 720, 1151 / 1280, 173 / 720)
    vid = build_segmented_video(
        tmp_path / "tail_boundary_25fps.mp4",
        timeline,
        roi_box=odd_roi,
        fps=25,
        font_size=30,
    )
    _srt, segs = scan_video_to_srt(
        str(vid),
        roi=odd_roi,
        language="en",
        device="auto",
        coarse_interval_ms=500,
    )
    assert len(segs) == len(expected), f"Expected {len(expected)} segments, got {segs}"
    for seg, (exp_s, exp_e, exp_txt) in zip(segs, expected):
        assert normalized_text_similarity(seg.text, exp_txt) >= 0.90
        assert seg.end_ms > seg.start_ms, f"Zero/negative duration segment: {seg}"
        assert abs(seg.start_ms - exp_s) <= 250, f"Start drift > 250ms: {seg} vs {exp_s}"
        assert abs(seg.end_ms - exp_e) <= 250, f"End drift > 250ms: {seg} vs {exp_e}"


def test_real_paddle_concurrent_recognize_en_and_zh():
    _require_real_ocr()
    import threading
    from tests.e2e.media_synth import make_paragraph_bgr, normalized_text_similarity

    release_cached_providers()
    try:
        provider = get_cached_provider(device="auto")
        en_lines = ["Concurrent OCR English line one", "Concurrent OCR English line two"]
        zh_lines = ["并发识别第一行中文", "并发识别第二行中文"]
        en_img = make_paragraph_bgr(en_lines, lang="en", font_size=32, size=(960, 180), line_spacing=54)
        zh_img = make_paragraph_bgr(zh_lines, lang="zh-cn", font_size=32, size=(960, 180), line_spacing=54)

        barrier = threading.Barrier(2)
        errors = []
        outputs = {"en": [], "zh-cn": []}

        def run_worker(lang_code, img):
            try:
                barrier.wait(timeout=15)
                for _ in range(3):
                    res = provider.recognize(img, language=lang_code)
                    outputs[lang_code].append(res.text)
            except Exception as exc:
                errors.append(exc)

        t_en = threading.Thread(target=run_worker, args=("en", en_img))
        t_zh = threading.Thread(target=run_worker, args=("zh-cn", zh_img))
        t_en.start()
        t_zh.start()
        t_en.join(timeout=60)
        t_zh.join(timeout=60)

        assert not errors
        assert len(outputs["en"]) == 3
        assert len(outputs["zh-cn"]) == 3
        for txt in outputs["en"]:
            assert normalized_text_similarity(txt, "\n".join(en_lines)) >= 0.90
        for txt in outputs["zh-cn"]:
            assert normalized_text_similarity(txt, "\n".join(zh_lines)) >= 0.90
    finally:
        release_cached_providers()

