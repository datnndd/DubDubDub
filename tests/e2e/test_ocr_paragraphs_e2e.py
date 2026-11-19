# -*- coding: utf-8 -*-
"""E2E Opaque-Box Tests for F10: R4 OCR Multi-Line Paragraph Extraction & Reading Order.

Uses REAL PaddleOcrProvider offline (HF_HUB_OFFLINE=1) against PIL-rendered
paragraphs and FFmpeg 720p videos. No OCR engine mocks.
"""

from __future__ import annotations

import functools
import re
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.e2e.media_synth import (
    build_segmented_video,
    isolate_runtime_env,
    make_blank_bgr,
    make_jittered_sameline_bgr,
    make_paragraph_bgr,
    make_tiny_roi_bgr,
    make_two_column_bgr,
    normalized_text_similarity,
    require_ocr_env,
    restore_runtime_config,
)

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _check_env():
    require_ocr_env()


@pytest.fixture(scope="module")
def shared_provider():
    from videotrans.ocr._paddle import PaddleOcrProvider

    return PaddleOcrProvider(device="auto")


# ---------------------------------------------------------------------------
# Tier 1: Primary Feature Coverage (R4 Multi-Line Paragraph & Stage 2 API)
# ---------------------------------------------------------------------------


def test_t1_en_four_line_paragraph_preserves_all_lines(shared_provider):
    from videotrans.ocr._types import OcrConfig

    cfg = OcrConfig()
    assert getattr(cfg, "use_angle_cls", None) is False
    assert getattr(cfg, "paragraph_mode", None) == "preserve_lines"

    expected_lines = [
        "Artificial intelligence is transforming how creators localize video content",
        "across multiple languages. Modern speech recognition and optical character",
        "recognition pipelines must handle full paragraphs accurately without dropping",
        "lines or scrambling the natural reading order of the text.",
    ]
    img = make_paragraph_bgr(expected_lines, lang="en", font_size=32, size=(1160, 280), line_spacing=54)
    res = shared_provider.recognize(img, language="en")

    out_lines = [ln.strip() for ln in res.text.splitlines() if ln.strip()]
    assert len(out_lines) == 4, f"Expected 4 lines preserved, got {len(out_lines)}: {res.text!r}"
    assert res.text.count("\n") == 3, f"Expected 3 newlines separating 4 lines, got: {res.text!r}"

    # Specifically verify middle lines 2 & 3 were NOT dropped
    assert normalized_text_similarity(out_lines[1], expected_lines[1]) >= 0.92
    assert normalized_text_similarity(out_lines[2], expected_lines[2]) >= 0.92
    assert normalized_text_similarity(res.text, "\n".join(expected_lines)) >= 0.95


def test_t1_zh_three_line_paragraph_no_spurious_cjk_spaces(shared_provider):
    expected_lines = [
        "人工智能正在改变创作者将视频内容本地化到多种语言的方式。",
        "现代语音识别和光学字符识别流程必须准确处理完整段落，",
        "而不能丢失行内容、破坏阅读顺序或在换行处插入多余空格。",
    ]
    img = make_paragraph_bgr(expected_lines, lang="zh-cn", font_size=32, size=(1120, 240), line_spacing=56)
    res = shared_provider.recognize(img, language="zh-cn")

    out_lines = [ln.strip() for ln in res.text.splitlines() if ln.strip()]
    assert len(out_lines) == 3, f"Expected 3 ZH lines preserved, got {len(out_lines)}: {res.text!r}"
    assert res.text.count("\n") == 2, f"Expected 2 newlines in 3-line ZH paragraph, got: {res.text!r}"

    # No spurious ASCII spaces inserted inside pure CJK lines or across CJK lines
    assert " " not in res.text, f"Spurious ASCII space found inside CJK output: {res.text!r}"
    assert normalized_text_similarity(out_lines[1], expected_lines[1]) >= 0.92
    assert normalized_text_similarity(res.text, "\n".join(expected_lines)) >= 0.95


def test_t1_mixed_en_zh_paragraph_accuracy(shared_provider):
    expected_lines = [
        "PaddleOCR 支持中文和 English 混合段落识别。",
        "Python 3.10 环境下的视频字幕提取保持 250ms 精度，",
        "Video OCR pipeline keeps every line in order.",
    ]
    img = make_paragraph_bgr(expected_lines, lang="zh-cn", font_size=32, size=(1120, 240), line_spacing=56)
    res = shared_provider.recognize(img, language="zh-cn")

    out_lines = [ln.strip() for ln in res.text.splitlines() if ln.strip()]
    assert len(out_lines) == 3, f"Expected 3 mixed lines, got {len(out_lines)}: {res.text!r}"
    assert normalized_text_similarity(res.text, "\n".join(expected_lines)) >= 0.90
    assert "PaddleOCR" in res.text
    assert "Video OCR" in res.text


def test_t1_single_provider_sequential_language_switch_en_to_zh_to_en():
    from videotrans.ocr._paddle import PaddleOcrProvider

    provider = PaddleOcrProvider(device="auto")
    en_lines = [
        "First English paragraph line one.",
        "Second English paragraph line two.",
    ]
    zh_lines = [
        "第一行中文测试字幕需要完整识别。",
        "第二行中文内容不能被英文模型损坏。",
    ]
    en_img = make_paragraph_bgr(en_lines, lang="en", font_size=32, size=(960, 180))
    zh_img = make_paragraph_bgr(zh_lines, lang="zh-cn", font_size=32, size=(960, 180))

    r_en1 = provider.recognize(en_img, language="en")
    assert normalized_text_similarity(r_en1.text, "\n".join(en_lines)) >= 0.93

    # Must switch engine to 'ch' instead of reusing 'en' engine
    r_zh = provider.recognize(zh_img, language="zh-cn")
    assert normalized_text_similarity(r_zh.text, "\n".join(zh_lines)) >= 0.90, (
        f"Language switch en->zh-cn corrupted output: {r_zh.text!r}"
    )

    # Switch back to 'en'
    r_en2 = provider.recognize(en_img, language="en")
    assert normalized_text_similarity(r_en2.text, "\n".join(en_lines)) >= 0.93


def test_t1_stage2_api_ocr_extract_preserves_multiline_paragraph(tmp_path, monkeypatch):
    from videotrans.api.app import create_app
    from videotrans.api.task_params import build_task_params

    iso = isolate_runtime_env(tmp_path, monkeypatch)
    try:
        expected_lines = [
            "Stage two interval OCR must keep",
            "all three paragraph lines intact",
            "when repairing a subtitle block.",
        ]
        roi = [0.05, 0.70, 0.90, 0.26]
        vid = build_segmented_video(
            tmp_path / "stage2_3line.mp4",
            [(2.5, expected_lines, "en")],
            roi_box=tuple(roi),
            font_size=28,
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
                up = client.post("/api/media", files={"file": ("stage2_3line.mp4", f, "video/mp4")})
            assert up.status_code == 201, up.text
            media_id = up.json()["id"]

            resp = client.post(
                "/api/ocr/extract",
                json={
                    "mediaId": media_id,
                    "startSec": 0.0,
                    "endSec": 2.5,
                    "roi": roi,
                    "language": "en",
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["ok"] is True
            assert body["entries"], "Expected at least one entry in /api/ocr/extract"

            top_lines = [ln.strip() for ln in body["text"].splitlines() if ln.strip()]
            entry_lines = [ln.strip() for ln in body["entries"][0]["text"].splitlines() if ln.strip()]
            assert len(top_lines) == 3, f"Top-level text lost lines: {body['text']!r}"
            assert len(entry_lines) == 3, f"Entry text lost lines: {body['entries'][0]['text']!r}"
            assert normalized_text_similarity(body["text"], "\n".join(expected_lines)) >= 0.90
    finally:
        restore_runtime_config(iso)


# ---------------------------------------------------------------------------
# Tier 2: Boundary & Corner Cases (R4 DLL Safety, Layout Geometry, Arrays)
# ---------------------------------------------------------------------------


def test_t2_windows_dll_and_device_select_before_recognize():
    import paddle
    from videotrans.ocr._paddle import PaddleOcrProvider

    provider = PaddleOcrProvider(device="auto")
    expected_device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"

    # Must not raise WinError 127 (shm.dll) when called before recognize()
    selected_auto = provider.select_device("auto")
    selected_cuda = provider.select_device("cuda")
    assert selected_auto == expected_device
    assert selected_cuda == expected_device

    img = make_paragraph_bgr(["Device selection safety check."], lang="en", size=(800, 140))
    res = provider.recognize(img, language="en")
    assert "Device selection" in res.text
    assert getattr(res, "device", None) == expected_device


def test_t2_jittered_sameline_preserves_left_to_right_reading_order(shared_provider):
    img, rows = make_jittered_sameline_bgr()
    res = shared_provider.recognize(img, language="en")

    out_lines = [ln.strip() for ln in res.text.splitlines() if ln.strip()]
    assert len(out_lines) == 2, f"Expected 2 visual rows despite 4px jitter, got: {res.text!r}"

    for out_row, (left_txt, right_txt) in zip(out_lines, rows):
        pos_left = out_row.lower().find("left")
        pos_right = out_row.lower().find("right")
        assert pos_left != -1 and pos_right != -1, f"Missing chunk in row {out_row!r}"
        assert pos_left < pos_right, f"Row order inverted due to vertical jitter: {out_row!r}"
        assert normalized_text_similarity(out_row, f"{left_txt} {right_txt}") >= 0.90


def test_t2_two_column_gutter_reads_column_first_without_interleaving(shared_provider):
    img, col1, col2 = make_two_column_bgr()
    res = shared_provider.recognize(img, language="en")

    lines = [ln.strip() for ln in res.text.splitlines() if ln.strip()]
    assert len(lines) == 6, f"Expected 6 lines across 2 columns, got {len(lines)}: {res.text!r}"

    # All 3 Col-1 lines must appear before all 3 Col-2 lines
    col1_positions = [i for i, ln in enumerate(lines) if "one" in ln.lower()]
    col2_positions = [i for i, ln in enumerate(lines) if "two" in ln.lower()]
    assert len(col1_positions) == 3 and len(col2_positions) == 3, (
        f"Columns mis-detected in: {lines!r}"
    )
    assert max(col1_positions) < min(col2_positions), (
        f"Columns interleaved instead of column-first order: {lines!r}"
    )


def test_t2_paragraph_mode_unwrap_vs_preserve_lines():
    from videotrans.ocr._paddle import PaddleOcrProvider

    en_lines = [
        "First line wraps smoothly",
        "into the second line here.",
    ]
    zh_lines = [
        "第一行中文自然衔接",
        "到第二行中文末尾。",
    ]
    en_img = make_paragraph_bgr(en_lines, lang="en", size=(960, 180))
    zh_img = make_paragraph_bgr(zh_lines, lang="zh-cn", size=(960, 180))

    p_preserve = PaddleOcrProvider(device="cpu", paragraph_mode="preserve_lines")
    p_unwrap = PaddleOcrProvider(device="cpu", paragraph_mode="unwrap")

    res_en_preserve = p_preserve.recognize(en_img, language="en")
    res_en_unwrap = p_unwrap.recognize(en_img, language="en")
    assert "\n" in res_en_preserve.text
    assert "\n" not in res_en_unwrap.text
    assert normalized_text_similarity(res_en_unwrap.text, " ".join(en_lines)) >= 0.93

    res_zh_preserve = p_preserve.recognize(zh_img, language="zh-cn")
    res_zh_unwrap = p_unwrap.recognize(zh_img, language="zh-cn")
    assert "\n" in res_zh_preserve.text
    assert "\n" not in res_zh_unwrap.text
    assert " " not in res_zh_unwrap.text
    assert normalized_text_similarity(res_zh_unwrap.text, "".join(zh_lines)) >= 0.93


def test_t2_ndarray_polygon_normalization_and_blank_image(shared_provider, monkeypatch):
    import paddleocr as paddleocr_mod
    from videotrans.ocr._paddle import PaddleOcrProvider

    # np.ndarray polygons in both 2.x list and 3.x dict outputs must NOT raise
    # ValueError: The truth value of an array with more than one element is ambiguous
    poly_arr = np.array([[10, 20], [110, 20], [110, 50], [10, 50]], dtype=np.float32)
    poly_arr_2 = np.array([[10, 60], [110, 60], [110, 90], [10, 90]], dtype=np.float32)
    flat_arr = np.array([10, 20, 110, 50], dtype=np.float32)

    raw_outputs = iter([
        [[[poly_arr, ("Array polygon", 0.96)]]],
        {
            "rec_texts": ["First", "Second"],
            "rec_scores": [0.95, 0.91],
            "rec_polys": np.array([poly_arr, poly_arr_2], dtype=np.float32),
        },
        {
            "rec_texts": ["Flat"],
            "rec_scores": [0.9],
            "rec_boxes": np.array([flat_arr], dtype=np.float32),
        },
    ])

    class _FakePaddleEngine:
        def ocr(self, _img, **_kwargs):
            return next(raw_outputs)

        def predict(self, _img, **_kwargs):
            return next(raw_outputs)

    # Blank/empty frame on real provider must run BEFORE monkeypatching PaddleOCR
    # so standalone `-k` execution never initializes shared_provider with the fake engine.
    blank = make_blank_bgr(size=(640, 160))
    res_blank = shared_provider.recognize(blank, language="en")
    assert res_blank.text == ""
    assert res_blank.lines == []

    with monkeypatch.context() as mp:
        mp.setattr(paddleocr_mod, "PaddleOCR", lambda **_kwargs: _FakePaddleEngine())
        provider = PaddleOcrProvider(device="cpu")
        dummy_img = make_blank_bgr(size=(200, 120))

        res_2x = provider.recognize(dummy_img, language="en")
        assert res_2x.text == "Array polygon"
        assert len(res_2x.lines) == 1
        assert res_2x.lines[0].box == (10.0, 20.0, 100.0, 30.0)

        res_3x = provider.recognize(dummy_img, language="en")
        assert len(res_3x.lines) == 2
        assert res_3x.lines[0].text == "First"
        assert res_3x.lines[1].text == "Second"
        assert res_3x.lines[0].box == (10.0, 20.0, 100.0, 30.0)
        assert "First" in res_3x.text and "Second" in res_3x.text

        res_flat = provider.recognize(dummy_img, language="en")
        assert res_flat.text == "Flat"
        assert len(res_flat.lines) == 1
        assert res_flat.lines[0].box == (10.0, 20.0, 100.0, 30.0)


def test_t2_tiny_roi_upscaling_recognizes_small_text(shared_provider):
    expected = "Small subtitle strip 42"
    tiny = make_tiny_roi_bgr(expected, lang="en", size=(720, 38), font_size=22)
    res = shared_provider.recognize(tiny, language="en")
    assert normalized_text_similarity(res.text, expected) >= 0.85, (
        f"Tiny (<48px) ROI recognition degraded: {res.text!r}"
    )
