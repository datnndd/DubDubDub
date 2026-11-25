# -*- coding: utf-8 -*-
"""Stage 4 On-Screen Annotations backend unit & FastAPI integration suite."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from videotrans.api.app import create_app
from videotrans.api import task_params as task_params_module
from videotrans.api.task_params import build_task_params
from videotrans.core import project_store
from videotrans.core.media_store import MediaRecord, MediaStore
from videotrans.core.stage_reset import reset_project_stage
from videotrans.ocr._annotations import (
    cluster_callout_lines,
    extract_region_annotation,
    sample_box_style,
    scan_video_annotations,
)
from videotrans.ocr._types import OcrLine, OcrResult
from videotrans.task.taskcfg import InputFile
from videotrans.util._annotations_ass import (
    append_annotations_to_ass,
    build_annotations_ass,
    build_annotations_srt,
)


class _FakeOcrProvider:
    def __init__(self, frame_map):
        self._frame_map = frame_map

    def recognize(self, image, language=None):
        arr = np.asarray(image)
        tag = int(arr[0, 0, 0])
        lines = self._frame_map.get(tag, [])
        return OcrResult(
            text="\n".join(ln.text for ln in lines),
            confidence=0.95 if lines else 0.0,
            lines=lines,
        )


def test_sample_box_style_picks_background_and_contrasting_text():
    # Light yellow slide background BGR=(180, 245, 255) -> RGB=(255, 245, 180)
    light_frame = np.full((360, 640, 3), (180, 245, 255), dtype=np.uint8)
    light_style = sample_box_style(light_frame, (0.2, 0.2, 0.3, 0.1), video_height=1080)
    assert light_style["bgColor"] == "#FFF5B4"
    assert light_style["textColor"] == "#111111"
    assert light_style["bgOpacity"] == 92

    # Dark slate background BGR=(40, 30, 25) -> RGB=(25, 30, 40)
    dark_frame = np.full((360, 640, 3), (40, 30, 25), dtype=np.uint8)
    dark_style = sample_box_style(dark_frame, (0.2, 0.2, 0.3, 0.1), video_height=1080)
    assert dark_style["bgColor"] == "#191E28"
    assert dark_style["textColor"] == "#FFFFFF"


def test_cluster_callout_lines_separates_corners_and_excludes_bottom_subtitle():
    lines = [
        # Top-left two-line callout (should merge into 1 block)
        OcrLine("Battery Life", 0.96, (80, 60, 220, 36)),
        OcrLine("Up to 24 Hours", 0.94, (82, 102, 240, 36)),
        # Top-right independent callout (must stay separate)
        OcrLine("OLED 120Hz", 0.95, (920, 70, 210, 38)),
        # Bottom main subtitle inside default exclude ROI [0, 0.78, 1, 0.22]
        OcrLine("Hello everyone welcome back", 0.99, (340, 630, 600, 44)),
    ]
    blocks = cluster_callout_lines(lines, 1280, 720)
    assert len(blocks) == 2
    assert blocks[0]["text"] == "Battery Life\nUp to 24 Hours"
    assert blocks[1]["text"] == "OLED 120Hz"


def test_scan_video_annotations_tracks_spans_and_filters_persistent_watermark():
    def _make_frame(tag: int) -> np.ndarray:
        frm = np.full((360, 640, 3), 240, dtype=np.uint8)
        # Distinct pattern so _sig_is_static sees scene changes across tags
        frm[0, 0, 0] = tag
        frm[20 : 20 + tag * 8, 20:180, :] = (tag * 37) % 255
        return frm

    watermark = OcrLine("CHANNEL LOGO", 0.98, (500, 20, 120, 24))
    callout = OcrLine("Fast Charging 65W", 0.93, (60, 80, 200, 34))
    bottom_sub = OcrLine("Spoken subtitle line", 0.97, (150, 310, 340, 30))

    provider = _FakeOcrProvider({
        1: [watermark, bottom_sub],
        2: [watermark, callout, bottom_sub],
        3: [watermark, callout],
        4: [watermark],
        5: [watermark],
        6: [watermark],
    })
    frames = [(i * 1000, _make_frame(i + 1)) for i in range(6)]
    res = scan_video_annotations(
        "dummy.mp4",
        frames=frames,
        provider_obj=provider,
        sample_interval_ms=1000,
        filter_static=True,
        video_meta={"width": 640, "height": 360, "duration_ms": 6000},
    )
    assert res["ok"] is True
    assert res["filteredWatermarks"] == 1
    assert len(res["annotations"]) == 1
    ann = res["annotations"][0]
    assert ann["sourceText"] == "Fast Charging 65W"
    assert ann["startSec"] == 1.0
    assert ann["endSec"] == 3.0


def test_extract_region_annotation_probes_backward_and_forward_span():
    def _frame_at(ts_ms: int) -> np.ndarray:
        frm = np.full((360, 640, 3), 220, dtype=np.uint8)
        active = 1500 <= ts_ms <= 3500
        frm[:, :, 0] = 2 if active else 9
        if not active:
            # Make ROI visually distinct when inactive
            frm[40:140, 60:320, :] = (ts_ms // 100) % 200
        return frm

    provider = _FakeOcrProvider({
        2: [OcrLine(" Titanium Frame ", 0.94, (10, 12, 180, 32))],
        9: [],
    })
    out = extract_region_annotation(
        "dummy.mp4",
        anchor_sec=2.5,
        roi=[0.1, 0.12, 0.35, 0.15],
        frame_fetcher=_frame_at,
        provider_obj=provider,
        video_meta={"width": 640, "height": 360, "duration_ms": 6000},
    )
    assert out["ok"] is True
    ann = out["annotation"]
    assert ann["sourceText"] == "Titanium Frame"
    assert 1.2 <= ann["startSec"] <= 1.6
    assert 3.4 <= ann["endSec"] <= 4.0


def test_annotations_srt_and_ass_generation_and_append(tmp_path):
    annotations = [
        {
            "id": "ann_1",
            "startSec": 1.25,
            "endSec": 4.5,
            "box": [0.1, 0.2, 0.25, 0.08],
            "sourceText": "超长续航",
            "targetText": "Pin siêu trâu",
            "style": {
                "bgColor": "#FFF5B4",
                "bgOpacity": 90,
                "textColor": "#111111",
                "fontSize": 34,
                "bold": True,
            },
        }
    ]
    srt = build_annotations_srt(annotations)
    assert "1\n00:00:01,250 --> 00:00:04,500\nPin siêu trâu\n" == srt

    ass = build_annotations_ass(annotations, width=1920, height=1080)
    assert "PlayResX: 1920" in ass
    assert "PlayResY: 1080" in ass
    assert "Style: AnnBox," in ass
    assert "Style: AnnText_1," in ass
    assert "[SRC] 超长续航" in ass
    assert "\\p1}m 0 0 l " in ass
    assert "\\an5\\pos(" in ass
    assert "Pin siêu trâu" in ass

    base_ass = tmp_path / "end.ass"
    base_ass.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,22,&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H00000000&,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:03.00,Default,,0,0,0,,Main dialogue line\n",
        encoding="utf-8",
    )
    append_annotations_to_ass(str(base_ass), annotations, source_video_height=1080)
    merged = base_ass.read_text(encoding="utf-8")
    assert "Style: Default,Arial,22," in merged
    assert "Main dialogue line" in merged
    assert "Style: AnnText_1," in merged
    assert "\\p4}m 0 0 l " in merged
    assert "Pin siêu trâu" in merged


def test_fastapi_annotation_routes_capcut_bundle_render_params_and_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DEFAULT_PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_store, "get_projects_root", lambda: tmp_path / "projects")

    media_file = tmp_path / "demo.mp4"
    media_file.write_bytes(b"fake-mp4-bytes")
    store = MediaStore()
    store._records["media_ann_1"] = MediaRecord("media_ann_1", media_file, "demo.mp4", 12, {"time": 5000})

    app = create_app()
    app.state.media_store = store
    app.state.annotation_scanner = lambda path, **kw: {
        "ok": True,
        "annotations": [
            {
                "id": "ann_1",
                "startSec": 0.5,
                "endSec": 2.5,
                "box": [0.15, 0.15, 0.3, 0.08],
                "sourceText": "Original Label",
                "targetText": "Nhãn đã dịch",
                "style": {"bgColor": "#112233", "bgOpacity": 92, "textColor": "#FFFFFF", "fontSize": 28, "bold": True},
            }
        ],
        "sampledFrames": 5,
        "ocrCalls": 2,
        "filteredWatermarks": 1,
    }
    app.state.annotation_region_extractor = lambda path, **kw: {
        "ok": True,
        "annotation": {
            "id": "ann_manual",
            "startSec": kw["anchor_sec"],
            "endSec": kw["anchor_sec"] + 2.0,
            "box": list(kw["roi"]),
            "sourceText": "Drawn Box Text",
            "targetText": "",
            "style": {"bgColor": "#222222", "bgOpacity": 92, "textColor": "#FFFFFF", "fontSize": 26, "bold": True},
        },
    }

    client = TestClient(app)

    scan_resp = client.post(
        "/api/ocr/annotations/scan",
        json={"mediaId": "media_ann_1", "language": "en", "filterStatic": True},
    )
    assert scan_resp.status_code == 200
    assert scan_resp.json()["annotations"][0]["sourceText"] == "Original Label"

    reg_resp = client.post(
        "/api/ocr/annotations/region",
        json={"mediaId": "media_ann_1", "anchorSec": 1.2, "roi": [0.2, 0.2, 0.3, 0.1]},
    )
    assert reg_resp.status_code == 200
    assert reg_resp.json()["annotation"]["sourceText"] == "Drawn Box Text"

    # Create project with annotations and verify CapCut + Aegisub export bundle
    proj = project_store.create_project(
        name="Annotation Export Test",
        media_id="media_ann_1",
        duration=5.0,
        state={
            "currentStep": 4,
            "project": {"resolution": "1280x720"},
            "stage3Baseline": [{"id": 1, "startSec": 0.0, "endSec": 2.0, "sourceText": "Hi", "targetText": "Chào"}],
            "segments": [{"id": 1, "startSec": 0.0, "endSec": 2.0, "sourceText": "Hi", "targetText": "Chào"}],
            "annotations": scan_resp.json()["annotations"],
            "burnAnnotations": True,
        },
    )
    pid = proj["id"]

    srt_dl = client.get(f"/api/projects/{pid}/export-capcut/annotations.srt")
    assert srt_dl.status_code == 200
    assert "Nhãn đã dịch" in srt_dl.text

    ass_dl = client.get(f"/api/projects/{pid}/export-capcut/annotations.ass")
    assert ass_dl.status_code == 200
    assert "PlayResX: 1280" in ass_dl.text
    assert "PlayResY: 720" in ass_dl.text
    assert "Nhãn đã dịch" in ass_dl.text

    zip_dl = client.get(f"/api/projects/{pid}/export-capcut/bundle.zip")
    assert zip_dl.status_code == 200
    with zipfile.ZipFile(project_store.get_project_dir(pid) / "exports" / f"capcut_export_{pid}.zip") as zf:
        names = set(zf.namelist())
        assert "annotations.srt" in names
        assert "annotations.ass" in names
        assert "subtitles_edited.srt" in names

    # Verify build_task_params forwards annotations & burn_annotations on render
    fake_info = InputFile(
        name=media_file.as_posix(),
        dirname=tmp_path.as_posix(),
        basename="demo.mp4",
        noextname="demo",
        ext="mp4",
        uuid="ann-render-uuid",
    )
    monkeypatch.setattr(task_params_module, "format_video", lambda _p: fake_info)
    params = build_task_params(
        media_file,
        {
            "segments": [],
            "annotations": scan_resp.json()["annotations"],
            "burnAnnotations": True,
        },
        job_type="render",
        project_id=pid,
    )
    assert params["burn_annotations"] is True
    assert len(params["annotations"]) == 1

    # Resetting Stage 4 clears annotations while restoring Stage 3 segments
    reset_out = reset_project_stage(pid, 4)
    assert reset_out["state"]["annotations"] == []
