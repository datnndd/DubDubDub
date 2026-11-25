# -*- coding: utf-8 -*-
"""SRT and positioned ASS serializers for Stage 4 on-screen annotations."""

from __future__ import annotations

import math
import os
import re
from typing import Any

from videotrans.util._srt_parse import ms_to_time_string


def _parse_hex_rgb(value: Any, fallback: tuple[int, int, int] = (24, 24, 27)) -> tuple[int, int, int]:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(value or "").strip())
    if not match:
        return fallback
    raw = match.group(1)
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def hex_to_ass_bgr(value: Any, fallback: tuple[int, int, int] = (255, 255, 255)) -> str:
    """Return ASS override color literal &HBBGGRR&."""
    r, g, b = _parse_hex_rgb(value, fallback)
    return f"&H{b:02X}{g:02X}{r:02X}&"


def hex_to_ass_style_color(
    value: Any,
    opacity_pct: float = 100.0,
    fallback: tuple[int, int, int] = (255, 255, 255),
) -> str:
    """Return ASS [V4+ Styles] color literal &HAABBGGRR& (00=opaque, FF=transparent)."""
    r, g, b = _parse_hex_rgb(value, fallback)
    clamped = max(0.0, min(100.0, float(opacity_pct)))
    alpha = int(round((1.0 - clamped / 100.0) * 255.0))
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}&"


def format_ass_timestamp(sec: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.CC (centiseconds)."""
    total_cs = max(0, int(round(float(sec or 0.0) * 100.0)))
    cs = total_cs % 100
    total_sec = total_cs // 100
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    escaped = str(text or "").replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    return re.sub(r"\r?\n", r"\\N", escaped.strip())


def _estimate_char_units(ch: str) -> float:
    if not ch:
        return 0.0
    code = ord(ch)
    if (
        0x4E00 <= code <= 0x9FFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
        or 0xFF00 <= code <= 0xFFEF
    ):
        return 1.0
    return 0.56


def _wrap_text_for_box(text: str, box_w_px: float, font_size: float) -> list[str]:
    """Word-wrap text so lines fit comfortably near box_w_px when possible."""
    max_units = max(6.0, (box_w_px / max(8.0, font_size)))
    out_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        clean = raw_line.strip()
        if not clean:
            continue
        words = clean.split(" ")
        if len(words) <= 1:
            out_lines.append(clean)
            continue
        cur = words[0]
        cur_u = sum(_estimate_char_units(c) for c in cur)
        for w in words[1:]:
            wu = sum(_estimate_char_units(c) for c in w) + 0.35
            if cur_u + wu <= max_units * 1.25:
                cur = f"{cur} {w}"
                cur_u += wu
            else:
                out_lines.append(cur)
                cur = w
                cur_u = wu
        if cur:
            out_lines.append(cur)
    return out_lines or [str(text or "").strip()]


def _normalize_annotation_item(raw: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    source_text = str(raw.get("sourceText") or "").strip()
    target_text = str(raw.get("targetText") or "").strip()
    display_text = target_text or source_text
    if not display_text:
        return None

    try:
        start_sec = max(0.0, float(raw.get("startSec") or 0.0))
        end_sec = max(start_sec + 0.1, float(raw.get("endSec") or (start_sec + 1.0)))
    except (TypeError, ValueError):
        start_sec, end_sec = 0.0, 1.0

    if not (math.isfinite(start_sec) and math.isfinite(end_sec)):
        return None

    raw_box = raw.get("box")
    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        try:
            bx, by, bw, bh = [float(v) for v in raw_box]
        except (TypeError, ValueError):
            bx, by, bw, bh = 0.1, 0.1, 0.3, 0.08
    elif isinstance(raw_box, dict):
        bx = float(raw_box.get("x") or 0.1)
        by = float(raw_box.get("y") or 0.1)
        bw = float(raw_box.get("width") or raw_box.get("w") or 0.3)
        bh = float(raw_box.get("height") or raw_box.get("h") or 0.08)
    else:
        bx, by, bw, bh = 0.1, 0.1, 0.3, 0.08

    bx = max(0.0, min(0.99, bx))
    by = max(0.0, min(0.99, by))
    bw = max(0.01, min(1.0 - bx, bw))
    bh = max(0.01, min(1.0 - by, bh))

    style = raw.get("style") if isinstance(raw.get("style"), dict) else {}
    bg_color = str(style.get("bgColor") or "#18181B")
    text_color = str(style.get("textColor") or "#FFFFFF")
    try:
        bg_opacity = max(0.0, min(100.0, float(style.get("bgOpacity") if style.get("bgOpacity") is not None else 92.0)))
    except (TypeError, ValueError):
        bg_opacity = 92.0
    try:
        font_size = int(style.get("fontSize") or 0)
    except (TypeError, ValueError):
        font_size = 0
    bold = bool(style.get("bold", True))
    font_family = str(style.get("fontFamily") or "Arial").replace(",", " ").strip() or "Arial"

    return {
        "index": index,
        "id": str(raw.get("id") or f"ann_{index}"),
        "startSec": start_sec,
        "endSec": end_sec,
        "box": (bx, by, bw, bh),
        "sourceText": source_text,
        "targetText": target_text,
        "displayText": display_text,
        "bgColor": bg_color,
        "bgOpacity": bg_opacity,
        "textColor": text_color,
        "fontSize": font_size,
        "bold": bold,
        "fontFamily": font_family,
    }


def build_annotations_srt(annotations: list[Any], *, field: str = "targetText") -> str:
    """Serialize annotations to UTF-8 SRT ordered by (startSec, y, x)."""
    items = []
    for idx, raw in enumerate(annotations or (), 1):
        norm = _normalize_annotation_item(raw, idx)
        if norm is not None:
            items.append(norm)
    if not items:
        return ""

    items.sort(key=lambda it: (it["startSec"], it["box"][1], it["box"][0]))
    blocks: list[str] = []
    for i, it in enumerate(items, 1):
        txt = (it["sourceText"] or it["targetText"]) if field == "sourceText" else it["displayText"]
        s_str = ms_to_time_string(ms=int(round(it["startSec"] * 1000.0)))
        e_str = ms_to_time_string(ms=int(round(it["endSec"] * 1000.0)))
        blocks.append(f"{i}\n{s_str} --> {e_str}\n{txt.strip()}\n")
    return "\n".join(blocks).strip() + "\n"


def _render_ass_styles_and_events(
    items: list[dict[str, Any]],
    *,
    play_res_x: int,
    play_res_y: int,
    ref_font_res_y: int = 1080,
    subpixel_scale: int = 1,
) -> tuple[list[str], list[str]]:
    """Generate [V4+ Styles] lines and [Events] lines in (play_res_x, play_res_y) space."""
    w_res = max(64, int(play_res_x))
    h_res = max(64, int(play_res_y))
    p_tag = "p4" if subpixel_scale == 8 else "p1"
    vec_mul = 8 if subpixel_scale == 8 else 1

    style_lines = [
        "Style: AnnBox,Arial,20,&H00000000&,&H00000000&,&H00000000&,&H00000000&,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1"
    ]
    event_lines: list[str] = []

    for seq_idx, it in enumerate(items, 1):
        bx, by, bw, bh = it["box"]
        box_x = bx * w_res
        box_y = by * h_res
        box_w = max(8.0, bw * w_res)
        box_h = max(8.0, bh * h_res)

        raw_lines_est = max(1, it["displayText"].count("\n") + 1)
        if it["fontSize"] > 0:
            fs_px = max(8, int(round(it["fontSize"] * (h_res / float(max(240, ref_font_res_y))))))
        else:
            fs_px = max(10, min(int(round(h_res * 0.09)), int(round((box_h / raw_lines_est) * 0.72))))

        wrapped_lines = _wrap_text_for_box(it["displayText"], box_w, fs_px)
        # Shrink font slightly if wrapped line count or width exceeds original box significantly
        max_line_units = max(sum(_estimate_char_units(c) for c in ln) for ln in wrapped_lines)
        est_text_w = max_line_units * fs_px
        est_text_h = len(wrapped_lines) * fs_px * 1.22
        if est_text_w > box_w * 1.35:
            scale = max(0.62, (box_w * 1.25) / max(1.0, est_text_w))
            fs_px = max(9, int(round(fs_px * scale)))
            wrapped_lines = _wrap_text_for_box(it["displayText"], box_w, fs_px)
            max_line_units = max(sum(_estimate_char_units(c) for c in ln) for ln in wrapped_lines)
            est_text_w = max_line_units * fs_px
            est_text_h = len(wrapped_lines) * fs_px * 1.22

        pad_x = max(4.0, fs_px * 0.35)
        pad_y = max(3.0, fs_px * 0.22)
        cx = box_x + 0.5 * box_w
        cy = box_y + 0.5 * box_h

        cov_w = max(box_w + 2.0 * pad_x, est_text_w + 2.0 * pad_x)
        cov_h = max(box_h + 2.0 * pad_y, est_text_h + 2.0 * pad_y)
        cov_x0 = max(0.0, min(w_res - 4.0, cx - 0.5 * cov_w))
        cov_y0 = max(0.0, min(h_res - 4.0, cy - 0.5 * cov_h))
        cov_w = min(cov_w, w_res - cov_x0)
        cov_h = min(cov_h, h_res - cov_y0)
        cx = cov_x0 + 0.5 * cov_w
        cy = cov_y0 + 0.5 * cov_h

        style_name = f"AnnText_{seq_idx}"
        primary_col = hex_to_ass_style_color(it["textColor"], 100.0, fallback=(255, 255, 255))
        r_t, g_t, b_t = _parse_hex_rgb(it["textColor"], (255, 255, 255))
        lum_t = 0.2126 * r_t + 0.7152 * g_t + 0.0722 * b_t
        outline_col = "&H60000000&" if lum_t > 128 else "&H60FFFFFF&"
        outline_w = max(0.5, round(fs_px * 0.05, 1))
        bold_flag = -1 if it["bold"] else 0

        style_lines.append(
            f"Style: {style_name},{it['fontFamily']},{fs_px},{primary_col},{primary_col},"
            f"{outline_col},&H00000000&,{bold_flag},0,0,0,100,100,0,0,1,{outline_w},0,5,6,6,6,1"
        )

        s_ass = format_ass_timestamp(it["startSec"])
        e_ass = format_ass_timestamp(it["endSec"])
        src_comment = re.sub(r"[\r\n]+", " / ", it["sourceText"] or it["displayText"])
        event_lines.append(
            f"Comment: 0,{s_ass},{e_ass},AnnBox,,0,0,0,,"
            f"[SRC] {src_comment} | id={it['id']} | box={bx:.4f},{by:.4f},{bw:.4f},{bh:.4f}"
        )

        if it["bgOpacity"] > 0.5:
            box_bgr = hex_to_ass_bgr(it["bgColor"], fallback=(24, 24, 27))
            alpha_int = max(0, min(255, int(round((1.0 - it["bgOpacity"] / 100.0) * 255.0))))
            vw = int(round(cov_w * vec_mul))
            vh = int(round(cov_h * vec_mul))
            event_lines.append(
                f"Dialogue: 0,{s_ass},{e_ass},AnnBox,,0,0,0,,"
                f"{{\\an7\\pos({cov_x0:.1f},{cov_y0:.1f})\\1c{box_bgr}\\1a&H{alpha_int:02X}&\\bord0\\shad0\\{p_tag}}}"
                f"m 0 0 l {vw} 0 l {vw} {vh} l 0 {vh}{{\\p0}}"
            )

        escaped_text = "\\N".join(_escape_ass_text(ln) for ln in wrapped_lines)
        event_lines.append(
            f"Dialogue: 1,{s_ass},{e_ass},{style_name},,0,0,0,,"
            f"{{\\an5\\pos({cx:.1f},{cy:.1f})}}{escaped_text}"
        )

    return style_lines, event_lines


def build_annotations_ass(
    annotations: list[Any],
    *,
    width: int = 1920,
    height: int = 1080,
) -> str:
    """Build a standalone Aegisub-ready ASS script at native video resolution."""
    play_w = max(64, int(width or 1920))
    play_h = max(64, int(height or 1080))

    items = []
    for idx, raw in enumerate(annotations or (), 1):
        norm = _normalize_annotation_item(raw, idx)
        if norm is not None:
            items.append(norm)
    items.sort(key=lambda it: (it["startSec"], it["box"][1], it["box"][0]))

    style_lines, event_lines = _render_ass_styles_and_events(
        items,
        play_res_x=play_w,
        play_res_y=play_h,
        ref_font_res_y=play_h,
        subpixel_scale=1,
    )

    header = [
        "[Script Info]",
        "; Generated by DubDub Studio — On-Screen Annotations",
        "ScriptType: v4.00+",
        f"PlayResX: {play_w}",
        f"PlayResY: {play_h}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        *style_lines,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *event_lines,
        "",
    ]
    return "\n".join(header)


def append_annotations_to_ass(
    ass_file_path: str,
    annotations: list[Any],
    *,
    source_video_height: int = 1080,
) -> str:
    """Append positioned annotation cover boxes & labels into an existing FFmpeg ASS file."""
    if not annotations or not os.path.exists(ass_file_path):
        return ass_file_path

    items = []
    for idx, raw in enumerate(annotations or (), 1):
        norm = _normalize_annotation_item(raw, idx)
        if norm is not None:
            items.append(norm)
    if not items:
        return ass_file_path
    items.sort(key=lambda it: (it["startSec"], it["box"][1], it["box"][0]))

    with open(ass_file_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    m_x = re.search(r"^\s*PlayResX\s*:\s*(\d+)", content, flags=re.MULTILINE | re.IGNORECASE)
    m_y = re.search(r"^\s*PlayResY\s*:\s*(\d+)", content, flags=re.MULTILINE | re.IGNORECASE)
    play_w = int(m_x.group(1)) if m_x else 384
    play_h = int(m_y.group(1)) if m_y else 288

    if not m_x or not m_y:
        script_inject = ""
        if not m_x:
            script_inject += f"PlayResX: {play_w}\n"
        if not m_y:
            script_inject += f"PlayResY: {play_h}\n"
        content = re.sub(
            r"(^\[Script Info\]\s*\r?\n)",
            r"\1" + script_inject,
            content,
            count=1,
            flags=re.MULTILINE,
        )

    subpixel = 8 if play_h < 720 else 1
    style_lines, event_lines = _render_ass_styles_and_events(
        items,
        play_res_x=play_w,
        play_res_y=play_h,
        ref_font_res_y=max(240, int(source_video_height or 1080)),
        subpixel_scale=subpixel,
    )

    style_block = "\n".join(style_lines) + "\n"
    if "[Events]" in content:
        content = content.replace("[Events]", f"{style_block}\n[Events]", 1)
    else:
        content = content.rstrip() + f"\n\n[V4+ Styles]\n{style_block}\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    content = content.rstrip() + "\n" + "\n".join(event_lines) + "\n"
    with open(ass_file_path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return ass_file_path
