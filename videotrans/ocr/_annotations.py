# -*- coding: utf-8 -*-
"""Full-frame and region annotation OCR scanning, clustering, and style sampling."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional

import numpy as np

from videotrans import ocr as ocr_pkg
from videotrans.ocr import PADDLE_OCR
from videotrans.ocr._scanner import _roi_signature, _sig_is_static, crop_roi
from videotrans.ocr._text import (
    is_same_subtitle_text,
    join_ordered_lines,
    normalize_text,
    order_lines,
    representative_text,
)
from videotrans.ocr._types import OcrLine
from videotrans.util._srt_parse import ms_to_time_string

DEFAULT_EXCLUDE_ROI: tuple[float, float, float, float] = (0.0, 0.78, 1.0, 0.22)


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def normalize_roi_tuple(
    raw_roi: Any,
    default: tuple[float, float, float, float] | None = DEFAULT_EXCLUDE_ROI,
) -> tuple[float, float, float, float] | None:
    """Parse a dict or 4-sequence into a normalized (x, y, w, h) tuple in [0, 1]."""
    if raw_roi is None:
        return default
    if isinstance(raw_roi, dict):
        rx = raw_roi.get("x")
        ry = raw_roi.get("y")
        rw = raw_roi.get("width") if raw_roi.get("width") is not None else raw_roi.get("w")
        rh = raw_roi.get("height") if raw_roi.get("height") is not None else raw_roi.get("h")
        vals = (rx, ry, rw, rh)
    elif isinstance(raw_roi, (list, tuple)) and len(raw_roi) == 4:
        vals = tuple(raw_roi)
    else:
        raise ValueError("ROI must be a 4-element [x, y, w, h] sequence or object")

    parsed: list[float] = []
    for v in vals:
        if v is None or isinstance(v, bool):
            raise ValueError("ROI coordinates must be numeric")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("ROI coordinates must be finite")
        parsed.append(f)

    x, y, w, h = parsed
    if w <= 0.0 or h <= 0.0:
        return None
    x = _clamp01(x)
    y = _clamp01(y)
    w = max(0.001, min(1.0 - x, w))
    h = max(0.001, min(1.0 - y, h))
    return (round(x, 6), round(y, 6), round(w, 6), round(h, 6))


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 1e-9 else 0.0


def _intersection_over_first(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    if ix1 <= ix0 or iy1 <= iy0 or aw * ah <= 1e-9:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / (aw * ah)


def sample_box_style(
    frame_bgr: Any,
    norm_box: tuple[float, float, float, float],
    *,
    video_height: int = 1080,
    line_count: int = 1,
) -> dict[str, Any]:
    """Sample perimeter BGR pixels around norm_box to pick cover bgColor & contrast textColor."""
    h_norm = max(0.015, float(norm_box[3]))
    lines = max(1, int(line_count))
    ref_h = max(240, int(video_height or 1080))
    est_font = max(14, min(96, int(round((h_norm * ref_h / lines) * 0.72))))

    default_style = {
        "bgColor": "#18181B",
        "bgOpacity": 92,
        "textColor": "#FFFFFF",
        "fontSize": est_font,
        "bold": True,
    }
    if frame_bgr is None:
        return default_style
    arr = np.asarray(frame_bgr)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return default_style

    fh, fw = arr.shape[:2]
    x, y, w, h = norm_box
    x0 = max(0, min(fw - 1, int(round(x * fw))))
    y0 = max(0, min(fh - 1, int(round(y * fh))))
    x1 = max(x0 + 1, min(fw, int(round((x + w) * fw))))
    y1 = max(y0 + 1, min(fh, int(round((y + h) * fh))))

    pad = max(3, min(8, int(round(min(fw, fh) * 0.008))))
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(fw, x1 + pad), min(fh, y1 + pad)

    strips = []
    if y0 > ry0:
        strips.append(arr[ry0:y0, rx0:rx1].reshape(-1, arr.shape[2])[:, :3])
    if ry1 > y1:
        strips.append(arr[y1:ry1, rx0:rx1].reshape(-1, arr.shape[2])[:, :3])
    if x0 > rx0:
        strips.append(arr[y0:y1, rx0:x0].reshape(-1, arr.shape[2])[:, :3])
    if rx1 > x1:
        strips.append(arr[y0:y1, x1:rx1].reshape(-1, arr.shape[2])[:, :3])

    ring = np.concatenate([s for s in strips if s.size > 0], axis=0) if strips else np.empty((0, 3))
    if ring.size == 0:
        ring = arr[y0:y1, x0:x1, :3].reshape(-1, 3)
    if ring.size == 0:
        return default_style

    med = np.median(ring, axis=0)
    b_val = max(0, min(255, int(round(float(med[0])))))
    g_val = max(0, min(255, int(round(float(med[1])))))
    r_val = max(0, min(255, int(round(float(med[2])))))

    bg_hex = f"#{r_val:02X}{g_val:02X}{b_val:02X}"
    lum = 0.2126 * r_val + 0.7152 * g_val + 0.0722 * b_val
    text_hex = "#111111" if lum > 145.0 else "#FFFFFF"

    return {
        "bgColor": bg_hex,
        "bgOpacity": 92,
        "textColor": text_hex,
        "fontSize": est_font,
        "bold": True,
    }


def _extract_norm_line(
    item: Any,
    frame_w: int,
    frame_h: int,
) -> tuple[str, float, tuple[float, float, float, float]] | None:
    if isinstance(item, dict):
        text = str(item.get("text") or "").strip()
        conf = float(item.get("confidence") or 0.0)
        raw_box = item.get("box")
    else:
        text = str(getattr(item, "text", "") or "").strip()
        conf = float(getattr(item, "confidence", 0.0) or 0.0)
        raw_box = getattr(item, "box", None)

    if not text or raw_box is None or len(raw_box) < 4:
        return None
    bx, by, bw, bh = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
    if bw <= 0.0 or bh <= 0.0:
        return None

    # PaddleOCR returns pixel coordinates; convert to normalized [0, 1] when needed.
    is_pixel = any(v > 1.001 for v in (bx, by, bw, bh))
    if is_pixel:
        if frame_w <= 0 or frame_h <= 0:
            return None
        nx, ny, nw, nh = bx / frame_w, by / frame_h, bw / frame_w, bh / frame_h
    else:
        nx, ny, nw, nh = bx, by, bw, bh

    nx = _clamp01(nx)
    ny = _clamp01(ny)
    nw = max(0.001, min(1.0 - nx, nw))
    nh = max(0.001, min(1.0 - ny, nh))
    return text, conf, (nx, ny, nw, nh)


def _lines_belong_same_callout(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    min_h = min(ah, bh)
    max_h = max(ah, bh)

    # Same visual row (inline split chunks)
    yc_diff = abs((ay0 + 0.5 * ah) - (by0 + 0.5 * bh))
    x_gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    if yc_diff <= 0.55 * min_h and x_gap <= 1.5 * max_h:
        return True

    # Vertically stacked lines within the same callout block
    y_gap = max(0.0, max(ay0, by0) - min(ay1, by1))
    if y_gap > 0.85 * min_h:
        return False

    x_overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_ratio = x_overlap / max(1e-9, min(aw, bw))
    left_aligned = abs(ax0 - bx0) <= 1.5 * max_h
    center_aligned = abs((ax0 + 0.5 * aw) - (bx0 + 0.5 * bw)) <= 1.5 * max_h
    return overlap_ratio >= 0.30 or left_aligned or center_aligned


def cluster_callout_lines(
    ocr_lines: Iterable[Any],
    frame_w: int,
    frame_h: int,
    *,
    exclude_roi: tuple[float, float, float, float] | None = DEFAULT_EXCLUDE_ROI,
    min_confidence: float = 0.45,
) -> list[dict[str, Any]]:
    """Filter subtitle ROI lines and group nearby OCR lines into distinct callout blocks."""
    kept: list[tuple[str, float, tuple[float, float, float, float]]] = []
    for raw in ocr_lines or ():
        parsed = _extract_norm_line(raw, frame_w, frame_h)
        if parsed is None:
            continue
        text, conf, box = parsed
        if conf < min_confidence:
            continue
        if exclude_roi is not None:
            ex, ey, ew, eh = exclude_roi
            cx, cy = box[0] + 0.5 * box[2], box[1] + 0.5 * box[3]
            in_excl_center = (ex <= cx <= ex + ew) and (ey <= cy <= ey + eh)
            if in_excl_center or _intersection_over_first(box, exclude_roi) >= 0.5:
                continue
        kept.append(parsed)

    if not kept:
        return []

    n = len(kept)
    visited = [False] * n
    blocks: list[dict[str, Any]] = []
    scale_w = max(320, int(frame_w or 1280))
    scale_h = max(180, int(frame_h or 720))

    for i in range(n):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        comp: list[int] = []
        while queue:
            cur = queue.pop()
            comp.append(cur)
            for j in range(n):
                if not visited[j] and _lines_belong_same_callout(kept[cur][2], kept[j][2]):
                    visited[j] = True
                    queue.append(j)

        comp_lines = [kept[idx] for idx in comp]
        px_lines = [
            OcrLine(
                text=t,
                confidence=c,
                box=(b[0] * scale_w, b[1] * scale_h, b[2] * scale_w, b[3] * scale_h),
            )
            for t, c, b in comp_lines
        ]
        ordered = order_lines(px_lines)
        joined_text = join_ordered_lines(ordered, paragraph_mode="preserve_lines").strip()
        if not joined_text:
            continue

        x0 = min(b[0] for _, _, b in comp_lines)
        y0 = min(b[1] for _, _, b in comp_lines)
        x1 = max(b[0] + b[2] for _, _, b in comp_lines)
        y1 = max(b[1] + b[3] for _, _, b in comp_lines)
        avg_conf = sum(c for _, c, _ in comp_lines) / len(comp_lines)

        blocks.append({
            "text": joined_text,
            "confidence": round(avg_conf, 4),
            "box": (round(x0, 5), round(y0, 5), round(max(0.002, x1 - x0), 5), round(max(0.002, y1 - y0), 5)),
            "line_count": max(1, joined_text.count("\n") + 1),
        })

    blocks.sort(key=lambda b: (b["box"][1], b["box"][0]))
    return blocks


def _same_callout_track(
    track_box: tuple[float, float, float, float],
    track_text: str,
    cand_box: tuple[float, float, float, float],
    cand_text: str,
) -> bool:
    if not is_same_subtitle_text(track_text, cand_text, threshold=0.78):
        return False
    if _box_iou(track_box, cand_box) >= 0.25:
        return True
    tcx, tcy = track_box[0] + 0.5 * track_box[2], track_box[1] + 0.5 * track_box[3]
    ccx, ccy = cand_box[0] + 0.5 * cand_box[2], cand_box[1] + 0.5 * cand_box[3]
    return math.hypot(tcx - ccx, tcy - ccy) <= 0.08


def scan_video_annotations(
    media_path: Any,
    *,
    language: str | None = None,
    exclude_roi: Any = DEFAULT_EXCLUDE_ROI,
    sample_interval_ms: int = 1000,
    filter_static: bool = True,
    frames: Iterable[tuple[int, Any]] | None = None,
    provider_obj: Any = None,
    video_meta: dict[str, Any] | None = None,
    stop_event: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Scan full video frames for non-subtitle callouts, tracking timing & cover styles."""
    norm_exclude = normalize_roi_tuple(exclude_roi, default=DEFAULT_EXCLUDE_ROI)
    step_ms = max(200, min(5000, int(sample_interval_ms or 1000)))

    from videotrans.configure import config as runtime_config
    ffmpeg_exe = (
        getattr(runtime_config, "FFMPEG_BIN", None)
        or runtime_config.settings.get("ffmpeg_cmd")
        or "ffmpeg"
    )

    if video_meta is None and frames is None:
        from videotrans.ocr._frame_source import probe_video_meta
        video_meta = probe_video_meta(str(media_path), ffmpeg_exe)
    meta = video_meta or {}
    orig_w = int(meta.get("width") or 1920)
    orig_h = int(meta.get("height") or 1080)
    duration_ms = int(meta.get("duration_ms") or 0)

    if frames is None:
        from videotrans.ocr._frame_source import sampled_roi_iter
        frames = sampled_roi_iter(
            ffmpeg_exe,
            str(media_path),
            roi=(0.0, 0.0, 1.0, 1.0),
            coarse_interval_ms=step_ms,
            duration_ms=duration_ms,
            max_roi_width=1280,
            stop_event=stop_event,
            meta=meta if meta else None,
        )

    provider = provider_obj
    active_tracks: list[dict[str, Any]] = []
    closed_tracks: list[dict[str, Any]] = []
    grace_ms = max(int(step_ms * 1.5), 600)

    last_sig = None
    sampled_count = 0
    ocr_calls = 0
    first_ts_ms: int | None = None
    last_ts_ms: int = 0

    for ts_ms, frame in frames:
        if stop_event is not None and stop_event():
            break
        if frame is None:
            continue
        arr = np.asarray(frame)
        if arr.ndim < 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            continue

        ts = int(ts_ms)
        if first_ts_ms is None:
            first_ts_ms = ts
        last_ts_ms = max(last_ts_ms, ts)
        sampled_count += 1

        sig = _roi_signature(arr, tile_w=16, tile_h=9)
        if last_sig is not None and sig is not None and _sig_is_static(last_sig, sig):
            for trk in active_tracks:
                trk["last_ms"] = ts
                trk["samples"] += 1
            continue
        last_sig = sig

        if provider is None:
            provider = ocr_pkg.get_cached_provider(provider=PADDLE_OCR)
        ocr_res = provider.recognize(arr, language)
        ocr_calls += 1

        fh, fw = arr.shape[:2]
        raw_lines = getattr(ocr_res, "lines", None) or (ocr_res.get("lines") if isinstance(ocr_res, dict) else None) or []
        blocks = cluster_callout_lines(raw_lines, fw, fh, exclude_roi=norm_exclude)

        matched_track_idxs: set[int] = set()
        for blk in blocks:
            best_idx = None
            for idx, trk in enumerate(active_tracks):
                if idx in matched_track_idxs:
                    continue
                if _same_callout_track(trk["box"], trk["rep_text"], blk["box"], blk["text"]):
                    best_idx = idx
                    break

            if best_idx is not None:
                matched_track_idxs.add(best_idx)
                trk = active_tracks[best_idx]
                trk["last_ms"] = ts
                trk["samples"] += 1
                trk["obs"].append((blk["text"], blk["confidence"]))
                trk["rep_text"] = representative_text(trk["obs"]) or blk["text"]
                # Smooth box union/average to stabilize minor frame jitter
                tx, ty, tw, th = trk["box"]
                bx, by, bw, bh = blk["box"]
                trk["box"] = (
                    round(0.7 * tx + 0.3 * bx, 5),
                    round(0.7 * ty + 0.3 * by, 5),
                    round(max(tw, bw), 5),
                    round(max(th, bh), 5),
                )
            else:
                style = sample_box_style(
                    arr,
                    blk["box"],
                    video_height=orig_h,
                    line_count=blk["line_count"],
                )
                active_tracks.append({
                    "start_ms": ts,
                    "last_ms": ts,
                    "samples": 1,
                    "obs": [(blk["text"], blk["confidence"])],
                    "rep_text": blk["text"],
                    "box": blk["box"],
                    "style": style,
                })
                matched_track_idxs.add(len(active_tracks) - 1)

        still_active: list[dict[str, Any]] = []
        for idx, trk in enumerate(active_tracks):
            if idx not in matched_track_idxs and (ts - trk["last_ms"]) >= grace_ms:
                trk["end_ms"] = min(ts, trk["last_ms"] + step_ms)
                closed_tracks.append(trk)
            else:
                still_active.append(trk)
        active_tracks = still_active

    effective_end_ms = max(duration_ms, last_ts_ms + step_ms) if sampled_count > 0 else 0
    for trk in active_tracks:
        trk["end_ms"] = min(effective_end_ms, trk["last_ms"] + step_ms) if duration_ms > 0 else (trk["last_ms"] + step_ms)
        closed_tracks.append(trk)

    total_span_ms = max(1, (effective_end_ms - (first_ts_ms or 0)))
    annotations: list[dict[str, Any]] = []
    filtered_watermarks = 0

    closed_tracks.sort(key=lambda t: (t["start_ms"], t["box"][1], t["box"][0]))
    for trk in closed_tracks:
        text = representative_text(trk["obs"]).strip()
        if not text or not normalize_text(text):
            continue
        confs = [c for _, c in trk["obs"]]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        if trk["samples"] <= 1 and avg_conf < 0.55 and sampled_count >= 4:
            continue

        span_ms = max(step_ms, int(trk["end_ms"]) - int(trk["start_ms"]))
        if (
            filter_static
            and sampled_count >= 5
            and total_span_ms >= 4000
            and (span_ms / float(total_span_ms)) >= 0.65
        ):
            filtered_watermarks += 1
            continue

        s_ms = max(0, int(trk["start_ms"]))
        e_ms = max(s_ms + 200, int(trk["end_ms"]))
        if duration_ms > 0:
            e_ms = min(max(s_ms + 200, duration_ms), e_ms)

        ann_idx = len(annotations) + 1
        annotations.append({
            "id": f"ann_{ann_idx}",
            "startSec": round(s_ms / 1000.0, 3),
            "endSec": round(e_ms / 1000.0, 3),
            "startTime": ms_to_time_string(ms=s_ms),
            "endTime": ms_to_time_string(ms=e_ms),
            "box": list(trk["box"]),
            "sourceText": text,
            "targetText": "",
            "confidence": round(avg_conf, 2),
            "style": trk["style"],
        })

    return {
        "ok": True,
        "annotations": annotations,
        "sampledFrames": sampled_count,
        "ocrCalls": ocr_calls,
        "filteredWatermarks": filtered_watermarks,
        "resolution": {"width": orig_w, "height": orig_h},
    }


def extract_region_annotation(
    media_path: Any,
    *,
    anchor_sec: float,
    roi: Any,
    language: str | None = None,
    max_span_sec: float = 15.0,
    frame_fetcher: Callable[[int], Any] | None = None,
    provider_obj: Any = None,
    video_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract a single callout inside roi at anchor_sec and probe its [startSec, endSec] span."""
    norm_roi = normalize_roi_tuple(roi, default=None)
    if norm_roi is None:
        raise ValueError("A positive normalized ROI [x, y, w, h] is required")

    from videotrans.configure import config as runtime_config
    ffmpeg_exe = (
        getattr(runtime_config, "FFMPEG_BIN", None)
        or runtime_config.settings.get("ffmpeg_cmd")
        or "ffmpeg"
    )

    if video_meta is None and frame_fetcher is None:
        from videotrans.ocr._frame_source import probe_video_meta
        video_meta = probe_video_meta(str(media_path), ffmpeg_exe)
    meta = video_meta or {}
    orig_w = int(meta.get("width") or 1920)
    orig_h = int(meta.get("height") or 1080)
    duration_ms = int(meta.get("duration_ms") or 0)

    def _fetch_full_frame(ts_ms: int) -> np.ndarray | None:
        clamped_ms = max(0, ts_ms)
        if duration_ms > 0:
            clamped_ms = min(max(0, duration_ms - 40), clamped_ms)
        if frame_fetcher is not None:
            res = frame_fetcher(clamped_ms)
            if isinstance(res, tuple):
                res = res[0]
            return np.asarray(res) if res is not None else None
        from videotrans.ocr._frame_source import frame_at
        frm, w, h = frame_at(ffmpeg_exe, str(media_path), clamped_ms)
        return frm if (frm is not None and w > 0 and h > 0) else None

    anchor_ms = max(0, int(round(float(anchor_sec) * 1000.0)))
    if duration_ms > 0:
        anchor_ms = min(max(0, duration_ms - 40), anchor_ms)

    anchor_frame = _fetch_full_frame(anchor_ms)
    if anchor_frame is None:
        raise RuntimeError("Could not decode video frame at the selected timestamp")

    anchor_crop = crop_roi(anchor_frame, norm_roi)
    if anchor_crop is None or anchor_crop.size == 0:
        raise RuntimeError("Selected region contains no video pixels")

    provider = provider_obj or ocr_pkg.get_cached_provider(provider=PADDLE_OCR)
    ocr_res = provider.recognize(anchor_crop, language)

    ch, cw = anchor_crop.shape[:2]
    raw_lines = getattr(ocr_res, "lines", None) or (ocr_res.get("lines") if isinstance(ocr_res, dict) else None) or []
    ordered_lines = order_lines(raw_lines)
    anchor_text = join_ordered_lines(ordered_lines, paragraph_mode="preserve_lines").strip()
    if not anchor_text:
        anchor_text = str(getattr(ocr_res, "text", "") or (ocr_res.get("text") if isinstance(ocr_res, dict) else "") or "").strip()

    confs = [
        float(ln.get("confidence") if isinstance(ln, dict) else getattr(ln, "confidence", 0.0) or 0.0)
        for ln in ordered_lines
    ]
    avg_conf = (sum(confs) / len(confs)) if confs else float(getattr(ocr_res, "confidence", 0.0) or 0.0)

    # Tighten box to detected text inside ROI when available, otherwise keep user ROI
    final_box = norm_roi
    valid_sub_boxes = []
    for ln in ordered_lines:
        parsed = _extract_norm_line(ln, cw, ch)
        if parsed is not None:
            valid_sub_boxes.append(parsed[2])
    if valid_sub_boxes:
        rx, ry, rw, rh = norm_roi
        sx0 = min(b[0] for b in valid_sub_boxes)
        sy0 = min(b[1] for b in valid_sub_boxes)
        sx1 = max(b[0] + b[2] for b in valid_sub_boxes)
        sy1 = max(b[1] + b[3] for b in valid_sub_boxes)
        final_box = (
            round(_clamp01(rx + sx0 * rw), 5),
            round(_clamp01(ry + sy0 * rh), 5),
            round(max(0.004, (sx1 - sx0) * rw), 5),
            round(max(0.004, (sy1 - sy0) * rh), 5),
        )

    line_count = max(1, anchor_text.count("\n") + 1)
    style = sample_box_style(anchor_frame, final_box, video_height=orig_h, line_count=line_count)
    anchor_sig = _roi_signature(anchor_crop)

    def _matches_anchor(ts_ms: int) -> bool:
        frm = _fetch_full_frame(ts_ms)
        if frm is None:
            return False
        crp = crop_roi(frm, norm_roi)
        if crp is None or crp.size == 0:
            return False
        cand_sig = _roi_signature(crp)
        if anchor_sig is not None and cand_sig is not None and _sig_is_static(anchor_sig, cand_sig):
            return True
        if not anchor_text:
            return False
        res = provider.recognize(crp, language)
        r_lines = getattr(res, "lines", None) or (res.get("lines") if isinstance(res, dict) else None) or []
        cand_txt = join_ordered_lines(order_lines(r_lines), paragraph_mode="preserve_lines").strip()
        if not cand_txt:
            cand_txt = str(getattr(res, "text", "") or "").strip()
        return is_same_subtitle_text(anchor_text, cand_txt, threshold=0.78)

    max_span_ms = max(500, int(round(float(max_span_sec or 15.0) * 1000.0)))
    coarse_ms = 500
    min_bound_ms = max(0, anchor_ms - max_span_ms)
    max_bound_ms = min(duration_ms, anchor_ms + max_span_ms) if duration_ms > 0 else (anchor_ms + max_span_ms)

    start_ms = anchor_ms
    probe_t = anchor_ms - coarse_ms
    while probe_t >= min_bound_ms:
        if _matches_anchor(probe_t):
            start_ms = probe_t
            probe_t -= coarse_ms
        else:
            mid = (probe_t + start_ms) // 2
            if mid != probe_t and mid != start_ms and _matches_anchor(mid):
                start_ms = mid
            break

    end_ms = anchor_ms + coarse_ms
    probe_t = anchor_ms + coarse_ms
    while probe_t <= max_bound_ms:
        if _matches_anchor(probe_t):
            end_ms = probe_t + 250
            probe_t += coarse_ms
        else:
            mid = ( (end_ms - 250) + probe_t ) // 2
            if mid > anchor_ms and _matches_anchor(mid):
                end_ms = mid + 150
            else:
                end_ms = max(anchor_ms + 250, probe_t - coarse_ms + 250)
            break

    if duration_ms > 0:
        end_ms = min(duration_ms, max(start_ms + 250, end_ms))
    else:
        end_ms = max(start_ms + 250, end_ms)

    annotation = {
        "id": f"ann_{anchor_ms}",
        "startSec": round(start_ms / 1000.0, 3),
        "endSec": round(end_ms / 1000.0, 3),
        "startTime": ms_to_time_string(ms=start_ms),
        "endTime": ms_to_time_string(ms=end_ms),
        "box": list(final_box),
        "sourceText": anchor_text,
        "targetText": "",
        "confidence": round(avg_conf, 2),
        "style": style,
    }
    return {"ok": True, "annotation": annotation}
