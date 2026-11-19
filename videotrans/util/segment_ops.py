"""Transcript segment operations: timestamp formatting, CPS calculation, and segment splitting."""

from __future__ import annotations

import math
import re
from typing import Any


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.mmm representation."""
    try:
        val = float(seconds)
        if val != val or math.isnan(val) or math.isinf(val):  # NaN / Inf check
            val = 0.0
    except (TypeError, ValueError, OverflowError):
        val = 0.0
    try:
        total_ms = int(round(max(0.0, val) * 1000))
    except (TypeError, ValueError, OverflowError):
        total_ms = 0
    minutes = total_ms // 60000
    remaining_ms = total_ms % 60000
    secs = remaining_ms // 1000
    ms = remaining_ms % 1000
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def parse_timestamp(ts: Any) -> float:
    """Parse MM:SS.mmm, HH:MM:SS,mmm, or numeric values into seconds."""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        try:
            val = float(ts)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return max(0.0, val)
        except (TypeError, ValueError, OverflowError):
            return 0.0
    cleaned = str(ts).strip().replace(",", ".")
    if not cleaned:
        return 0.0
    try:
        parts = cleaned.split(":")
        if len(parts) == 3:
            h, m, s = parts
            val = int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            val = int(m) * 60 + float(s)
        else:
            val = float(cleaned)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return max(0.0, val)
    except (ValueError, TypeError, OverflowError):
        return 0.0


def calculate_cps(text: str, duration: float) -> tuple[float, str]:
    """Calculate characters per second and qualitative pacing status."""
    char_count = len(str(text or "").strip())
    if char_count == 0 or duration <= 0:
        return 0.0, "Optimal"
    safe_duration = max(0.1, duration)
    cps = round(char_count / safe_duration, 1)
    if cps <= 14.5:
        status = "Optimal"
    elif cps <= 18.0:
        status = "Good"
    else:
        status = "Fast"
    return cps, status


def split_segment(
    segment: dict[str, Any],
    split_time: float,
    cursor_position: int | None = None,
    *,
    next_id: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a dialog segment card at timestamp split_time.

    Requires startSec < split_time < endSec.
    If cursor_position is provided and within range, splits text at the cursor.
    Otherwise, splits text cleanly at the nearest word boundary proportional
    to the split timestamp.
    """
    raw_start = segment.get("startSec") if segment.get("startSec") is not None else segment.get("startTime", 0.0)
    raw_end = segment.get("endSec") if segment.get("endSec") is not None else segment.get("endTime", 0.0)
    start_sec = round(parse_timestamp(raw_start), 3)
    end_sec = round(parse_timestamp(raw_end), 3)
    try:
        split_sec = round(float(split_time), 3)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid split timestamp: {split_time}")

    if not (start_sec < split_sec < end_sec):
        raise ValueError(
            f"Split timestamp {split_time} must be strictly between startSec ({start_sec}) and endSec ({end_sec})"
        )

    source_text = str(segment.get("sourceText") if segment.get("sourceText") is not None else (segment.get("text") or ""))

    # Safely coerce cursor_position to int if provided
    cur_pos: int | None = None
    if cursor_position is not None:
        try:
            cur_pos = int(cursor_position)
        except (TypeError, ValueError):
            cur_pos = None

    if cur_pos is not None and 0 <= cur_pos <= len(source_text):
        text1 = source_text[:cur_pos].strip()
        text2 = source_text[cur_pos:].strip()
    else:
        # Split cleanly at nearest word boundary proportional to time
        trimmed = source_text.strip()
        if not trimmed:
            text1, text2 = "", ""
        else:
            ratio = (split_sec - start_sec) / (end_sec - start_sec)
            target_char = int(round(len(trimmed) * ratio))

            # Collect natural boundaries (spaces and punctuation marks)
            punct_chars = set("，。！？、；：,!?;:.")
            candidates: list[tuple[int, bool]] = []
            for i, c in enumerate(trimmed):
                if 0 < i < len(trimmed) - 1:
                    if c.isspace():
                        candidates.append((i, True))
                    elif c in punct_chars:
                        candidates.append((i + 1, False))

            if candidates:
                # Pick boundary closest to target_char; prefer punctuation on equal distance
                best_idx, is_space = min(
                    candidates,
                    key=lambda item: (abs(item[0] - target_char), 0 if not item[1] else 1),
                )
                if is_space:
                    text1 = trimmed[:best_idx].strip()
                    text2 = trimmed[best_idx + 1:].strip()
                else:
                    text1 = trimmed[:best_idx].strip()
                    text2 = trimmed[best_idx:].strip()
            else:
                # For CJK text without spaces or punctuation, try word boundary tokenization via jieba
                cjk_split_done = False
                try:
                    import jieba
                    words = list(jieba.cut(trimmed))
                    if len(words) > 1:
                        boundaries = []
                        curr_len = 0
                        for w in words[:-1]:
                            curr_len += len(w)
                            boundaries.append(curr_len)
                        if boundaries:
                            best_i = min(boundaries, key=lambda idx: abs(idx - target_char))
                            text1 = trimmed[:best_i].strip()
                            text2 = trimmed[best_i:].strip()
                            cjk_split_done = True
                except Exception:
                    cjk_split_done = False

                if not cjk_split_done:
                    # Fallback: split at character count ratio
                    split_idx = max(1, min(target_char, max(1, len(trimmed) - 1)))
                    text1 = trimmed[:split_idx].strip()
                    text2 = trimmed[split_idx:].strip()

    dur1 = max(0.001, split_sec - start_sec)
    dur2 = max(0.001, end_sec - split_sec)
    cps1, status1 = calculate_cps(text1, dur1)
    cps2, status2 = calculate_cps(text2, dur2)

    orig_id = segment.get("id", 1)
    if next_id is not None:
        second_id = next_id
    elif isinstance(orig_id, int):
        second_id = orig_id + 1
    elif str(orig_id).isdigit():
        second_id = int(orig_id) + 1
    elif isinstance(orig_id, str) and orig_id.startswith("seg-"):
        num_part = orig_id[4:]
        second_id = f"seg-{int(num_part) + 1}" if num_part.isdigit() else f"{orig_id}_b"
    else:
        second_id = f"{orig_id}_b"

    seg1 = {
        **segment,
        "id": orig_id,
        "startSec": start_sec,
        "endSec": split_sec,
        "startTime": format_timestamp(start_sec),
        "endTime": format_timestamp(split_sec),
        "sourceText": text1,
        "text": text1,
        "cps": cps1,
        "cpsStatus": status1,
        "hasOcrDiff": False,
        "ocrSlideText": None,
    }

    seg2 = {
        **segment,
        "id": second_id,
        "startSec": split_sec,
        "endSec": end_sec,
        "startTime": format_timestamp(split_sec),
        "endTime": format_timestamp(end_sec),
        "sourceText": text2,
        "text": text2,
        "cps": cps2,
        "cpsStatus": status2,
        "hasOcrDiff": False,
        "ocrSlideText": None,
    }

    return seg1, seg2
