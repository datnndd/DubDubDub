"""Text normalization, similarity, line ordering, and consensus helpers.

Pure stdlib (re, difflib). No heavy dependency imports. The normalization path
here produces a COMPARISON key only; display text must be preserved separately
by callers.
"""
import re
from difflib import SequenceMatcher
from statistics import median
from typing import List, Tuple

# Characters treated as leading/trailing punctuation / common OCR noise.
# Kept broad but conservative; interior word characters are never stripped.
_EDGE_PUNCT = " \t\r\n\"'`~!@#$%^&*()_+\\-=\\[\\]{};:,.<>/?|，。！？、：；「」『』（）《》【】…—·"
_EDGE_PATTERN = re.compile(r"^[%s]+|[%s]+$" % (_EDGE_PUNCT, _EDGE_PUNCT))
_WS_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Return a normalized COMPARISON key for the given text.

    Steps: strip, collapse internal whitespace to single spaces, remove
    leading/trailing punctuation / common OCR noise, and casefold. This value
    is intended for comparison/grouping only and MUST NOT be shown to the user
    as display text.
    """
    if not text:
        return ""
    s = str(text).strip()
    # Collapse internal whitespace to single spaces.
    s = _WS_PATTERN.sub(" ", s)
    # Remove leading/trailing punctuation and OCR noise.
    s = _EDGE_PATTERN.sub("", s)
    # Collapse again in case edge removal exposed spaces, then casefold.
    s = _WS_PATTERN.sub(" ", s).strip()
    return s.casefold()


_DIGIT_SEQ_RE = re.compile(r"\d+")


def is_same_subtitle_text(a: str, b: str, threshold: float = 0.85) -> bool:
    """Return True when two OCR strings represent the same subtitle cue.

    Guards against merging distinct cues that share most characters (e.g.
    ``Wait 10s`` vs ``Wait 20s``, ``Part A`` vs ``Part B``, or ``line one`` vs
    ``line two``) while still tolerating minor single-char OCR noise.
    """
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na and not nb:
        return True
    if not na or not nb:
        return False
    if na == nb:
        return True
    if SequenceMatcher(None, na, nb).ratio() < threshold:
        return False

    nums_a = _DIGIT_SEQ_RE.findall(na)
    nums_b = _DIGIT_SEQ_RE.findall(nb)
    if nums_a and nums_b and nums_a != nums_b:
        return False

    words_a = na.split()
    words_b = nb.split()
    if len(words_a) == len(words_b) and len(words_a) >= 2:
        for wa, wb in zip(words_a, words_b):
            if wa != wb and SequenceMatcher(None, wa, wb).ratio() < 0.45:
                return False

    return True


def text_similarity(a: str, b: str) -> float:
    """Return a 0..1 similarity ratio over normalized text.

    Two empties => 1.0; exactly one empty => 0.0.
    """
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _get_box(line) -> Tuple[float, float, float, float] | None:
    """Return (x, y, w, h) as floats if line carries a valid 4-element box."""
    box = None
    if hasattr(line, "box"):
        box = getattr(line, "box")
    elif isinstance(line, dict):
        box = line.get("box")
    if box is None:
        return None
    try:
        if len(box) < 4:
            return None
        x, y, w, h = box[0], box[1], box[2], box[3]
    except (TypeError, IndexError, KeyError):
        return None
    if not all(_is_num(v) for v in (x, y, w, h)):
        return None
    return (float(x), float(y), float(w), float(h))


def _get_text(line) -> str:
    if isinstance(line, dict):
        return str(line.get("text") or "")
    if hasattr(line, "text"):
        return str(getattr(line, "text") or "")
    return str(line or "")


def _is_cjk_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def _join_inline_chunks(chunks: List[str]) -> str:
    """Join text chunks within the same visual row without inserting spaces inside CJK."""
    acc = ""
    for raw in chunks:
        nxt = raw.strip()
        if not nxt:
            continue
        if not acc:
            acc = nxt
        elif _is_cjk_char(acc[-1]) and _is_cjk_char(nxt[0]):
            acc += nxt
        else:
            acc += " " + nxt
    return acc


def _cluster_row_bands(boxed_entries: List[tuple]) -> List[List[tuple]]:
    """Cluster boxed items into visual rows tolerant of minor vertical jitter."""
    if not boxed_entries:
        return []
    # Sort by vertical center yc, then x, then original index.
    sorted_entries = sorted(boxed_entries, key=lambda e: (e[6], e[2], e[0]))
    bands: List[List[tuple]] = []
    cur_band = [sorted_entries[0]]
    yc_sum = sorted_entries[0][6]
    h_sum = sorted_entries[0][5]
    for entry in sorted_entries[1:]:
        n = len(cur_band)
        band_yc = yc_sum / n
        band_h = h_sum / n
        if abs(entry[6] - band_yc) <= 0.5 * min(entry[5], band_h):
            cur_band.append(entry)
            yc_sum += entry[6]
            h_sum += entry[5]
        else:
            bands.append(sorted(cur_band, key=lambda e: (e[2], e[0])))
            cur_band = [entry]
            yc_sum = entry[6]
            h_sum = entry[5]
    bands.append(sorted(cur_band, key=lambda e: (e[2], e[0])))
    return bands


def _order_bands_column_aware(bands: List[List[tuple]], gutter_thresh: float) -> List[List[tuple]]:
    """Split multi-column layouts column-first when >= 3 rows share a clean vertical gutter."""
    if len(bands) < 3:
        return bands

    candidate_mids: List[float] = []
    for band in bands:
        if len(band) < 2:
            continue
        for i in range(len(band) - 1):
            prev_e = band[i]
            curr_e = band[i + 1]
            gap = curr_e[2] - prev_e[7]
            if gap > gutter_thresh:
                candidate_mids.append(0.5 * (prev_e[7] + curr_e[2]))

    if not candidate_mids:
        return bands

    best_split = None
    best_score = (0, 0.0)
    for mid in candidate_mids:
        left_bands: List[List[tuple]] = []
        right_bands: List[List[tuple]] = []
        max_left_x1 = None
        min_right_x0 = None
        straddles = False
        both_sides_count = 0

        for band in bands:
            l_items: List[tuple] = []
            r_items: List[tuple] = []
            for e in band:
                x0, x1 = e[2], e[7]
                if x1 <= mid:
                    l_items.append(e)
                    max_left_x1 = x1 if max_left_x1 is None else max(max_left_x1, x1)
                elif x0 >= mid:
                    r_items.append(e)
                    min_right_x0 = x0 if min_right_x0 is None else min(min_right_x0, x0)
                else:
                    straddles = True
                    break
            if straddles:
                break
            if l_items and r_items:
                both_sides_count += 1
            if l_items:
                left_bands.append(l_items)
            if r_items:
                right_bands.append(r_items)

        if straddles or max_left_x1 is None or min_right_x0 is None:
            continue
        gutter_width = min_right_x0 - max_left_x1
        if both_sides_count >= 3 and gutter_width > gutter_thresh:
            score = (both_sides_count, gutter_width)
            if score > best_score:
                best_score = score
                best_split = (left_bands, right_bands)

    if best_split is None:
        return bands

    left_bands, right_bands = best_split
    return _order_bands_column_aware(left_bands, gutter_thresh) + _order_bands_column_aware(
        right_bands, gutter_thresh
    )


def order_lines(lines: List, *, return_rows: bool = False):
    """Order OcrLine items in natural reading order (row-band + multi-column aware).

    When ``return_rows=True``, returns ``(unboxed, rows)`` where ``rows`` is a
    list of visual row item lists. By default returns a flat ordered list.
    """
    if not lines:
        return ([], []) if return_rows else (list(lines) if lines is not None else [])

    unboxed = []
    boxed_entries = []
    char_widths: List[float] = []

    for idx, item in enumerate(lines):
        box = _get_box(item)
        if box is None:
            unboxed.append(item)
            continue
        x, y, w, h = box
        h_eff = max(1.0, h)
        yc = y + 0.5 * h_eff
        x1 = x + max(0.0, w)
        boxed_entries.append((idx, item, x, y, w, h_eff, yc, x1))
        txt = _get_text(item).strip()
        if txt and w > 0:
            char_widths.append(max(4.0, w / max(1, len(txt))))

    bands = _cluster_row_bands(boxed_entries)
    med_cw = median(char_widths) if char_widths else 12.0
    gutter_thresh = max(24.0, 1.5 * med_cw)
    ordered_bands = _order_bands_column_aware(bands, gutter_thresh)
    rows = [[entry[1] for entry in band] for band in ordered_bands]

    if return_rows:
        return unboxed, rows
    return unboxed + [item for row in rows for item in row]


def join_ordered_lines(lines: List, *, paragraph_mode: str = "preserve_lines") -> str:
    """Format ordered OCR lines into coherent paragraph text."""
    if not lines:
        return ""
    unboxed, rows = order_lines(lines, return_rows=True)
    row_strings: List[str] = []
    for item in unboxed:
        t = _get_text(item).strip()
        if t:
            row_strings.append(t)
    for row in rows:
        row_str = _join_inline_chunks([_get_text(item) for item in row])
        if row_str:
            row_strings.append(row_str)

    if not row_strings:
        return ""
    if paragraph_mode != "unwrap":
        return "\n".join(row_strings)

    acc = row_strings[0]
    for nxt in row_strings[1:]:
        if len(acc) >= 2 and acc[-1] == "-" and acc[-2].isascii() and acc[-2].isalnum() and nxt[0].isascii() and nxt[0].islower():
            acc = acc[:-1] + nxt
        elif _is_cjk_char(acc[-1]) and _is_cjk_char(nxt[0]):
            acc += nxt
        else:
            acc += " " + nxt
    return acc


def _obs_text_conf(obs):
    """Extract (text, confidence) from an observation tuple/list/mapping."""
    if isinstance(obs, dict):
        return obs.get("text", ""), float(obs.get("confidence", 0.0) or 0.0)
    text = obs[0]
    conf = 0.0
    if len(obs) > 1 and obs[1] is not None:
        conf = float(obs[1])
    return text, conf


def representative_text(observations: List) -> str:
    """Pick the best-agreement/confidence text from (text, confidence) obs.

    Group observations by normalized comparison key. Choose the group with the
    highest cumulative confidence, breaking ties by group size (count), then by
    the highest single-observation confidence. Within the chosen group, return
    the raw text of the highest-confidence observation.
    """
    if not observations:
        return ""

    groups = {}  # norm_key -> {"sum": float, "count": int, "best": (conf, text)}
    for obs in observations:
        text, conf = _obs_text_conf(obs)
        key = normalize_text(text)
        if key == "":
            # Skip empty observations for representative selection.
            continue
        g = groups.get(key)
        if g is None:
            groups[key] = {"sum": conf, "count": 1, "best": (conf, text)}
        else:
            g["sum"] += conf
            g["count"] += 1
            if conf >= g["best"][0]:
                g["best"] = (conf, text)

    if not groups:
        return ""

    # Rank groups: cumulative confidence, then count, then best single conf.
    best_key = None
    best_rank = None
    for key, g in groups.items():
        rank = (g["sum"], g["count"], g["best"][0])
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_key = key
    return groups[best_key]["best"][1]
