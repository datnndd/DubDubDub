"""Pure-stdlib OCR data contracts for the Hard-Subtitle OCR Source feature.

These dataclasses mirror the dict-style access pattern used by
``videotrans.task.taskcfg.SrtItem`` so downstream code can treat them either as
attribute-bearing objects (``obj.text``) or as dict-like mappings
(``obj["text"]`` / ``obj.get("text")``).

This module must remain import-side-effect free and must NOT import any heavy
dependency (paddleocr, cv2, numpy, torch). Only Python stdlib is used.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class _DictAccessMixin:
    """Provide SrtItem-style dict access on top of a dataclass."""

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class OcrLine(_DictAccessMixin):
    """A single OCR-detected line of text.

    ``box`` is stored exactly as given. It is conventionally
    ``(x, y, w, h)`` either normalized (0..1) or in pixels; the value is not
    re-interpreted here.
    """

    text: str
    confidence: float = 0.0
    box: Optional[Tuple] = None


@dataclass
class OcrConfig(_DictAccessMixin):
    """Grouped OCR task configuration.

    ``roi`` is ``(x, y, width, height)`` normalized to the 0..1 range.
    """

    provider: str = "paddle"
    roi: Tuple = (0.0, 0.85, 1.0, 0.15)
    language: Optional[str] = None
    device: str = "auto"
    coarse_interval_ms: int = 500
    boundary_interval_ms: int = 150
    similarity_threshold: float = 0.85
    min_stable_samples: int = 2
    grace_gap_ms: int = 400
    use_angle_cls: bool = False
    paragraph_mode: str = "preserve_lines"
    cpu_threads: Optional[int] = None
    max_roi_width: int = 1280

    def is_valid_roi(self) -> bool:
        """Return True only when the ROI is a well-formed normalized box.

        Requires x, y, w, h all numbers, w > 0, h > 0, and the whole box to
        stay within 0..1 (i.e. x >= 0, y >= 0, x + w <= 1, y + h <= 1).
        """
        roi = self.roi
        if roi is None:
            return False
        try:
            if len(roi) != 4:
                return False
        except TypeError:
            return False
        x, y, w, h = roi
        if not all(_is_number(v) for v in (x, y, w, h)):
            return False
        if w <= 0 or h <= 0:
            return False
        if x < 0 or y < 0:
            return False
        if x + w > 1 or y + h > 1:
            return False
        return True


@dataclass
class OcrResult(_DictAccessMixin):
    """The recognition result for one sampled frame."""

    text: str = ""
    confidence: float = 0.0
    lines: List = field(default_factory=list)
    timestamp_ms: int = 0
    device: Optional[str] = None


@dataclass
class OcrSegment(_DictAccessMixin):
    """A consolidated subtitle segment produced by the segment builder."""

    start_ms: int = 0
    end_ms: int = 0
    text: str = ""
    confidence: float = 0.0
    samples: int = 0
