"""Hard-Subtitle OCR Source package.

Mirrors the small-scale structure of videotrans/recognition/__init__.py: a
provider-id constant, a lazy provider registry keyed by id, and a `run` entry
that resolves the provider on demand via `get_class`.

This module is import-side-effect free and must NOT eagerly import any heavy
dependency (paddleocr, cv2, numpy, torch) or the paddle provider
module. The paddle provider (imp="._paddle") is registered but only imported
lazily inside `run`.
"""
from collections import OrderedDict
import threading
from typing import Optional

from videotrans import ChannelProvider, get_class

# Public re-exports.
from videotrans.ocr._types import OcrConfig, OcrResult, OcrSegment, OcrLine
from videotrans.ocr._segment import SegmentBuilder
from videotrans.ocr._text import (
    normalize_text,
    text_similarity,
    order_lines,
    join_ordered_lines,
    representative_text,
)
from videotrans.ocr._srt_adapter import segments_to_srt_items
from videotrans.ocr._scanner import (
    crop_roi,
    gray_thumbnail,
    change_metric,
    OcrScanner,
)
from videotrans.ocr._checkpoint import (
    OcrCheckpoint,
    SCHEMA_VERSION,
    video_fingerprint,
    save_checkpoint,
    load_checkpoint,
    is_compatible,
    dedup_on_resume,
)

# Provider-id constants.
PADDLE_OCR = 0

# Lazy provider registry. The paddle provider module is added in a later task;
# reference imp="._paddle" WITHOUT importing it here.
_ID_NAME_DICT = {
    PADDLE_OCR: ChannelProvider("PaddleOCR", imp="._paddle"),
}

OCR_NAME_LIST = [it.name for it in _ID_NAME_DICT.values()]

__all__ = [
    "PADDLE_OCR",
    "OcrConfig",
    "OcrResult",
    "OcrSegment",
    "OcrLine",
    "SegmentBuilder",
    "normalize_text",
    "text_similarity",
    "order_lines",
    "join_ordered_lines",
    "representative_text",
    "segments_to_srt_items",
    "OcrCheckpoint",
    "SCHEMA_VERSION",
    "video_fingerprint",
    "save_checkpoint",
    "load_checkpoint",
    "is_compatible",
    "dedup_on_resume",
    "crop_roi",
    "gray_thumbnail",
    "change_metric",
    "OcrScanner",
    "get_provider",
    "get_cached_provider",
    "release_cached_providers",
    "scan_video_to_srt",
    "run",
]

_PROVIDER_CACHE: OrderedDict = OrderedDict()
_PROVIDER_CACHE_LOCK = threading.Lock()
_PROVIDER_CACHE_MAXSIZE = 2


def get_provider(*, provider=PADDLE_OCR, device="auto", **kwargs):
    """Build a concrete OCR provider instance (lazy import)."""
    _cls = get_class(provider, "ocr", _ID_NAME_DICT)
    if not _cls:
        raise RuntimeError(f"No this OCR Channel:{provider=}")
    return _cls(device=device, **kwargs)


_ORIG_GET_PROVIDER = get_provider


def _resolve_auto_device(provider=PADDLE_OCR) -> str:
    """Resolve ``"auto"`` to ``"gpu"`` or ``"cpu"`` without importing at module load."""
    try:
        _cls = get_class(provider, "ocr", _ID_NAME_DICT)
        if _cls is not None and hasattr(_cls, "select_device"):
            resolved = str(_cls(device="auto").select_device("auto") or "cpu").strip().lower()
            if resolved in ("cpu", "gpu"):
                return resolved
    except Exception:
        pass
    return "cpu"


def get_cached_provider(*, provider=PADDLE_OCR, device="auto", **kwargs):
    """Return a thread-safe bounded LRU-cached OCR provider instance.

    Bypasses the cache if ``videotrans.ocr.get_provider`` has been monkeypatched
    by tests. Normalizes default OCR kwargs and resolves ``"auto"`` to the
    concrete device (``"cpu"`` or ``"gpu"``) so ``device="auto"`` and explicit
    ``device="cpu"`` share the same cached provider on CPU-only hosts.
    """
    if get_provider is not _ORIG_GET_PROVIDER:
        return get_provider(provider=provider, device=device, **kwargs)
    extra = dict(kwargs)
    use_angle_cls = bool(extra.pop("use_angle_cls", False))
    paragraph_mode = str(extra.pop("paragraph_mode", "preserve_lines") or "preserve_lines")
    cpu_threads_raw = extra.pop("cpu_threads", None)
    cpu_threads = int(cpu_threads_raw) if cpu_threads_raw is not None else None
    dev_norm = str(device or "auto").strip().lower()
    if dev_norm == "auto":
        dev_norm = _resolve_auto_device(provider)
    key = (
        provider,
        dev_norm,
        use_angle_cls,
        paragraph_mode,
        cpu_threads,
        tuple(sorted(extra.items())),
    )
    with _PROVIDER_CACHE_LOCK:
        cached = _PROVIDER_CACHE.get(key)
        if cached is not None:
            _PROVIDER_CACHE.move_to_end(key)
            return cached
        inst = _ORIG_GET_PROVIDER(
            provider=provider,
            device=dev_norm,
            use_angle_cls=use_angle_cls,
            paragraph_mode=paragraph_mode,
            cpu_threads=cpu_threads,
            **extra,
        )
        _PROVIDER_CACHE[key] = inst
        _PROVIDER_CACHE.move_to_end(key)
        while len(_PROVIDER_CACHE) > _PROVIDER_CACHE_MAXSIZE:
            _PROVIDER_CACHE.popitem(last=False)
        return inst


def release_cached_providers() -> None:
    """Release cached OCR provider instances (e.g., on app shutdown)."""
    with _PROVIDER_CACHE_LOCK:
        _PROVIDER_CACHE.clear()


def scan_video_to_srt(
    video,
    *,
    roi,
    language=None,
    device="auto",
    ffmpeg="ffmpeg",
    checkpoint_path=None,
    on_progress=None,
    progress_callback=None,
    provider_obj=None,
    frames=None,
    duration_ms=None,
    coarse_interval_ms=500,
    stop_event=None,
    job_id=None,
    use_angle_cls=False,
    paragraph_mode="preserve_lines",
    cpu_threads=None,
    max_roi_width=1280,
):
    """Scan a video ROI with OCR and convert segments to SRT contract items.

    Returns (srt_items, segments). ``frames`` may be injected (tests); when
    omitted it streams cropped ROI frames from ``video`` via a single ffmpeg
    pipe with PyAV/frame_at_fast boundary refinement. ``provider_obj`` may be
    injected (tests); when omitted a cached Paddle provider is used.
    """
    cfg = OcrConfig(
        provider="paddle",
        roi=(target_roi := tuple(roi)),
        language=language,
        device=device,
        coarse_interval_ms=coarse_interval_ms,
        use_angle_cls=bool(use_angle_cls),
        paragraph_mode=paragraph_mode or "preserve_lines",
        cpu_threads=cpu_threads,
        max_roi_width=int(max_roi_width),
    )
    if not cfg.is_valid_roi():
        raise ValueError(f"Invalid OCR ROI for scan: {cfg.roi}")

    prog_cb = progress_callback or on_progress
    active_job_id = job_id
    if active_job_id is None:
        try:
            from videotrans.core.proc_registry import get_current_job_id
            active_job_id = get_current_job_id()
        except Exception:
            active_job_id = None

    reader = None
    frames_are_cropped = False
    frame_fetcher = None
    if frames is None:
        from videotrans.ocr._frame_source import (
            PyAVFastReader,
            frame_at_fast,
            probe_video_meta,
            sampled_roi_iter,
        )

        meta = probe_video_meta(str(video), ffmpeg)
        if duration_ms is None:
            duration_ms = int(meta.get("duration_ms") or 0)
        frames = sampled_roi_iter(
            ffmpeg,
            str(video),
            roi=target_roi,
            coarse_interval_ms=coarse_interval_ms,
            duration_ms=duration_ms or 0,
            max_roi_width=cfg.max_roi_width,
            stop_event=stop_event,
            job_id=active_job_id,
            meta=meta,
        )
        frames_are_cropped = True
        try:
            reader = PyAVFastReader(str(video), target_roi, max_roi_width=cfg.max_roi_width)
        except Exception:
            reader = None
        if reader is not None:
            frame_fetcher = reader.frame_at
        else:
            frame_fetcher = lambda ts: frame_at_fast(
                str(video),
                ts,
                roi=target_roi,
                max_roi_width=cfg.max_roi_width,
                ffmpeg_exe=ffmpeg,
                meta=meta,
            )

    if provider_obj is None:
        provider_obj = get_cached_provider(
            provider=PADDLE_OCR,
            device=device,
            use_angle_cls=cfg.use_angle_cls,
            paragraph_mode=cfg.paragraph_mode,
            cpu_threads=cfg.cpu_threads,
        )

    from videotrans.ocr._checkpoint import video_fingerprint
    cfg.fingerprint = video_fingerprint(video, duration_ms)

    try:
        scanner = OcrScanner(
            provider_obj,
            cfg,
            checkpoint_path=checkpoint_path,
            on_progress=prog_cb,
            frames_are_cropped=frames_are_cropped,
            frame_fetcher=frame_fetcher,
        )
        segments = scanner.scan(
            frames,
            duration_ms=duration_ms or 0,
            stop_event=stop_event,
            resume=checkpoint_path is not None,
        )
    finally:
        if reader is not None:
            reader.close()

    from videotrans.ocr._srt_adapter import segments_to_srt_items
    return segments_to_srt_items(segments), segments

def run(*, provider=PADDLE_OCR, image=None, language=None, **kwargs) -> OcrResult:
    """Unified OCR entry point.

    Lazily resolves the provider via ``get_class`` and calls its
    ``recognize(image, language)``. No provider module is imported at package
    import time, so importing ``videotrans.ocr`` never triggers importing the
    paddle backend.
    """
    _cls = get_class(provider, "ocr", _ID_NAME_DICT)
    if not _cls:
        raise RuntimeError(f"No this OCR Channel:{provider=}")
    return _cls(**kwargs).recognize(image, language)
