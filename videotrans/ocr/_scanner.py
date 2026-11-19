"""Streamed, memory-bounded OCR scanner with checkpoint/resume.

The scanner decodes a video sequentially (via an injected frame iterator),
crops only the ROI in memory, samples coarsely, uses a cheap grayscale change
metric to run OCR selectively, refines transitions locally, and feeds a
SegmentBuilder. It persists an atomic checkpoint after each closed segment and
periodically, and can resume from a compatible checkpoint with a short overlap.

This module uses only Python stdlib + numpy. It never imports PaddleOCR, cv2,
or torch. A frame iterator yields ``(timestamp_ms, frame_bgr)`` tuples; the
production producer lives in ``_frame_source`` and tests inject small lists.
"""
from collections import OrderedDict
import hashlib
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np

from videotrans.ocr._checkpoint import (
    OcrCheckpoint,
    dedup_on_resume,
    is_compatible,
    load_checkpoint,
    save_checkpoint,
)
from videotrans.ocr._segment import SegmentBuilder
from videotrans.ocr._text import is_same_subtitle_text
from videotrans.ocr._types import OcrConfig, OcrResult, OcrSegment

# Default fallback when config does not provide a checkpoint interval.
_CHECKPOINT_INTERVAL_MS = 30_000
_FINE_STEP_MS = 150
# Frames are decoded without scaling; thumbnails are scaled by block averaging.
_GRID_W = 24
_GRID_H = 8
_DEFAULT_CHANGE_THRESHOLD = 0.02
# Sparse confidence-gated heartbeat when dedup skips frames.
_HEARTBEAT_EVERY = 20
_HEARTBEAT_CONF = 0.85
_LRU_CACHE_MAXSIZE = 32


def gray_thumbnail(roi_bgr, grid_w: int = _GRID_W, grid_h: int = _GRID_H) -> np.ndarray:
    """Downsample a BGR crop to a small normalized grayscale thumb (0..1).

    Pure numpy; stable memory regardless of video duration because only one
    thumbnail is held at a time.
    """
    if roi_bgr is None:
        return _empty_thumb(grid_w, grid_h)
    arr = np.asarray(roi_bgr)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3:
        gray = (
            0.299 * arr[..., 2].astype(np.float32)
            + 0.587 * arr[..., 1].astype(np.float32)
            + 0.114 * arr[..., 0].astype(np.float32)
        )
    else:
        return _empty_thumb(grid_w, grid_h)
    h, w = gray.shape
    if h == 0 or w == 0:
        return _empty_thumb(grid_w, grid_h)
    # Block-averaged downsample to grid.
    if h >= grid_h and w >= grid_w:
        rows_per = h // grid_h
        cols_per = w // grid_w
        uh = rows_per * grid_h
        uw = cols_per * grid_w
        resized = (
            gray[:uh, :uw]
            .reshape(grid_h, rows_per, grid_w, cols_per)
            .mean(axis=(1, 3))
        )
    else:
        # Smaller than grid in at least one axis: nearest resize.
        resized = _resize_nearest(gray, grid_h, grid_w)
    lo, hi = float(resized.min()), float(resized.max())
    if hi - lo < 1e-6:
        return np.zeros_like(resized, dtype=np.float32)
    return ((resized.astype(np.float32) - lo) / (hi - lo)).astype(np.float32)


def _empty_thumb(grid_w, grid_h):
    return np.zeros((grid_h, grid_w), dtype=np.float32)


def _resize_nearest(gray, out_h, out_w):
    h, w = gray.shape
    ys = (np.arange(out_h) * h / out_h).astype(np.int64)
    xs = (np.arange(out_w) * w / out_w).astype(np.int64)
    return gray[np.ix_(np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1))].astype(np.float32)


def change_metric(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two normalized thumbs (0..1)."""
    if a is None or b is None:
        return 1.0
    return float(np.mean(np.abs(a - b)))


def crop_roi(frame, roi) -> Optional[np.ndarray]:
    """Crop a normalized (x, y, w, h) ROI from a frame, returning a fresh copy."""
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim not in (2, 3):
        return None
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return None
    x, y, rw, rh = roi
    x0 = int(round(x * w))
    y0 = int(round(y * h))
    x1 = min(w, int(round((x + rw) * w)))
    y1 = min(h, int(round((y + rh) * h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array(arr[y0:y1, x0:x1], copy=True)


class _RoiSignature(NamedTuple):
    tile_luma: np.ndarray
    tile_edge: np.ndarray
    small_i16: np.ndarray
    patch_u8: np.ndarray
    dhash: int
    sha1: bytes


def _to_gray_f32(roi_bgr) -> Optional[np.ndarray]:
    if roi_bgr is None:
        return None
    arr = np.asarray(roi_bgr)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3:
        gray = (
            0.299 * arr[..., 2].astype(np.float32)
            + 0.587 * arr[..., 1].astype(np.float32)
            + 0.114 * arr[..., 0].astype(np.float32)
        )
    else:
        return None
    if gray.shape[0] <= 0 or gray.shape[1] <= 0:
        return None
    return gray


def _downsample_grid(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = arr.shape
    if h >= out_h and w >= out_w:
        rows_per = h // out_h
        cols_per = w // out_w
        uh = rows_per * out_h
        uw = cols_per * out_w
        return arr[:uh, :uw].reshape(out_h, rows_per, out_w, cols_per).mean(axis=(1, 3))
    return _resize_nearest(arr, out_h, out_w)


def _downsample_patch_u8(gray: np.ndarray, target_h: int = 64, target_w: int = 320) -> np.ndarray:
    h, w = gray.shape
    out_h = min(h, target_h)
    out_w = min(w, target_w)
    if h == out_h and w == out_w:
        return np.rint(gray).clip(0, 255).astype(np.uint8)
    ys = np.clip((np.arange(out_h) * h / out_h).astype(np.int64), 0, h - 1)
    xs = np.clip((np.arange(out_w) * w / out_w).astype(np.int64), 0, w - 1)
    return np.rint(gray[np.ix_(ys, xs)]).clip(0, 255).astype(np.uint8)


def _roi_signature(crop_bgr, tile_w: int = 16, tile_h: int = 4) -> Optional[_RoiSignature]:
    gray = _to_gray_f32(crop_bgr)
    if gray is None:
        return None
    h, w = gray.shape
    tile_mean = _downsample_grid(gray, tile_h, tile_w)
    tile_luma = (tile_mean / 255.0).astype(np.float32)
    small_i16 = np.rint(tile_mean).astype(np.int16)

    if w > 1:
        grad_x = np.abs(gray[:, 1:] - gray[:, :-1])
    else:
        grad_x = np.zeros_like(gray)
    tile_edge = (_downsample_grid(grad_x, tile_h, tile_w) / 255.0).astype(np.float32)

    g9 = _downsample_grid(gray, 8, 9)
    bits = (g9[:, 1:] > g9[:, :-1]).reshape(-1)
    dhash = 0
    for b in bits:
        dhash = (dhash << 1) | int(bool(b))

    patch_u8 = _downsample_patch_u8(gray)
    edge_u8 = np.clip(np.rint(tile_edge * 255.0), 0, 255).astype(np.uint8)
    sha1 = hashlib.sha1(small_i16.tobytes() + edge_u8.tobytes()).digest()
    return _RoiSignature(
        tile_luma=tile_luma,
        tile_edge=tile_edge,
        small_i16=small_i16,
        patch_u8=patch_u8,
        dhash=dhash,
        sha1=sha1,
    )


def _sig_is_static(prev_sig: Optional[_RoiSignature], cur_sig: Optional[_RoiSignature]) -> bool:
    if prev_sig is None or cur_sig is None:
        return False
    dhash_ham = (prev_sig.dhash ^ cur_sig.dhash).bit_count()
    if dhash_ham > 2:
        return False
    max_tile_mad = float(np.max(np.abs(cur_sig.tile_luma - prev_sig.tile_luma)))
    if max_tile_mad >= 0.012:
        return False
    max_tile_edge = float(np.max(np.abs(cur_sig.tile_edge - prev_sig.tile_edge)))
    if max_tile_edge >= 0.015:
        return False
    max_px_diff = float(np.max(np.abs(cur_sig.small_i16 - prev_sig.small_i16)))
    if max_px_diff >= 12.0:
        return False
    if prev_sig.patch_u8.shape == cur_sig.patch_u8.shape:
        pdiff = np.abs(cur_sig.patch_u8.astype(np.int16) - prev_sig.patch_u8.astype(np.int16))
        if int(np.count_nonzero(pdiff > 24)) >= 6:
            return False
        if float(np.mean(pdiff)) >= 8.0:
            return False
    return True


def _same_state(norm_a: str, norm_b: str, sim_thresh: float) -> bool:
    return is_same_subtitle_text(norm_a, norm_b, sim_thresh)


class OcrScanner:
    """Stream OCR over frames, consolidate segments, persist checkpoints.

    ``frames`` is an iterable of ``(timestamp_ms, frame_bgr)``. In production a
    video frame producer supplies it; tests inject small lists.
    """

    def __init__(
        self,
        provider,
        config: OcrConfig,
        *,
        checkpoint_path: Optional[str] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
        frames_are_cropped: bool = False,
        frame_fetcher: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.on_progress = on_progress
        self._frames_are_cropped = bool(frames_are_cropped)
        self._frame_fetcher = frame_fetcher
        self._builder = SegmentBuilder(config)
        self._last_sig: Optional[_RoiSignature] = None
        self._last_ocr_ts = None
        self._last_coarse_ts = None
        self._last_norm = ""
        self._last_conf = 0.0
        self._coarse_steps = 0
        self._last_ckpt_ms = 0
        self._stopped = False
        self._duration_ms = 0
        init_dev = self._resolve_provider_device()
        self._stats = {
            "frames": 0,
            "ocr_calls": 0,
            "skipped_frames": 0,
            "device": init_dev,
            "cpu": init_dev,
        }
        self._saved_last_processed_ms = 0
        self._preexisting = []
        self._lru_cache: OrderedDict[int, tuple[_RoiSignature, OcrResult]] = OrderedDict()
        self._lru_seq = 0

    def _resolve_provider_device(self) -> str:
        active = getattr(self.provider, "_active_device", None)
        if active in ("cpu", "gpu"):
            return active
        pref = getattr(self.provider, "_prefer_device", None)
        if pref in ("cpu", "gpu"):
            return pref
        return "cpu"

    # -- progress ---------------------------------------------------------
    def _report(self, ts_ms, duration_ms):
        if not self.on_progress:
            return
        self.on_progress(
            {
                "timestamp_ms": ts_ms,
                "duration_ms": duration_ms,
                "percent": (ts_ms / duration_ms * 100.0) if duration_ms else 0.0,
                "device": self._stats.get("device", "cpu"),
                "segments": self._count_segments(),
                "latest_text": self._last_norm,
                "ocr_calls": self._stats["ocr_calls"],
                "skipped_frames": self._stats["skipped_frames"],
            }
        )

    def _count_segments(self):
        try:
            completed = list(self._builder._segments)
            if self._preexisting:
                completed = dedup_on_resume(
                    self._preexisting,
                    completed,
                    similarity_threshold=self.config.similarity_threshold,
                )
            return len(completed)
        except Exception:
            return 0

    # -- main scan --------------------------------------------------------
    def scan(
        self,
        frames: Iterable[Tuple[int, np.ndarray]],
        duration_ms: int = 0,
        *,
        stop_event: Optional[Callable[[], bool]] = None,
        resume: bool = True,
    ) -> List:
        """Run the scan over ``frames`` and return the finalized segments."""
        roi = self.config.roi
        if not self.config.is_valid_roi():
            raise ValueError(f"Invalid ROI for OCR scan: {self.config.roi}")

        self._duration_ms = int(duration_ms or 0)
        self._stopped = False

        loaded_ckpt = self._maybe_load_checkpoint(resume)
        skip_until_ms = 0
        self._preexisting = []
        if loaded_ckpt is not None:
            overlap = self.config.coarse_interval_ms * 4 + 3000
            skip_until_ms = max(0, loaded_ckpt.last_processed_ms - overlap)
            self._preexisting = [self._dict_to_seg(x) for x in loaded_ckpt.segments]

        for ts_ms, frame in frames:
            ts_ms = int(ts_ms)
            if stop_event is not None and stop_event():
                self._stopped = True
                break
            self._stats["frames"] += 1
            if ts_ms < skip_until_ms:
                continue
            self._process_frame(ts_ms, frame, roi, duration_ms)
            if self._coarse_steps % 4 == 0:
                self._maybe_periodic_checkpoint(ts_ms)

        if stop_event is not None and stop_event():
            self._stopped = True

        # Stop event or exhausted input: persist final checkpoint and finalize.
        return self.finalize(duration_ms, persist=True)

    # -- per-frame processing ---------------------------------------------
    def _process_frame(self, ts_ms, frame, roi, duration_ms):
        crop = frame if self._frames_are_cropped else crop_roi(frame, roi)
        sig = _roi_signature(crop)

        if self._last_sig is None:
            result, new_norm, new_conf = self._ocr_raw(ts_ms, crop, sig)
            self._builder.observe(ts_ms, result)
            self._last_ocr_ts = ts_ms
            self._last_coarse_ts = ts_ms
            self._last_sig = sig
            self._last_norm = new_norm
            self._last_conf = new_conf
            self._coarse_steps += 1
            self._report(ts_ms, duration_ms)
            return

        if self._last_coarse_ts is not None and ts_ms <= self._last_coarse_ts:
            self._report(ts_ms, duration_ms)
            return

        self._coarse_steps += 1
        is_static = _sig_is_static(self._last_sig, sig)
        needs_heartbeat = (
            self._coarse_steps % _HEARTBEAT_EVERY == 0
            and self._last_conf < _HEARTBEAT_CONF
        )
        if is_static and not needs_heartbeat:
            self._stats["skipped_frames"] += 1
            self._builder.on_dedup_skip(ts_ms)
            self._last_coarse_ts = ts_ms
            self._report(ts_ms, duration_ms)
            return

        prev_norm = self._last_norm
        prev_coarse_ts = self._last_coarse_ts
        result, new_norm, new_conf = self._ocr_raw(ts_ms, crop, sig)

        changed = not _same_state(prev_norm, new_norm, self.config.similarity_threshold)
        eff_res, eff_norm, eff_conf, eff_sig = result, new_norm, new_conf, sig
        after_probes: List[Tuple[int, OcrResult, str, float, Optional[_RoiSignature]]] = []
        if (
            changed
            and prev_coarse_ts is not None
            and (ts_ms - prev_coarse_ts) > min(125, max(1, int(self.config.boundary_interval_ms)))
            and self._frame_fetcher is not None
        ):
            probes_ext, eff_res, eff_norm, eff_conf, eff_sig = self._refine_transition_ext(
                prev_coarse_ts,
                ts_ms,
                prev_norm,
                to_sig=sig,
                to_res=result,
                to_norm=new_norm,
                to_conf=new_conf,
                duration_ms=duration_ms,
            )
            for p_ts, p_res, p_norm, p_conf, p_sig in probes_ext:
                if p_ts < ts_ms:
                    self._builder.observe(p_ts, p_res)
                elif p_ts > ts_ms:
                    after_probes.append((p_ts, p_res, p_norm, p_conf, p_sig))

        self._builder.observe(ts_ms, eff_res)
        last_ts = ts_ms
        for p_ts, p_res, p_norm, p_conf, p_sig in after_probes:
            self._builder.observe(p_ts, p_res)
            eff_norm, eff_conf, eff_sig = p_norm, p_conf, p_sig
            last_ts = p_ts

        self._last_norm = eff_norm
        self._last_conf = eff_conf
        self._last_ocr_ts = last_ts
        self._last_coarse_ts = last_ts
        self._last_sig = eff_sig
        self._report(ts_ms, duration_ms)

    def _ocr_raw(
        self,
        ts_ms: int,
        crop: Optional[np.ndarray],
        sig: Optional[_RoiSignature] = None,
    ) -> Tuple[OcrResult, str, float]:
        if crop is None:
            return OcrResult(timestamp_ms=ts_ms), "", 0.0
        if sig is None:
            sig = _roi_signature(crop)
        if sig is not None:
            for k in reversed(list(self._lru_cache.keys())):
                cached_sig, c_res = self._lru_cache[k]
                if _sig_is_static(cached_sig, sig):
                    self._lru_cache.move_to_end(k)
                    res = OcrResult(
                        text=c_res.text,
                        confidence=c_res.confidence,
                        lines=list(c_res.lines),
                        timestamp_ms=ts_ms,
                        device=c_res.device,
                    )
                    return res, _normalize_for_track(res.text), float(res.confidence or 0.0)

        result = self.provider.recognize(crop, self.config.language)
        if result is None:
            result = OcrResult(timestamp_ms=ts_ms)
        result.timestamp_ms = ts_ms
        self._stats["ocr_calls"] += 1

        dev = getattr(result, "device", None) or getattr(self.provider, "_active_device", None)
        if dev in ("cpu", "gpu"):
            self._stats["device"] = dev
            self._stats["cpu"] = dev

        norm = _normalize_for_track(result.text)
        conf = float(getattr(result, "confidence", 0.0) or 0.0)

        if sig is not None:
            self._lru_seq += 1
            k = self._lru_seq
            self._lru_cache[k] = (
                sig,
                OcrResult(
                    text=result.text,
                    confidence=conf,
                    lines=list(getattr(result, "lines", []) or []),
                    timestamp_ms=0,
                    device=getattr(result, "device", None),
                ),
            )
            while len(self._lru_cache) > _LRU_CACHE_MAXSIZE:
                self._lru_cache.popitem(last=False)

        return result, norm, conf

    def _refine_transition(
        self,
        from_ts: int,
        to_ts: int,
        prev_norm: str,
    ) -> List[Tuple[int, OcrResult]]:
        """Binary-search the transition interval [from_ts, to_ts] before commit."""
        probes_ext, _res, _norm, _conf, _sig = self._refine_transition_ext(
            from_ts,
            to_ts,
            prev_norm,
            duration_ms=self._duration_ms,
        )
        return [(p_ts, p_res) for (p_ts, p_res, _pn, _pc, _ps) in probes_ext]

    def _refine_transition_ext(
        self,
        from_ts: int,
        to_ts: int,
        prev_norm: str,
        *,
        to_sig: Optional[_RoiSignature] = None,
        to_res: Optional[OcrResult] = None,
        to_norm: Optional[str] = None,
        to_conf: Optional[float] = None,
        duration_ms: int = 0,
    ) -> Tuple[
        List[Tuple[int, OcrResult, str, float, Optional[_RoiSignature]]],
        OcrResult,
        str,
        float,
        Optional[_RoiSignature],
    ]:
        """Binary-search transition bracket, verifying ``to_ts`` via ``frame_fetcher``.

        Converges to a bracket of width ``<= min(125, boundary_interval_ms)`` ms.
        If ``frame_fetcher(to_ts)`` disagrees with the coarse frame at ``to_ts``
        and still matches ``prev_norm``, searches forward into
        ``[to_ts, to_ts + coarse_span]`` before bisecting so non-500ms-aligned
        transitions stay within the ``<= 250ms`` SLA.
        """
        lo = int(from_ts)
        hi = int(to_ts)
        step_limit = min(125, max(1, int(self.config.boundary_interval_ms)))
        sim_thresh = self.config.similarity_threshold
        probes: List[Tuple[int, OcrResult, str, float, Optional[_RoiSignature]]] = []

        eff_res = to_res if to_res is not None else OcrResult(timestamp_ms=hi)
        eff_norm = to_norm if to_norm is not None else ""
        eff_conf = float(to_conf or 0.0)
        eff_sig = to_sig

        if self._frame_fetcher is None:
            return probes, eff_res, eff_norm, eff_conf, eff_sig

        fetched_to = self._frame_fetcher(hi)
        if fetched_to is not None:
            ft_crop = fetched_to if self._frames_are_cropped else crop_roi(fetched_to, self.config.roi)
            ft_sig = _roi_signature(ft_crop)
            if to_res is None or ft_sig is None or eff_sig is None or not _sig_is_static(ft_sig, eff_sig):
                eff_res, eff_norm, eff_conf = self._ocr_raw(hi, ft_crop, ft_sig)
                eff_sig = ft_sig

        should_bisect = True
        if _same_state(eff_norm, prev_norm, sim_thresh):
            # Coarse stream at to_ts was shifted ahead; bracket forward from to_ts.
            should_bisect = False
            coarse_span = max(step_limit * 2, hi - lo)
            max_fwd = hi + coarse_span
            if duration_ms > 0:
                max_fwd = min(max_fwd, int(duration_ms))
            f_step = max(step_limit, coarse_span // 2)
            cur_lo = hi
            for _ in range(3):
                next_ts = min(max_fwd, cur_lo + f_step)
                if next_ts <= cur_lo:
                    break
                nf = self._frame_fetcher(next_ts)
                if nf is None:
                    break
                nf_crop = nf if self._frames_are_cropped else crop_roi(nf, self.config.roi)
                nf_sig = _roi_signature(nf_crop)
                nf_res, nf_norm, nf_conf = self._ocr_raw(next_ts, nf_crop, nf_sig)
                probes.append((next_ts, nf_res, nf_norm, nf_conf, nf_sig))
                if _same_state(nf_norm, prev_norm, sim_thresh):
                    cur_lo = next_ts
                else:
                    lo = cur_lo
                    hi = next_ts
                    should_bisect = True
                    break

        if should_bisect:
            max_probes = 6
            n_probes = 0
            while (hi - lo) > step_limit and n_probes < max_probes:
                mid = (lo + hi) // 2
                if mid <= lo or mid >= hi:
                    break
                n_probes += 1
                mid_frame = self._frame_fetcher(mid)
                if mid_frame is None:
                    break
                mid_crop = mid_frame if self._frames_are_cropped else crop_roi(mid_frame, self.config.roi)
                mid_sig = _roi_signature(mid_crop)
                mid_res, mid_norm, mid_conf = self._ocr_raw(mid, mid_crop, mid_sig)
                probes.append((mid, mid_res, mid_norm, mid_conf, mid_sig))
                if _same_state(mid_norm, prev_norm, sim_thresh):
                    lo = mid
                else:
                    hi = mid

        probes.sort(key=lambda item: item[0])
        return probes, eff_res, eff_norm, eff_conf, eff_sig

    def _maybe_periodic_checkpoint(self, ts_ms):
        if not self.checkpoint_path:
            return
        if ts_ms - self._last_ckpt_ms >= getattr(self.config, "checkpoint_interval_ms", _CHECKPOINT_INTERVAL_MS):
            self._save_checkpoint(ts_ms, last_error=None)
            self._last_ckpt_ms = ts_ms

    # -- checkpoint -------------------------------------------------------
    def _maybe_load_checkpoint(self, resume):
        if not resume or not self.checkpoint_path:
            return None
        cp = load_checkpoint(self.checkpoint_path)
        if cp is None:
            return None
        fp = self._fingerprint()
        if not is_compatible(
            cp,
            fingerprint=fp,
            roi=self.config.roi,
            language=self.config.language,
            provider=getattr(self.config, "provider", "paddle"),
            model_version=None,
            scanner_params=self._scanner_params(),
        ):
            return None
        self._saved_last_processed_ms = cp.last_processed_ms
        return cp

    def _fingerprint(self):
        # Provided by the frame producer via config extension if available.
        fp = getattr(self.config, "fingerprint", None)
        if fp:
            return fp
        return {"path": None, "size": None, "content_hash": None}

    def _scanner_params(self):
        return {
            "coarse_interval_ms": self.config.coarse_interval_ms,
            "boundary_interval_ms": self.config.boundary_interval_ms,
            "similarity_threshold": self.config.similarity_threshold,
            "min_stable_samples": self.config.min_stable_samples,
            "grace_gap_ms": self.config.grace_gap_ms,
        }

    def _save_checkpoint(self, ts_ms, last_error=None, segments_override=None):
        if not self.checkpoint_path:
            return
        completed = (
            list(segments_override)
            if segments_override is not None
            else list(self._builder.snapshot_completed())
        )
        if self._preexisting and segments_override is None:
            completed = dedup_on_resume(
                self._preexisting,
                completed,
                similarity_threshold=self.config.similarity_threshold,
            )
        effective_ts = max(int(ts_ms), int(self._saved_last_processed_ms or 0))
        cp = OcrCheckpoint(
            fingerprint=self._fingerprint(),
            roi=tuple(self.config.roi),
            language=self.config.language,
            provider=getattr(self.config, "provider", "paddle"),
            model_version=None,
            device=self._stats.get("device", "cpu"),
            scanner_params=self._scanner_params(),
            last_processed_ms=effective_ts,
            segments=[self._seg_to_dict(s) for s in completed],
            open_candidate=None,
            stats=dict(self._stats),
            last_error=last_error,
        )
        save_checkpoint(cp, self.checkpoint_path)

    @staticmethod
    def _dict_to_seg(d):
        try:
            return OcrSegment(
                start_ms=int(d.get("start_ms", 0)),
                end_ms=int(d.get("end_ms", 0)),
                text=d.get("text", ""),
                confidence=float(d.get("confidence", 0.0) or 0.0),
                samples=int(d.get("samples", 0)),
            )
        except Exception:
            return OcrSegment(text="")

    @staticmethod
    def _seg_to_dict(seg):
        try:
            return {"start_ms": seg["start_ms"], "end_ms": seg["end_ms"], "text": seg["text"], "confidence": seg["confidence"], "samples": seg["samples"]}
        except Exception:
            return {"start_ms": 0, "end_ms": 0, "text": "", "confidence": 0.0, "samples": 0}

    def finalize(self, duration_ms=0, *, persist=False) -> List:
        dur = int(duration_ms or self._duration_ms or 0)
        if (
            not self._stopped
            and dur > 0
            and self._builder._state == "ACTIVE"
            and self._builder._gap_start_ms is None
            and self._builder._last_active_ts is not None
        ):
            last_active_ts = int(self._builder._last_active_ts)
            from_ts = int(self._last_coarse_ts) if self._last_coarse_ts is not None else last_active_ts
            if from_ts < last_active_ts:
                from_ts = last_active_ts
            gap_to_end = dur - last_active_ts
            if 0 < gap_to_end <= self.config.coarse_interval_ms + 100 and from_ts < dur:
                if self._frame_fetcher is not None:
                    end_probe = min(dur, max(from_ts + 1, dur - 80))
                    end_frame = self._frame_fetcher(end_probe)
                    if end_frame is None and dur - 150 > from_ts:
                        end_probe = dur - 150
                        end_frame = self._frame_fetcher(end_probe)
                    if end_frame is not None:
                        end_crop = end_frame if self._frames_are_cropped else crop_roi(end_frame, self.config.roi)
                        end_sig = _roi_signature(end_crop)
                        end_res, end_norm, end_conf = self._ocr_raw(end_probe, end_crop, end_sig)

                        prev_norm = self._last_norm
                        changed = not _same_state(prev_norm, end_norm, self.config.similarity_threshold)
                        eff_res, eff_norm, eff_conf, eff_sig = end_res, end_norm, end_conf, end_sig
                        before_probes: List[Tuple[int, OcrResult, str, float, Optional[_RoiSignature]]] = []
                        after_probes: List[Tuple[int, OcrResult, str, float, Optional[_RoiSignature]]] = []
                        step_limit = min(125, max(1, int(self.config.boundary_interval_ms)))
                        if changed and (end_probe - from_ts) > step_limit:
                            probes_ext, eff_res, eff_norm, eff_conf, eff_sig = self._refine_transition_ext(
                                from_ts,
                                end_probe,
                                prev_norm,
                                to_sig=end_sig,
                                to_res=end_res,
                                to_norm=end_norm,
                                to_conf=end_conf,
                                duration_ms=dur,
                            )
                            for p_ts, p_res, p_norm, p_conf, p_sig in probes_ext:
                                if p_ts < end_probe:
                                    before_probes.append((p_ts, p_res, p_norm, p_conf, p_sig))
                                elif p_ts > end_probe:
                                    after_probes.append((p_ts, p_res, p_norm, p_conf, p_sig))

                        last_ts = from_ts
                        last_norm = prev_norm
                        last_conf = self._last_conf
                        last_sig = self._last_sig
                        for p_ts, p_res, p_norm, p_conf, p_sig in before_probes:
                            if p_ts > last_ts:
                                self._builder.observe(p_ts, p_res)
                                last_ts = p_ts
                                last_norm = p_norm
                                last_conf = p_conf
                                last_sig = p_sig

                        if end_probe > last_ts:
                            self._builder.observe(end_probe, eff_res)
                            last_ts = end_probe
                            last_norm = eff_norm
                            last_conf = eff_conf
                            last_sig = eff_sig

                        for p_ts, p_res, p_norm, p_conf, p_sig in after_probes:
                            if p_ts > last_ts:
                                self._builder.observe(p_ts, p_res)
                                last_ts = p_ts
                                last_norm = p_norm
                                last_conf = p_conf
                                last_sig = p_sig

                        self._last_norm = last_norm
                        self._last_conf = last_conf
                        self._last_ocr_ts = last_ts
                        self._last_coarse_ts = last_ts
                        self._last_sig = last_sig

                        if (
                            last_norm != ""
                            and self._builder._state == "ACTIVE"
                            and self._builder._gap_start_ms is None
                            and self._builder._open is not None
                        ):
                            self._builder._last_active_ts = max(int(self._builder._last_active_ts or 0), dur)
                            self._builder._open["last_ts"] = max(int(self._builder._open["last_ts"]), dur)
                else:
                    if self._builder._open is not None:
                        self._builder._last_active_ts = max(int(self._builder._last_active_ts or 0), dur)
                        self._builder._open["last_ts"] = max(int(self._builder._open["last_ts"]), dur)

        segments = self._builder.finalize()
        # De-duplicate the resumed-overlap boundary against preexisting segments.
        if self._preexisting:
            segments = dedup_on_resume(
                self._preexisting,
                segments,
                similarity_threshold=self.config.similarity_threshold,
            )
        if self.checkpoint_path and persist:
            last_ts = self._last_coarse_ts if self._last_coarse_ts is not None else (self._last_ocr_ts or 0)
            self._save_checkpoint(last_ts, last_error=None, segments_override=segments)
        self._report(self._last_coarse_ts if self._last_coarse_ts is not None else 0, dur)
        return segments


def _normalize_for_track(text: str) -> str:
    from videotrans.ocr._text import normalize_text

    return normalize_text(text)


