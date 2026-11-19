"""FFmpeg and PyAV-based streaming ROI & fast random-access frame decoders for OCR.

Provides:
- ``probe_video_meta``: single-probe cached metadata (width, height, fps, duration_ms)
- ``sampled_roi_iter``: single-process streaming ffmpeg pipe with ``format=bgr24``
  before ``crop`` and pre-allocated ROI buffer (with PyAV fallback and
  ``proc_registry`` cancellation tracking)
- ``PyAVFastReader`` / ``frame_at_fast``: lightweight random-access decoder for
  binary-search boundary refinement without per-frame ``_probe_size`` calls.
"""
from collections import OrderedDict
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Optional

import numpy as np

_META_CACHE: OrderedDict = OrderedDict()
_META_CACHE_LOCK = threading.Lock()
_META_CACHE_MAXSIZE = 32


# Generic subtitle-agnostic timestamp helper.
def _ts_arg(ts_ms):
    ms = max(0, int(ts_ms))
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    u = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{u:03d}"


def _cmd(ffmpeg):
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


def _meta_cache_key(video: str) -> Optional[tuple]:
    try:
        p = Path(video)
        st = p.stat()
        return (str(p.resolve()), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None


def _probe_via_ffmpeg_stderr(ffmpeg_exe: str, video: str) -> dict:
    probe = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", str(video)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_cmd(ffmpeg_exe),
    )
    text = probe.stderr.decode("utf-8", errors="replace")
    width, height = 0, 0
    m_size = re.search(r"(\d{2,5})x(\d{2,5})", text)
    if m_size:
        width, height = int(m_size.group(1)), int(m_size.group(2))
    fps = 30.0
    m_fps = re.search(r"(\d+(?:\.\d+)?)\s*(?:fps|tbr)\b", text)
    if m_fps:
        try:
            fps = float(m_fps.group(1)) or 30.0
        except ValueError:
            fps = 30.0
    duration_ms = 0
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m_dur:
        h, mm, ss = int(m_dur.group(1)), int(m_dur.group(2)), float(m_dur.group(3))
        duration_ms = int((h * 3600 + mm * 60 + ss) * 1000)
    return {"width": width, "height": height, "fps": fps, "duration_ms": duration_ms}


def probe_video_meta(video: str, ffmpeg_exe: str = "ffmpeg") -> dict:
    """Probe (width, height, fps, duration_ms) once per video with bounded LRU cache."""
    key = _meta_cache_key(str(video))
    if key is not None:
        with _META_CACHE_LOCK:
            cached = _META_CACHE.get(key)
            if cached is not None:
                _META_CACHE.move_to_end(key)
                return dict(cached)

    meta = {"width": 0, "height": 0, "fps": 30.0, "duration_ms": 0}
    try:
        from videotrans.util._ffprobe import get_video_info

        info = get_video_info(str(video))
        if isinstance(info, dict):
            meta["width"] = int(info.get("width") or 0)
            meta["height"] = int(info.get("height") or 0)
            meta["fps"] = float(info.get("video_fps") or 30.0)
            meta["duration_ms"] = int(info.get("time") or 0)
    except Exception:
        pass

    if meta["width"] <= 0 or meta["height"] <= 0 or meta["duration_ms"] <= 0:
        try:
            fb = _probe_via_ffmpeg_stderr(ffmpeg_exe, str(video))
            if meta["width"] <= 0:
                meta["width"] = fb["width"]
            if meta["height"] <= 0:
                meta["height"] = fb["height"]
            if meta["fps"] <= 0:
                meta["fps"] = fb["fps"]
            if meta["duration_ms"] <= 0:
                meta["duration_ms"] = fb["duration_ms"]
        except Exception:
            pass

    if key is not None and meta["width"] > 0 and meta["height"] > 0:
        with _META_CACHE_LOCK:
            _META_CACHE[key] = dict(meta)
            _META_CACHE.move_to_end(key)
            while len(_META_CACHE) > _META_CACHE_MAXSIZE:
                _META_CACHE.popitem(last=False)
    return meta


def compute_roi_geometry(
    src_w: int,
    src_h: int,
    roi: tuple,
    max_roi_width: int = 1280,
) -> tuple[int, int, int, int, int, int]:
    """Return integer (cx, cy, cw, ch, out_w, out_h) for a normalized ROI."""
    x, y, rw, rh = roi
    cx = max(0, min(src_w - 1, int(round(float(x) * src_w))))
    cy = max(0, min(src_h - 1, int(round(float(y) * src_h))))
    cx1 = min(src_w, int(round((float(x) + float(rw)) * src_w)))
    cy1 = min(src_h, int(round((float(y) + float(rh)) * src_h)))
    cw = max(1, cx1 - cx)
    ch = max(1, cy1 - cy)
    if max_roi_width and cw > int(max_roi_width):
        out_w = int(max_roi_width)
        out_h = max(1, int(round(ch * (out_w / float(cw)))))
    else:
        out_w, out_h = cw, ch
    return cx, cy, cw, ch, out_w, out_h


def _resize_bgr(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    if arr.shape[1] == out_w and arr.shape[0] == out_h:
        return arr
    try:
        import cv2
        return cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        h, w = arr.shape[:2]
        ys = np.clip((np.arange(out_h) * h / out_h).astype(np.int64), 0, h - 1)
        xs = np.clip((np.arange(out_w) * w / out_w).astype(np.int64), 0, w - 1)
        return arr[np.ix_(ys, xs)]


def frame_at(ffmpeg_exe: str, video: str, ts_ms: int, *, end_ms: int | None = None):
    """Decode one BGR frame at ts_ms using ffmpeg.

    Returns (frame, width, height) as a numpy (H, W, 3) uint8 BGR array, or
    (None, w, h) when no frame could be decoded.
    """
    ts = _ts_arg(ts_ms)
    cmd = [ffmpeg_exe, "-hide_banner", "-nostdin", "-loglevel", "error"]
    if end_ms is not None:
        cmd.append("-copyts")
    cmd += ["-ss", ts, "-i", video]
    if end_ms is not None:
        # A seek just before the boundary may decode the following frame.
        cmd += ["-vf", f"select='gte(t,{ts_ms / 1000:.3f})*lt(t,{end_ms / 1000:.3f})'"]
    cmd += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_cmd(ffmpeg_exe),
    )
    if proc.returncode != 0 or len(proc.stdout) == 0:
        return None, 0, 0

    meta = probe_video_meta(video, ffmpeg_exe)
    width, height = int(meta["width"]), int(meta["height"])
    if width <= 0 or height <= 0:
        return None, 0, 0
    expected = width * height * 3
    buf = proc.stdout
    if len(buf) < expected:
        return None, width, height
    frame = np.frombuffer(buf[:expected], dtype=np.uint8).reshape(height, width, 3)
    return np.array(frame, copy=True), width, height


def _probe_size(ffmpeg_exe: str, video: str):
    meta = probe_video_meta(video, ffmpeg_exe)
    return int(meta["width"]), int(meta["height"])


def sampled_frame_iter(ffmpeg_exe: str, video: str, *, coarse_interval_ms: int = 500, duration_ms: int = 0):
    """Yield (ts_ms, frame) at coarse intervals by random-access ffmpeg seeks."""
    if duration_ms <= 0:
        duration_ms = probe_duration(ffmpeg_exe, video)
    ts = 0
    while ts < duration_ms:
        frame, w, h = frame_at(ffmpeg_exe, video, ts)
        if frame is None:
            ts += coarse_interval_ms
            continue
        yield int(ts), frame
        ts += coarse_interval_ms


def _pyav_sampled_roi_iter(
    video: str,
    *,
    roi: tuple,
    coarse_interval_ms: int,
    start_ms: int,
    end_ms: Optional[int],
    max_roi_width: int,
    stop_event=None,
):
    import av

    container = av.open(str(video))
    try:
        if not container.streams.video:
            return
        stream = container.streams.video[0]
        try:
            stream.thread_type = "SLICE"
            stream.thread_count = 2
        except Exception:
            pass
        src_w = int(stream.width or 0)
        src_h = int(stream.height or 0)
        if src_w <= 0 or src_h <= 0:
            return
        cx, cy, cw, ch, out_w, out_h = compute_roi_geometry(src_w, src_h, roi, max_roi_width)
        cx1, cy1 = cx + cw, cy + ch
        if start_ms > 0 and stream.time_base is not None:
            target_pts = max(0, int((start_ms / 1000.0) / float(stream.time_base)))
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)

        next_ts = int(start_ms)
        step = max(1, int(coarse_interval_ms))
        for packet in container.demux(stream):
            if stop_event is not None and stop_event():
                break
            for frame in packet.decode():
                if stop_event is not None and stop_event():
                    return
                if frame.pts is None or stream.time_base is None:
                    continue
                frame_ms = int(round(float(frame.pts * stream.time_base) * 1000.0))
                if frame_ms < next_ts - 15:
                    continue
                if end_ms is not None and frame_ms >= end_ms:
                    return
                arr = frame.to_ndarray(format="bgr24")
                crop = arr[cy:cy1, cx:cx1]
                crop = _resize_bgr(crop, out_w, out_h)
                actual_ts = max(int(start_ms), int(frame_ms))
                yield actual_ts, np.ascontiguousarray(crop)
                next_ts += step
                if next_ts <= frame_ms:
                    next_ts += ((frame_ms - next_ts) // step + 1) * step
                if end_ms is not None and next_ts >= end_ms:
                    return
    finally:
        container.close()


def sampled_roi_iter(
    ffmpeg_exe: str,
    video: str,
    *,
    roi: tuple = (0.0, 0.85, 1.0, 0.15),
    coarse_interval_ms: int = 500,
    start_ms: int = 0,
    end_ms: Optional[int] = None,
    duration_ms: int = 0,
    max_roi_width: int = 1280,
    stop_event=None,
    job_id: Optional[str] = None,
    meta: Optional[dict] = None,
):
    """Stream cropped BGR ROI frames via a single ffmpeg child process (PyAV fallback).

    Places ``format=bgr24`` BEFORE ``crop`` so odd ROI heights/widths are never
    silently truncated by YUV420p chroma alignment, and uses ``fps=...:round=up``
    so each yielded ``ts_ms`` receives a frame at or just before ``ts_ms``
    (never up to ``+249ms`` ahead as ``round=near`` can). Uses a single
    pre-allocated reusable ``bytearray`` buffer and registers the child process
    with ``proc_registry`` when a job ID is active.

    Important buffer-aliasing contract: the yielded ``ndarray`` views a single
    reusable ``bytearray`` that is overwritten on the next iteration. Callers
    MUST consume each yielded frame synchronously or copy it (``frame.copy()``)
    if retaining a reference across iterations.
    """
    meta = meta or probe_video_meta(str(video), ffmpeg_exe)
    src_w = int(meta.get("width") or 0)
    src_h = int(meta.get("height") or 0)
    if src_w <= 0 or src_h <= 0:
        try:
            yield from _pyav_sampled_roi_iter(
                str(video),
                roi=roi,
                coarse_interval_ms=coarse_interval_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                max_roi_width=max_roi_width,
                stop_event=stop_event,
            )
        except Exception:
            return
        return

    cx, cy, cw, ch, out_w, out_h = compute_roi_geometry(src_w, src_h, roi, max_roi_width)
    step = max(1, int(coarse_interval_ms))
    vf = f"fps=fps=1000/{step}:start_time=0:round=up,format=bgr24,crop={cw}:{ch}:{cx}:{cy}"
    if (out_w, out_h) != (cw, ch):
        vf += f",scale={out_w}:{out_h}:flags=bilinear"

    cmd = [ffmpeg_exe, "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start_ms > 0:
        cmd += ["-ss", _ts_arg(start_ms)]
    cmd += ["-i", str(video)]
    if end_ms is not None and end_ms > start_ms:
        cmd += ["-t", f"{(end_ms - start_ms) / 1000.0:.3f}"]
    elif duration_ms > 0 and start_ms == 0:
        pass
    cmd += ["-an", "-sn", "-dn", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

    frame_bytes = out_w * out_h * 3
    buf = bytearray(frame_bytes)
    mv = memoryview(buf)
    frame_view = np.frombuffer(buf, dtype=np.uint8).reshape((out_h, out_w, 3))

    active_job_id = job_id
    if active_job_id is None:
        try:
            from videotrans.core.proc_registry import get_current_job_id
            active_job_id = get_current_job_id()
        except Exception:
            active_job_id = None

    try:
        from videotrans.core.proc_registry import kill_process_tree, register_proc, unregister_proc
    except Exception:
        register_proc = unregister_proc = kill_process_tree = None  # type: ignore[assignment]

    proc = None
    yielded = 0
    failed_early = False
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=max(frame_bytes, 65536),
            creationflags=_cmd(ffmpeg_exe),
        )
        if register_proc is not None and active_job_id:
            register_proc(active_job_id, proc)

        idx = 0
        while True:
            if stop_event is not None and stop_event():
                break
            ts_ms = int(start_ms) + idx * step
            if end_ms is not None and ts_ms >= end_ms:
                break
            offset = 0
            assert proc.stdout is not None
            while offset < frame_bytes:
                n = proc.stdout.readinto(mv[offset:])
                if not n:
                    break
                offset += n
            if offset < frame_bytes:
                break
            if stop_event is not None and stop_event():
                break
            yielded += 1
            idx += 1
            yield ts_ms, frame_view

        if yielded == 0 and (stop_event is None or not stop_event()):
            rc = proc.poll()
            if rc is None:
                try:
                    rc = proc.wait(timeout=2)
                except Exception:
                    rc = None
            if rc not in (0, None):
                failed_early = True
    except OSError:
        failed_early = True
    finally:
        if proc is not None:
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            if proc.poll() is None:
                if kill_process_tree is not None:
                    try:
                        kill_process_tree(proc)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                else:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
            if unregister_proc is not None and active_job_id:
                unregister_proc(active_job_id, proc)

    if failed_early and yielded == 0 and (stop_event is None or not stop_event()):
        try:
            yield from _pyav_sampled_roi_iter(
                str(video),
                roi=roi,
                coarse_interval_ms=coarse_interval_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                max_roi_width=max_roi_width,
                stop_event=stop_event,
            )
        except Exception:
            return


class PyAVFastReader:
    """Reusable random-access PyAV reader for boundary-refinement ROI probes."""

    def __init__(
        self,
        video: str,
        roi: tuple = (0.0, 0.85, 1.0, 0.15),
        max_roi_width: int = 1280,
    ) -> None:
        import av

        self._av = av
        self._video = str(video)
        self._roi = tuple(roi)
        self._max_roi_width = int(max_roi_width)
        self._container = av.open(self._video)
        if not self._container.streams.video:
            self._container.close()
            raise RuntimeError(f"No video stream in {self._video}")
        self._stream = self._container.streams.video[0]
        try:
            self._stream.thread_type = "SLICE"
            self._stream.thread_count = 2
        except Exception:
            pass
        src_w = int(self._stream.width or 0)
        src_h = int(self._stream.height or 0)
        self._geom = compute_roi_geometry(src_w, src_h, self._roi, self._max_roi_width)

    def frame_at(self, ts_ms: int) -> Optional[np.ndarray]:
        if self._container is None:
            return None
        cx, cy, cw, ch, out_w, out_h = self._geom
        cx1, cy1 = cx + cw, cy + ch
        tb = self._stream.time_base
        target_pts = max(0, int(round((max(0, int(ts_ms)) / 1000.0) / float(tb)))) if tb else 0
        try:
            self._container.seek(target_pts, stream=self._stream, backward=True, any_frame=False)
            last_arr = None
            for packet in self._container.demux(self._stream):
                for frame in packet.decode():
                    if frame.pts is not None and tb is not None:
                        frame_ms = int(round(float(frame.pts * tb) * 1000.0))
                    else:
                        frame_ms = int(ts_ms)
                    last_arr = frame
                    if frame_ms >= int(ts_ms) - 15:
                        arr = frame.to_ndarray(format="bgr24")
                        crop = arr[cy:cy1, cx:cx1]
                        crop = _resize_bgr(crop, out_w, out_h)
                        return np.ascontiguousarray(crop)
            if last_arr is not None:
                arr = last_arr.to_ndarray(format="bgr24")
                crop = arr[cy:cy1, cx:cx1]
                crop = _resize_bgr(crop, out_w, out_h)
                return np.ascontiguousarray(crop)
        except Exception:
            return None
        return None

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def frame_at_fast(
    video: str,
    ts_ms: int,
    *,
    roi: Optional[tuple] = None,
    max_roi_width: int = 1280,
    ffmpeg_exe: str = "ffmpeg",
    meta: Optional[dict] = None,
    reader: Optional[PyAVFastReader] = None,
) -> Optional[np.ndarray]:
    """Decode a single cropped ROI frame at ``ts_ms`` without ``_probe_size``."""
    target_roi = tuple(roi) if roi is not None else (0.0, 0.0, 1.0, 1.0)
    if reader is not None:
        res = reader.frame_at(ts_ms)
        if res is not None:
            return res
    else:
        try:
            with PyAVFastReader(str(video), target_roi, max_roi_width=max_roi_width) as tmp_reader:
                res = tmp_reader.frame_at(ts_ms)
                if res is not None:
                    return res
        except Exception:
            pass

    meta = meta or probe_video_meta(str(video), ffmpeg_exe)
    src_w = int(meta.get("width") or 0)
    src_h = int(meta.get("height") or 0)
    if src_w <= 0 or src_h <= 0:
        return None
    cx, cy, cw, ch, out_w, out_h = compute_roi_geometry(src_w, src_h, target_roi, max_roi_width)
    vf = f"format=bgr24,crop={cw}:{ch}:{cx}:{cy}"
    if (out_w, out_h) != (cw, ch):
        vf += f",scale={out_w}:{out_h}:flags=bilinear"
    cmd = [
        ffmpeg_exe, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-ss", _ts_arg(ts_ms), "-i", str(video),
        "-frames:v", "1", "-an", "-sn", "-dn", "-vf", vf,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_cmd(ffmpeg_exe),
    )
    expected = out_w * out_h * 3
    if proc.returncode != 0 or len(proc.stdout) < expected:
        return None
    arr = np.frombuffer(proc.stdout[:expected], dtype=np.uint8).reshape(out_h, out_w, 3)
    return np.array(arr, copy=True)


def probe_duration(ffmpeg_exe: str, video: str) -> int:
    meta = probe_video_meta(str(video), ffmpeg_exe)
    return int(meta.get("duration_ms") or 0)
