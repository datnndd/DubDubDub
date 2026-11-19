# -*- coding: utf-8 -*-
"""Bounded LRU/TTL directory cache eviction helpers for TTS previews and dubbing cache."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Optional, Iterable

logger = logging.getLogger("videotrans.tts.cache")

_ORPHAN_SIDECAR_GRACE_SECONDS = 60.0


def _is_temp_cache_name(name: str) -> bool:
    lower = name.lower()
    return name.startswith(".") or ".tmp" in lower or lower.endswith(".tmp")


def _unlink_with_retry(path: Path, retries: int = 4, delay: float = 0.005) -> None:
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt + 1 >= retries:
                return
            time.sleep(delay * (attempt + 1))
        except OSError:
            return


def touch_cache_entry(path: Path, paired_suffixes: tuple[str, ...] = ()) -> None:
    """Update mtime on a cache entry (and optional paired sidecars) for LRU tracking."""
    try:
        os.utime(path, None)
    except OSError:
        pass
    for suffix in paired_suffixes:
        try:
            sidecar = path.with_suffix(suffix)
            if sidecar.exists():
                os.utime(sidecar, None)
        except OSError:
            pass


def prune_cache_dir(
    cache_dir: Path | str,
    *,
    max_age_seconds: float,
    max_bytes: int,
    max_files: Optional[int] = None,
    primary_suffixes: tuple[str, ...] = (".wav",),
    paired_suffixes: tuple[str, ...] = (),
    exclude_names: Optional[Iterable[str]] = None,
    preserve_prefixes: tuple[str, ...] = (),
    preserve_suffixes: tuple[str, ...] = (),
) -> int:
    """Evict expired (TTL) and oldest-accessed (LRU by mtime) entries in ``cache_dir``.

    Returns the number of primary entries removed. Safe against concurrent deletions.
    """
    root = Path(cache_dir)
    if not root.is_dir():
        return 0

    excluded = set(exclude_names or ())
    now = time.time()
    entries: list[tuple[float, int, Path]] = []
    primary_stems: set[str] = set()
    all_files: list[Path] = []

    try:
        with os.scandir(root) as it:
            for entry in it:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    p = Path(entry.path)
                    name = entry.name
                    if _is_temp_cache_name(name):
                        continue
                    all_files.append(p)
                    if name in excluded:
                        primary_stems.add(p.stem)
                        continue
                    if preserve_prefixes and any(name.startswith(pref) for pref in preserve_prefixes):
                        if preserve_suffixes and any(name.endswith(suf) for suf in preserve_suffixes):
                            primary_stems.add(p.stem)
                            continue
                    if primary_suffixes and p.suffix.lower() not in primary_suffixes:
                        continue
                    st = entry.stat(follow_symlinks=False)
                    entries.append((float(st.st_mtime), int(st.st_size), p))
                    primary_stems.add(p.stem)
                except OSError:
                    continue
    except OSError:
        return 0

    removed = 0
    kept: list[tuple[float, int, Path]] = []

    def _unlink_with_sidecars(p: Path) -> None:
        _unlink_with_retry(p)
        for suffix in paired_suffixes:
            _unlink_with_retry(p.with_suffix(suffix))

    # 1. TTL eviction
    for mtime, size, p in entries:
        if max_age_seconds > 0 and (now - mtime) > max_age_seconds:
            _unlink_with_sidecars(p)
            removed += 1
        else:
            kept.append((mtime, size, p))

    # 2. Size / count LRU eviction (oldest mtime first)
    kept.sort(key=lambda item: item[0])
    total_bytes = sum(item[1] for item in kept)
    total_count = len(kept)

    idx = 0
    while idx < len(kept):
        over_bytes = max_bytes > 0 and total_bytes > max_bytes
        over_files = max_files is not None and max_files > 0 and total_count > max_files
        if not over_bytes and not over_files:
            break
        _, size, p = kept[idx]
        _unlink_with_sidecars(p)
        total_bytes -= size
        total_count -= 1
        removed += 1
        idx += 1

    # 3. Clean up orphaned sidecars (e.g., .json without matching primary .wav)
    if paired_suffixes:
        remaining_stems = {p.stem for _, _, p in kept[idx:]} | {
            p.stem for p in all_files if p.name in excluded or (
                preserve_prefixes and any(p.name.startswith(pref) for pref in preserve_prefixes)
                and (not preserve_suffixes or any(p.name.endswith(suf) for suf in preserve_suffixes))
            )
        }
        for p in all_files:
            if _is_temp_cache_name(p.name):
                continue
            if p.suffix.lower() in paired_suffixes and p.stem not in remaining_stems:
                try:
                    st = p.stat()
                    if (now - float(st.st_mtime)) > _ORPHAN_SIDECAR_GRACE_SECONDS:
                        _unlink_with_retry(p)
                except OSError:
                    pass

    return removed
