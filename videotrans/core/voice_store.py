# -*- coding: utf-8 -*-
"""Voice model, storage, CWE-22 path security, and legacy migration."""
from __future__ import annotations

import json
import logging
from pathlib import Path, PureWindowsPath
import shutil
import time
from typing import Any, Optional
import uuid

from videotrans.configure.config import ROOT_DIR
from videotrans.core.db import db_conn

logger = logging.getLogger("videotrans.voice_store")

VOICES_DIR = Path(ROOT_DIR) / "data" / "voices"
PREVIEWS_DIR = Path(ROOT_DIR) / "tmp" / "voice_previews"


def init_voice_dirs() -> tuple[Path, Path]:
    """Ensure storage directories for voice reference audio and preview clips exist."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    return VOICES_DIR, PREVIEWS_DIR


def get_voice_audio_path(filename_or_path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve and validate a reference audio path strictly within VOICES_DIR (CWE-22 defense).
    Raises ValueError on directory traversal attempts.
    """
    if not filename_or_path:
        raise ValueError("Audio filename or path cannot be empty")

    target_base = (base_dir or VOICES_DIR).resolve()
    raw_path = str(filename_or_path)
    candidate = Path(raw_path.replace("\\", "/"))
    if PureWindowsPath(raw_path).drive and not candidate.is_absolute():
        raise ValueError(f"Path traversal detected: {filename_or_path} is outside {target_base}")
    if not candidate.is_absolute():
        candidate = target_base / candidate
    resolved = candidate.resolve()

    try:
        resolved.relative_to(target_base)
    except ValueError:
        raise ValueError(f"Path traversal detected: {filename_or_path} is outside {target_base}")

    return resolved


def get_preview_audio_path(filename_or_path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve and validate a preview audio path strictly within PREVIEWS_DIR (CWE-22 defense).
    Raises ValueError on directory traversal attempts.
    """
    if not filename_or_path:
        raise ValueError("Preview audio filename or path cannot be empty")

    target_base = (base_dir or PREVIEWS_DIR).resolve()
    raw_path = str(filename_or_path)
    candidate = Path(raw_path.replace("\\", "/"))
    if PureWindowsPath(raw_path).drive and not candidate.is_absolute():
        raise ValueError(f"Path traversal detected: {filename_or_path} is outside {target_base}")
    if not candidate.is_absolute():
        candidate = target_base / candidate
    resolved = candidate.resolve()

    try:
        resolved.relative_to(target_base)
    except ValueError:
        raise ValueError(f"Path traversal detected: {filename_or_path} is outside {target_base}")

    return resolved


def _row_to_dict(row: Any) -> dict[str, Any]:
    d = dict(row)
    raw_tuning = d.get("tuning_params")
    if isinstance(raw_tuning, str):
        try:
            d["tuning_params"] = json.loads(raw_tuning) if raw_tuning else {}
        except Exception:
            d["tuning_params"] = {}
    elif raw_tuning is None:
        d["tuning_params"] = {}
    d["tuning"] = d["tuning_params"]
    d["is_active"] = bool(d.get("is_active", 1))
    return d


def create_voice(
    name: str,
    provider: int,
    *,
    voice_id: Optional[str] = None,
    description: str = "",
    kind: str = "clone",
    language: str = "Auto",
    ref_audio_path: str = "",
    ref_text: str = "",
    instruct: str = "",
    external_voice_id: str = "",
    tuning_params: Optional[dict[str, Any]] = None,
    preview_audio_path: str = "",
    is_active: bool = True,
    db_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Create a new custom voice record in the database."""
    clean_name = " ".join(str(name).split())
    if not clean_name:
        raise ValueError("Voice name cannot be empty")

    vid = voice_id or f"voice_{uuid.uuid4().hex[:8]}"
    now = time.time()
    tuning_json = json.dumps(tuning_params or {})

    with db_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO voices (
                id, name, description, provider, kind, language,
                ref_audio_path, ref_text, instruct, external_voice_id,
                tuning_params, preview_audio_path, is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                vid,
                clean_name,
                description,
                int(provider),
                kind,
                language,
                ref_audio_path,
                ref_text,
                instruct,
                external_voice_id,
                tuning_json,
                preview_audio_path,
                1 if is_active else 0,
                now,
                now,
            ),
        )

    voice = get_voice(vid, db_path=db_path)
    if voice is None:
        raise RuntimeError(f"Failed to retrieve newly created voice: {vid}")
    return voice


def get_voice(voice_id: str, db_path: Optional[str | Path] = None) -> Optional[dict[str, Any]]:
    """Retrieve a single voice by ID."""
    with db_conn(db_path) as conn:
        cursor = conn.execute("SELECT * FROM voices WHERE id = ?", (voice_id,))
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None


def find_voice_by_name(
    name: str,
    provider: Optional[int] = None,
    active_only: bool = True,
    db_path: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    """Look up a voice by case-insensitive name, optionally filtered by provider."""
    query = "SELECT * FROM voices WHERE LOWER(name) = LOWER(?)"
    params: list[Any] = [" ".join(str(name).split())]
    if provider is not None:
        query += " AND provider = ?"
        params.append(int(provider))
    if active_only:
        query += " AND is_active = 1"
    query += " ORDER BY created_at DESC LIMIT 1"

    with db_conn(db_path) as conn:
        cursor = conn.execute(query, params)
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None


def list_voices(
    provider: Optional[int] = None,
    active_only: bool = True,
    db_path: Optional[str | Path] = None,
) -> list[dict[str, Any]]:
    """List voices, optionally filtered by provider and active status."""
    query = "SELECT * FROM voices WHERE 1=1"
    params: list[Any] = []
    if provider is not None:
        query += " AND provider = ?"
        params.append(int(provider))
    if active_only:
        query += " AND is_active = 1"
    query += " ORDER BY created_at DESC"

    with db_conn(db_path) as conn:
        cursor = conn.execute(query, params)
        return [_row_to_dict(row) for row in cursor.fetchall()]


def update_voice(
    voice_id: str,
    db_path: Optional[str | Path] = None,
    **fields: Any,
) -> Optional[dict[str, Any]]:
    """Update fields of an existing voice record."""
    allowed = {
        "name", "description", "provider", "kind", "language",
        "ref_audio_path", "ref_text", "instruct", "external_voice_id",
        "tuning_params", "preview_audio_path", "is_active",
    }
    updates: dict[str, Any] = {}
    for k, v in fields.items():
        if k in allowed:
            if k == "tuning_params" and isinstance(v, dict):
                updates[k] = json.dumps(v)
            elif k == "is_active":
                updates[k] = 1 if v else 0
            elif k == "name":
                clean = " ".join(str(v).split())
                if not clean:
                    raise ValueError("Voice name cannot be empty")
                updates[k] = clean
            else:
                updates[k] = v
        elif k == "tuning" and isinstance(v, dict):
            updates["tuning_params"] = json.dumps(v)

    if not updates:
        return get_voice(voice_id, db_path=db_path)

    # Only bump updated_at if fields other than preview_audio_path are changed
    if set(updates.keys()) != {"preview_audio_path"}:
        updates["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    vals = list(updates.values()) + [voice_id]

    with db_conn(db_path) as conn:
        conn.execute(f"UPDATE voices SET {set_clause} WHERE id = ?", vals)

    return get_voice(voice_id, db_path=db_path)


def delete_voice(
    voice_id: str,
    hard: bool = False,
    db_path: Optional[str | Path] = None,
) -> bool:
    """Delete a voice (soft-delete by default, hard-delete if requested)."""
    voice = get_voice(voice_id, db_path=db_path)
    if not voice:
        return False

    with db_conn(db_path) as conn:
        if hard:
            conn.execute("DELETE FROM voices WHERE id = ?", (voice_id,))
            if voice.get("ref_audio_path"):
                try:
                    fpath = get_voice_audio_path(voice["ref_audio_path"])
                    if fpath.is_file():
                        fpath.unlink(missing_ok=True)
                except Exception:
                    pass
            if voice.get("preview_audio_path"):
                try:
                    ppath = get_preview_audio_path(voice["preview_audio_path"])
                    if ppath.is_file():
                        ppath.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            conn.execute(
                "UPDATE voices SET is_active = 0, updated_at = ? WHERE id = ?",
                (time.time(), voice_id),
            )
    return True


def bulk_delete_voices(
    voice_ids: list[str],
    hard: bool = False,
    db_path: Optional[str | Path] = None,
) -> list[str]:
    """Delete multiple voices. Returns list of successfully deleted voice IDs."""
    deleted: list[str] = []
    for vid in voice_ids:
        if delete_voice(str(vid), hard=hard, db_path=db_path):
            deleted.append(str(vid))
    return deleted


def migrate_legacy_voices(db_path: Optional[str | Path] = None) -> int:
    """Migrate legacy params['vieneu_roles'] and params['f5tts_role'] to voices table idempotently."""
    from videotrans.configure.config import params
    from videotrans import tts

    init_voice_dirs()
    migrated_count = 0

    # 1. VieNeu roles (dict: {name: audio_path})
    saved_vieneu = params.get("vieneu_roles", {})
    if isinstance(saved_vieneu, dict):
        for name, audio_path in saved_vieneu.items():
            name_str = " ".join(str(name).split())
            if not name_str or not audio_path:
                continue
            src_file = Path(audio_path).expanduser().resolve()
            if not src_file.is_file():
                continue
            existing = find_voice_by_name(name_str, provider=tts.VIENEU_TTS, active_only=False, db_path=db_path)
            if existing:
                continue
            v_id = f"voice_{uuid.uuid4().hex[:8]}"
            dest_filename = f"{v_id}.wav"
            dest_file = VOICES_DIR / dest_filename
            try:
                shutil.copy2(src_file, dest_file)
            except Exception:
                continue
            create_voice(
                name=name_str,
                provider=tts.VIENEU_TTS,
                voice_id=v_id,
                ref_audio_path=dest_filename,
                language="vi",
                kind="clone",
                db_path=db_path,
            )
            migrated_count += 1

    # 2. OmniVoice / f5tts roles (multiline string: "path#ref_text")
    saved_f5 = params.get("f5tts_role", "")
    if isinstance(saved_f5, str) and saved_f5.strip():
        for line in saved_f5.strip().split("\n"):
            parts = line.strip().split("#")
            if len(parts) >= 2:
                audio_ref = parts[0].strip()
                ref_text = parts[1].strip()
                src_file = Path(audio_ref).expanduser().resolve()
                if not src_file.is_file():
                    continue
                v_name = src_file.stem
                existing = find_voice_by_name(v_name, provider=tts.OMNIVOICE_TTS, active_only=False, db_path=db_path)
                if existing:
                    continue
                v_id = f"voice_{uuid.uuid4().hex[:8]}"
                dest_filename = f"{v_id}.wav"
                dest_file = VOICES_DIR / dest_filename
                try:
                    shutil.copy2(src_file, dest_file)
                except Exception:
                    continue
                create_voice(
                    name=v_name,
                    provider=tts.OMNIVOICE_TTS,
                    voice_id=v_id,
                    ref_audio_path=dest_filename,
                    ref_text=ref_text,
                    language="Auto",
                    kind="clone",
                    db_path=db_path,
                )
                migrated_count += 1

    return migrated_count


def cache_voice_embedding(
    voice_id: str,
    speaker_emb: Any,
    ref_codes: Optional[Any] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Save pre-extracted speaker embedding and optional reference codes as .npz."""
    import numpy as np

    target_dir = base_dir or VOICES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"{voice_id}_embedding.npz"

    save_dict = {}
    if speaker_emb is not None:
        if hasattr(speaker_emb, "detach"):
            speaker_emb = speaker_emb.detach().cpu().numpy()
        save_dict["speaker_emb"] = np.asarray(speaker_emb, dtype=np.float32)
    if ref_codes is not None:
        if hasattr(ref_codes, "detach"):
            ref_codes = ref_codes.detach().cpu().numpy()
        save_dict["ref_codes"] = np.asarray(ref_codes, dtype=np.int64)

    np.savez_compressed(out_file, **save_dict)
    return out_file


def get_voice_embedding(
    voice_id: str,
    base_dir: Optional[Path] = None,
) -> Optional[tuple[Any, Optional[Any]]]:
    """Retrieve pre-extracted (speaker_emb, ref_codes) if cached on disk."""
    import numpy as np

    target_dir = base_dir or VOICES_DIR
    npz_file = target_dir / f"{voice_id}_embedding.npz"
    if not npz_file.is_file():
        return None

    try:
        with np.load(npz_file) as data:
            speaker_emb = data["speaker_emb"] if "speaker_emb" in data else None
            ref_codes = data["ref_codes"] if "ref_codes" in data else None
            return (speaker_emb, ref_codes)
    except Exception as exc:
        logger.warning("Failed to load cached embedding for %s: %s", voice_id, exc)
        return None


def resolve_voice_params(
    voice_id_or_name: str,
    provider: Optional[int] = None,
    db_path: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    """
    Resolve a custom voice record, flattening designed voice hierarchies
    to determine the effective base audio, preset voice, and tuning parameters.
    """
    if not voice_id_or_name:
        return None

    voice = get_voice(voice_id_or_name, db_path=db_path)
    if not voice:
        voice = find_voice_by_name(voice_id_or_name, provider=provider, db_path=db_path)
    if not voice:
        return None

    result = dict(voice)
    tuning = dict(voice.get("tuning_params") or {})

    # If it's a designed voice, resolve its base voice
    if voice.get("kind") == "design":
        base_ref = voice.get("external_voice_id") or tuning.get("base_voice", "")
        if base_ref:
            # Check if base_ref points to another custom voice
            parent_voice = get_voice(base_ref, db_path=db_path) or find_voice_by_name(base_ref, provider=provider, db_path=db_path)
            if parent_voice:
                result["ref_audio_path"] = parent_voice.get("ref_audio_path", "")
                result["effective_voice_id"] = parent_voice["id"]
                result["base_type"] = "custom"
            else:
                # Base is a preset voice name (e.g. "Bình (nam miền Bắc)")
                result["preset_voice"] = base_ref
                result["base_type"] = "preset"
        else:
            result["base_type"] = "preset"

    result["tuning_params"] = tuning
    return result
