# -*- coding: utf-8 -*-
"""Machine-bound encrypted secret store for API keys and sensitive tokens.

Features:
  - Symmetric authenticated encryption via Fernet (AES-128-CBC + HMAC-SHA256).
  - Machine-bound key derivation function (KDF) using Scrypt and OS Machine ID.
  - Per-install 16-byte random salt stored in SQLite `settings` table.
  - At-rest security: DB theft does not decrypt secrets on another machine.
  - Machine migration isolation: InvalidToken logs warning and returns None gracefully.
"""
from __future__ import annotations

import base64
import getpass
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from videotrans.configure._paths import ROOT_DIR
from videotrans.core.db import db_conn

logger = logging.getLogger("videotrans.secret_store")

_KEY_CACHE: bytes | None = None
_CACHE_LOCK = threading.Lock()

_SALT_KEY = "_secret_key_salt"
_SALT_BYTES = 16

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_KEYLEN = 32

SECRET_KEYS = {
    "deepgram_apikey",
    "chatgpt_key",
    "deepseek_key",
    "gemini_key",
    "elevenlabstts_key",
    "hf_token",
}

ENV_KEY_MAP: dict[str, list[str]] = {
    "deepgram_apikey": ["DEEPGRAM_API_KEY", "DEEPGRAM_APIKEY"],
    "chatgpt_key": ["OPENAI_API_KEY", "CHATGPT_KEY"],
    "deepseek_key": ["DEEPSEEK_API_KEY", "DEEPSEEK_KEY"],
    "gemini_key": ["GEMINI_API_KEY", "GEMINI_KEY"],
    "elevenlabstts_key": ["ELEVENLABS_API_KEY", "ELEVENLABSTTS_KEY", "ELEVEN_API_KEY"],
    "hf_token": ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"],
}

PRIMARY_ENV_KEY_MAP: dict[str, str] = {
    "deepgram_apikey": "DEEPGRAM_API_KEY",
    "chatgpt_key": "OPENAI_API_KEY",
    "deepseek_key": "DEEPSEEK_API_KEY",
    "gemini_key": "GEMINI_API_KEY",
    "elevenlabstts_key": "ELEVENLABS_API_KEY",
    "hf_token": "HF_TOKEN",
}


def _read_machine_id() -> bytes:
    """Best-effort cross-platform machine identifier."""
    plat = sys.platform
    try:
        if plat.startswith("win"):
            import winreg  # type: ignore[import-not-found]
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Cryptography",
                )
                try:
                    val, _ = winreg.QueryValueEx(key, "MachineGuid")
                    if val:
                        return str(val).strip().encode("utf-8")
                finally:
                    winreg.CloseKey(key)
            except OSError:
                pass
        elif plat == "darwin":
            out = subprocess.check_output(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode("utf-8", errors="replace")
            for line in out.splitlines():
                if "IOPlatformUUID" in line:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        uuid_val = parts[1].strip().strip('"').strip()
                        if uuid_val:
                            return uuid_val.encode("utf-8")
        elif plat.startswith("linux"):
            for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
                if os.path.isfile(p):
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        mid = f.read().strip()
                    if mid:
                        return mid.encode("utf-8")
    except Exception as exc:
        logger.debug("Primary machine-id lookup failed: %s", exc)

    logger.warning(
        "Could not read OS machine-id; falling back to hostname+user. "
        "Moving database to another environment will require re-entering API keys."
    )
    fallback = f"{socket.gethostname()}::{getpass.getuser()}"
    return fallback.encode("utf-8")


def _load_or_create_salt() -> bytes:
    """Read the persisted salt row or generate and save one."""
    with db_conn() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (_SALT_KEY,)
        ).fetchone()
        if row is not None and row[0]:
            try:
                return base64.b64decode(row[0])
            except (ValueError, TypeError):
                logger.warning("Persisted secret key salt is corrupt; regenerating.")
        salt = os.urandom(_SALT_BYTES)
        conn.execute(
            "INSERT OR REPLACE INTO settings(key, value, updated_at) VALUES (?, ?, ?)",
            (_SALT_KEY, base64.b64encode(salt).decode("ascii"), time.time()),
        )
    return salt


def derive_fernet_key() -> bytes:
    """Derive 32-byte base64-encoded Fernet key via Scrypt."""
    global _KEY_CACHE
    with _CACHE_LOCK:
        if _KEY_CACHE is not None:
            return _KEY_CACHE

        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

        salt = _load_or_create_salt()
        machine_id = _read_machine_id()
        kdf = Scrypt(
            salt=salt,
            length=_SCRYPT_KEYLEN,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
        )
        raw = kdf.derive(machine_id)
        _KEY_CACHE = base64.urlsafe_b64encode(raw)
        return _KEY_CACHE


def invalidate() -> None:
    """Drop the cached key to force re-derivation in tests or environment changes."""
    global _KEY_CACHE
    with _CACHE_LOCK:
        _KEY_CACHE = None


def _fernet():
    from cryptography.fernet import Fernet
    return Fernet(derive_fernet_key())


def _db_key(key: str) -> str:
    key = key.strip()
    return key if key.startswith("secret.") else f"secret.{key}"


def _clean_key(db_key_str: str) -> str:
    return db_key_str[7:] if db_key_str.startswith("secret.") else db_key_str


def get_secret(key: str) -> Optional[str]:
    """Retrieve and decrypt secret by key name, returning None if not found or invalid."""
    if not key:
        return None
    db_k = _db_key(key)
    try:
        with db_conn() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (db_k,)).fetchone()
        if row is None or not row[0]:
            return None
        blob = row[0]
        try:
            from cryptography.fernet import InvalidToken
        except ImportError:
            logger.error("cryptography unavailable; cannot decrypt secret")
            return None
        try:
            return _fernet().decrypt(blob.encode("ascii")).decode("utf-8")
        except InvalidToken:
            logger.warning(
                "Stored secret '%s' failed to decrypt (machine-id or salt changed).", key
            )
            return None
    except Exception as exc:
        logger.warning("Secret store read failed for '%s': %s", key, exc)
        return None


def set_secret(key: str, value: Optional[str], persist_to_env: bool = True) -> None:
    """Encrypt and store secret in SQLite, and persist to private .env file."""
    if not key:
        return
    if not value or not str(value).strip():
        delete_secret(key, persist_to_env=persist_to_env)
        return
    val_str = str(value).strip()
    cipher = _fernet()
    blob = cipher.encrypt(val_str.encode("utf-8")).decode("ascii")
    db_k = _db_key(key)
    with db_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings(key, value, updated_at) VALUES (?, ?, ?)",
            (db_k, blob, time.time()),
        )

    clean_k = _clean_key(key)
    env_var_name = PRIMARY_ENV_KEY_MAP.get(clean_k)
    if persist_to_env and env_var_name:
        try:
            from videotrans.configure._env_loader import save_env_var
            save_env_var(env_var_name, val_str)
        except Exception as exc:
            logger.warning("Failed to persist %s to .env: %s", env_var_name, exc)
    elif env_var_name:
        os.environ[env_var_name] = val_str


def delete_secret(key: str, persist_to_env: bool = True) -> None:
    """Delete encrypted secret from database and .env."""
    if not key:
        return
    db_k = _db_key(key)
    try:
        with db_conn() as conn:
            conn.execute("DELETE FROM settings WHERE key = ?", (db_k,))
    except Exception as exc:
        logger.warning("Failed to delete secret '%s': %s", key, exc)

    clean_k = _clean_key(key)
    env_var_name = PRIMARY_ENV_KEY_MAP.get(clean_k)
    if persist_to_env and env_var_name:
        try:
            from videotrans.configure._env_loader import delete_env_var
            delete_env_var(env_var_name)
        except Exception as exc:
            logger.warning("Failed to delete %s from .env: %s", env_var_name, exc)
    elif env_var_name:
        os.environ.pop(env_var_name, None)


def has_secret(key: str) -> bool:
    """Return whether a valid, non-empty secret exists in store."""
    sec = get_secret(key)
    return bool(sec and sec.strip())


def list_configured_secrets() -> set[str]:
    """Return set of secret keys currently configured in database."""
    results: set[str] = set()
    try:
        with db_conn() as conn:
            rows = conn.execute("SELECT key FROM settings WHERE key LIKE 'secret.%'").fetchall()
        for r in rows:
            k = _clean_key(r[0])
            if has_secret(k):
                results.add(k)
    except Exception as exc:
        logger.warning("Failed to list configured secrets: %s", exc)
    return results


def resolve_secret(key: str, fallback: Optional[str] = None) -> Optional[str]:
    """Resolve secret with 4-tier precedence: os.environ -> secret_store -> fallback."""
    # 1. Environment variables
    aliases = ENV_KEY_MAP.get(key, [])
    for env_name in aliases:
        v = os.environ.get(env_name)
        if v and v.strip():
            return v.strip()
    direct = os.environ.get(key) or os.environ.get(key.upper())
    if direct and direct.strip():
        return direct.strip()

    # 2. Encrypted SQLite secret store
    db_val = get_secret(key)
    if db_val and db_val.strip():
        return db_val.strip()

    # 3. Fallback
    if fallback and fallback.strip():
        return fallback.strip()
    return None


def is_from_env(key: str) -> bool:
    """Return True if secret is supplied via system environment variable."""
    aliases = ENV_KEY_MAP.get(key, [])
    for env_name in aliases:
        if os.environ.get(env_name, "").strip():
            return True
    if os.environ.get(key, "").strip() or os.environ.get(key.upper(), "").strip():
        return True
    return False


def is_secret_configured(key: str) -> bool:
    """Return True if secret is configured either in environment or database."""
    val = resolve_secret(key)
    return bool(val and val.strip())


def migrate_plaintext_secrets() -> dict[str, int]:
    """Migrate legacy plaintext secrets from params.json, cfg.json, and hf_token.txt to secret_store.

    Redacts plaintext secrets on disk after migration.
    """
    stats = {"migrated": 0, "redacted": 0}

    # 1. params.json
    params_path = Path(ROOT_DIR) / "videotrans" / "params.json"
    if params_path.is_file():
        try:
            data = json.loads(params_path.read_text(encoding="utf-8"))
            modified = False
            for sk in SECRET_KEYS:
                raw_val = str(data.get(sk) or "").strip()
                if raw_val:
                    if not has_secret(sk):
                        set_secret(sk, raw_val)
                        stats["migrated"] += 1
                    data[sk] = ""
                    modified = True
                    stats["redacted"] += 1
            if modified:
                params_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Error migrating params.json secrets: %s", exc)

    # 2. cfg.json
    cfg_path = Path(ROOT_DIR) / "videotrans" / "cfg.json"
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            hf_val = str(data.get("hf_token") or "").strip()
            if hf_val:
                if not has_secret("hf_token"):
                    set_secret("hf_token", hf_val)
                    stats["migrated"] += 1
                data["hf_token"] = ""
                stats["redacted"] += 1
                cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Error migrating cfg.json secrets: %s", exc)

    # 3. hf_token.txt
    token_txt = Path(ROOT_DIR) / "models" / "hf_token.txt"
    if token_txt.is_file():
        try:
            raw_token = token_txt.read_text(encoding="utf-8").strip()
            if raw_token:
                if not has_secret("hf_token"):
                    set_secret("hf_token", raw_token)
                    stats["migrated"] += 1
                token_txt.write_text("", encoding="utf-8")
                stats["redacted"] += 1
        except Exception as exc:
            logger.warning("Error migrating hf_token.txt: %s", exc)

    return stats


def sync_env_and_db_secrets() -> None:
    """Synchronize secrets between private .env file and encrypted SQLite store.

    1. Loads .env file into os.environ.
    2. If a secret exists in .env, ensure it is also saved to DB (if missing).
    3. If a secret exists in DB but is missing from .env, sync it into .env so restarts never lose it.
    """
    try:
        from videotrans.configure._env_loader import load_env, save_env_var
        load_env(override=True)
        for secret_k, env_k in PRIMARY_ENV_KEY_MAP.items():
            env_val = os.environ.get(env_k, "").strip()
            db_val = get_secret(secret_k)
            if env_val:
                if not db_val or db_val != env_val:
                    set_secret(secret_k, env_val, persist_to_env=False)
            elif db_val and db_val.strip():
                save_env_var(env_k, db_val.strip())
    except Exception as exc:
        logger.warning("Failed to sync secrets between .env and DB: %s", exc)
