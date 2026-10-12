# -*- coding: utf-8 -*-
"""Zero-dependency .env loader for early-boot environment variables."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("videotrans.env_loader")

_LINE_REGEX = re.compile(
    r"""
    ^\s*
    (?:export\s+)?
    (?P<key>[A-Za-z_][A-Za-z0-9_]*)
    \s*=\s*
    (?P<val>.*?)
    \s*$
    """,
    re.VERBOSE,
)


def _parse_value(val: str) -> str:
    """Parse and unquote an env value string."""
    val = val.strip()
    if not val:
        return ""
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    # Remove inline comment if present outside quotes
    if " #" in val:
        val = val.split(" #", 1)[0].strip()
    return val


def parse_env_file(path: Path | str) -> dict[str, str]:
    """Parse key-value pairs from a file without modifying os.environ."""
    target = Path(path)
    if not target.is_file():
        return {}
    results: dict[str, str] = {}
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _LINE_REGEX.match(line)
            if m:
                key = m.group("key")
                val = _parse_value(m.group("val"))
                results[key] = val
    except Exception as exc:
        logger.warning("Failed to parse env file %s: %s", target, exc)
    return results


def get_default_env_path() -> Path:
    """Return the primary .env file path in ROOT_DIR or overridden by VIDEOTRANS_ENV_FILE."""
    custom = os.environ.get("VIDEOTRANS_ENV_FILE")
    if custom:
        return Path(custom)
    from videotrans.configure._paths import ROOT_DIR
    return Path(ROOT_DIR) / ".env"


def find_default_env_paths() -> list[Path]:
    """Return candidates for default .env configuration files."""
    candidates = [
        get_default_env_path(),
        Path.home() / ".config" / "dubdub" / "env",
    ]
    return [p for p in candidates if p.is_file()]


def load_env(env_path: Optional[Path | str] = None, override: bool = True) -> dict[str, str]:
    """Load environment variables from file into os.environ.

    If env_path is None, checks default locations (ROOT_DIR/.env, ~/.config/dubdub/env).
    Returns the dictionary of loaded variables.
    """
    loaded: dict[str, str] = {}
    paths_to_try: list[Path] = []
    if env_path is not None:
        p = Path(env_path)
        if p.is_file():
            paths_to_try.append(p)
    else:
        paths_to_try.extend(find_default_env_paths())

    for p in paths_to_try:
        pairs = parse_env_file(p)
        for k, v in pairs.items():
            if override or k not in os.environ:
                os.environ[k] = v
                loaded[k] = v
            elif k not in loaded and k in os.environ:
                loaded[k] = os.environ[k]
    return loaded


def save_env_var(key: str, value: str, env_path: Optional[Path | str] = None) -> None:
    """Save or update an environment variable in the target .env file and update os.environ."""
    clean_k = key.strip()
    clean_v = str(value or "").strip()

    if clean_v:
        os.environ[clean_k] = clean_v
    else:
        os.environ.pop(clean_k, None)

    # Protect developer's real .env when automated tests run without an explicit test env file
    if "PYTEST_CURRENT_TEST" in os.environ and not env_path and not os.environ.get("VIDEOTRANS_ENV_FILE"):
        return

    target = Path(env_path) if env_path else get_default_env_path()

    if not clean_v:
        line_val = ""
    elif any(c in clean_v for c in (' ', '"', "'", '#', '\n', '$')):
        escaped = clean_v.replace('"', '\\"')
        line_val = f'"{escaped}"'
    else:
        line_val = clean_v

    target_line = f"{clean_k}={line_val}"

    lines: list[str] = []
    if target.is_file():
        try:
            raw = target.read_text(encoding="utf-8", errors="replace")
            lines = raw.splitlines()
        except Exception as exc:
            logger.warning("Failed to read existing env file %s: %s", target, exc)

    found = False
    key_pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(clean_k)}\s*=", re.IGNORECASE)
    new_lines: list[str] = []
    for line in lines:
        if key_pattern.match(line):
            new_lines.append(target_line)
            found = True
        else:
            new_lines.append(line)

    if not found:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(target_line)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write env file %s: %s", target, exc)


def delete_env_var(key: str, env_path: Optional[Path | str] = None) -> None:
    """Remove an environment variable from the target .env file and os.environ."""
    clean_k = key.strip()
    os.environ.pop(clean_k, None)

    if "PYTEST_CURRENT_TEST" in os.environ and not env_path and not os.environ.get("VIDEOTRANS_ENV_FILE"):
        return

    target = Path(env_path) if env_path else get_default_env_path()
    if target.is_file():
        try:
            raw = target.read_text(encoding="utf-8", errors="replace")
            lines = raw.splitlines()
            key_pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(clean_k)}\s*=", re.IGNORECASE)
            new_lines = [line for line in lines if not key_pattern.match(line)]
            target.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to update env file on delete %s: %s", target, exc)
