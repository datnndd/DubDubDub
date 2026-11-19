# -*- coding: utf-8 -*-
"""SQLite WAL persistence engine for projects, jobs, and job events."""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from videotrans.configure.config import ROOT_DIR

logger = logging.getLogger("videotrans.db")

DEFAULT_DB_PATH = Path(os.environ.get("DB_PATH", Path(ROOT_DIR) / "projects.db"))
_current_db_path: Path = DEFAULT_DB_PATH
_DB_INIT_LOCK = threading.RLock()
_WAL_INITIALIZED: set[str] = set()

_BASE_SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id TEXT,
    media_id TEXT,
    media_path TEXT,
    duration REAL DEFAULT 0,
    stage INTEGER DEFAULT 1,
    status TEXT DEFAULT 'pending',
    audio_hash TEXT DEFAULT '',
    state_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_projects_updated ON projects(updated_at);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_projects_audio_hash ON projects(audio_hash);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    project_id TEXT,
    status TEXT NOT NULL,
    stage TEXT,
    progress REAL DEFAULT 0,
    message TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    finished_at REAL,
    error TEXT,
    meta_json TEXT DEFAULT '{}',
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_job_events_job_seq ON job_events(job_id, seq);

CREATE TABLE IF NOT EXISTS voices (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    provider INTEGER NOT NULL,
    kind TEXT DEFAULT 'clone',
    language TEXT DEFAULT 'Auto',
    ref_audio_path TEXT DEFAULT '',
    ref_text TEXT DEFAULT '',
    instruct TEXT DEFAULT '',
    external_voice_id TEXT DEFAULT '',
    tuning_params TEXT DEFAULT '{}',
    preview_audio_path TEXT DEFAULT '',
    is_active INTEGER DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_voices_provider ON voices(provider);
CREATE INDEX IF NOT EXISTS idx_voices_active ON voices(is_active);
CREATE INDEX IF NOT EXISTS idx_voices_created ON voices(created_at);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_settings_updated ON settings(updated_at);
"""


def set_db_path(path: str | Path) -> None:
    """Set the database path (useful for tests or custom installations)."""
    global _current_db_path
    _current_db_path = Path(path)


def get_db_path() -> Path:
    """Return the currently configured database path."""
    return _current_db_path


def _ensure_wal_mode(conn: sqlite3.Connection, key: str) -> None:
    with _DB_INIT_LOCK:
        if key in _WAL_INITIALIZED:
            return
        row = conn.execute("PRAGMA journal_mode").fetchone()
        cur_mode = str(row[0]).lower() if row and row[0] is not None else ""
        if cur_mode == "wal":
            _WAL_INITIALIZED.add(key)
            return
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                res = conn.execute("PRAGMA journal_mode=WAL").fetchone()
                if res and str(res[0]).lower() == "wal":
                    _WAL_INITIALIZED.add(key)
                    return
            except sqlite3.OperationalError as exc:
                last_err = exc
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    raise
                time.sleep(0.02 * (attempt + 1))
        if last_err is not None:
            raise last_err


def get_db(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Open a SQLite connection configured with busy timeout, foreign keys, NORMAL sync, and WAL mode."""
    target_path = Path(db_path) if db_path is not None else _current_db_path
    is_memory = str(target_path) == ":memory:"
    if not is_memory:
        target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    if not is_memory:
        key = str(target_path.resolve())
        if key not in _WAL_INITIALIZED:
            _ensure_wal_mode(conn, key)
    return conn


@contextmanager
def db_conn(db_path: Optional[str | Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context-managed SQLite connection committing on clean exit and rolling back on error."""
    conn = get_db(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[str | Path] = None) -> None:
    """Initialize database tables and indexes idempotently."""
    target_path = Path(db_path) if db_path is not None else _current_db_path
    with _DB_INIT_LOCK:
        if str(target_path) != ":memory:" and not target_path.exists():
            _WAL_INITIALIZED.discard(str(target_path.resolve()))
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                with db_conn(db_path) as conn:
                    conn.executescript(_BASE_SCHEMA)
                    # Check and migrate existing projects table if parent_id is missing
                    cols = [r["name"] for r in conn.execute("PRAGMA table_info(projects)").fetchall()]
                    if "parent_id" not in cols:
                        conn.execute("ALTER TABLE projects ADD COLUMN parent_id TEXT")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_parent ON projects(parent_id)")
                break
            except sqlite3.OperationalError as exc:
                last_err = exc
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    raise
                time.sleep(0.02 * (attempt + 1))
        else:
            if last_err is not None:
                raise last_err
    logger.debug("Database initialized at %s", db_path or _current_db_path)
