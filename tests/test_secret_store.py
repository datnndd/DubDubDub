# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import pytest

from videotrans.core.db import db_conn, init_db, set_db_path
from videotrans.core import secret_store
from videotrans.configure._env_loader import parse_env_file, load_env
from videotrans.configure._app_params import AppParams


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Set up an isolated test database and clear cached Fernet key."""
    test_db = tmp_path / "test_secrets.db"
    set_db_path(test_db)
    init_db(test_db)
    secret_store.invalidate()
    yield
    secret_store.invalidate()


def test_derive_fernet_key_produces_valid_key():
    key = secret_store.derive_fernet_key()
    assert isinstance(key, bytes)
    assert len(key) == 44  # Base64 encoded 32 bytes


def test_secret_crud_and_at_rest_encryption():
    secret_store.set_secret("deepgram_apikey", "test-dg-secret-key-12345")
    assert secret_store.has_secret("deepgram_apikey") is True
    assert secret_store.get_secret("deepgram_apikey") == "test-dg-secret-key-12345"

    # Verify at-rest encryption: raw SQLite value must not contain the plaintext
    with db_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = 'secret.deepgram_apikey'").fetchone()
    assert row is not None
    raw_val = row[0]
    assert "test-dg-secret-key-12345" not in raw_val
    assert raw_val.startswith("gAAAAA")  # Fernet token prefix

    # Test delete
    secret_store.delete_secret("deepgram_apikey")
    assert secret_store.has_secret("deepgram_apikey") is False
    assert secret_store.get_secret("deepgram_apikey") is None


def test_secret_migration_isolation_corrupt_or_different_machine(monkeypatch):
    secret_store.set_secret("chatgpt_key", "sk-proj-original-key")
    assert secret_store.get_secret("chatgpt_key") == "sk-proj-original-key"

    # Simulate moving DB to a machine with different machine ID
    secret_store.invalidate()
    monkeypatch.setattr(secret_store, "_read_machine_id", lambda: b"different-machine-guid-999")

    # Should gracefully return None without raising InvalidToken
    migrated_val = secret_store.get_secret("chatgpt_key")
    assert migrated_val is None


def test_app_params_secret_precedence_and_redaction(tmp_path, monkeypatch):
    json_path = tmp_path / "test_params.json"
    params = AppParams(_json_path=str(json_path))

    # 1. Stored in secret store
    secret_store.set_secret("deepgram_apikey", "db-key-value")
    assert params.get("deepgram_apikey") == "db-key-value"
    assert params["deepgram_apikey"] == "db-key-value"

    # 2. Environment variable overrides secret store
    monkeypatch.setenv("DEEPGRAM_API_KEY", "env-override-key")
    assert params.get("deepgram_apikey") == "env-override-key"
    assert params["deepgram_apikey"] == "env-override-key"

    monkeypatch.delenv("DEEPGRAM_API_KEY")
    assert params.get("deepgram_apikey") == "db-key-value"

    # 3. Disk redaction
    params._save_to_disk()
    disk_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert disk_data.get("deepgram_apikey") == ""

    # 4. In-memory privacy and attribute resolution
    assert params.to_dict()["deepgram_apikey"] == ""
    assert params.__dict__["deepgram_apikey"] == ""
    assert params.deepgram_apikey == "db-key-value"



def test_env_loader_parser(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
# Comments should be ignored
OPENAI_API_KEY="sk-test-quoted-value"
DEEPGRAM_API_KEY='dg-single-quoted'
export DEEPSEEK_KEY=deepseek-raw-value # trailing comment
EMPTY_KEY=
    SPACED_KEY = spaced-value
""",
        encoding="utf-8",
    )
    parsed = parse_env_file(env_file)
    assert parsed["OPENAI_API_KEY"] == "sk-test-quoted-value"
    assert parsed["DEEPGRAM_API_KEY"] == "dg-single-quoted"
    assert parsed["DEEPSEEK_KEY"] == "deepseek-raw-value"
    assert parsed["SPACED_KEY"] == "spaced-value"
    assert parsed["EMPTY_KEY"] == ""


def test_migrate_plaintext_secrets(tmp_path, monkeypatch):
    # Setup legacy params.json and cfg.json
    v_dir = tmp_path / "videotrans"
    v_dir.mkdir(parents=True)
    m_dir = tmp_path / "models"
    m_dir.mkdir(parents=True)

    params_file = v_dir / "params.json"
    params_file.write_text(
        json.dumps({"deepgram_apikey": "legacy-dg-key", "output_dir": "/legacy/out"}),
        encoding="utf-8",
    )

    cfg_file = v_dir / "cfg.json"
    cfg_file.write_text(
        json.dumps({"hf_token": "legacy-hf-token", "crf": 23}),
        encoding="utf-8",
    )

    hf_txt = m_dir / "hf_token.txt"
    hf_txt.write_text("hf_txt_value", encoding="utf-8")

    monkeypatch.setattr(secret_store, "ROOT_DIR", str(tmp_path))

    stats = secret_store.migrate_plaintext_secrets()
    assert stats["migrated"] >= 2

    # Verify secrets are now in encrypted store
    assert secret_store.get_secret("deepgram_apikey") == "legacy-dg-key"
    assert secret_store.get_secret("hf_token") == "legacy-hf-token"

    # Verify files on disk have secrets redacted
    p_data = json.loads(params_file.read_text(encoding="utf-8"))
    assert p_data["deepgram_apikey"] == ""
    assert p_data["output_dir"] == "/legacy/out"

    c_data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert c_data["hf_token"] == ""
    assert c_data["crf"] == 23


def test_set_secret_persists_to_env_and_sync(tmp_path, monkeypatch):
    test_env = tmp_path / "custom.env"
    monkeypatch.setenv("VIDEOTRANS_ENV_FILE", str(test_env))

    # 1. Setting secret writes to custom.env
    secret_store.set_secret("chatgpt_key", "sk-custom-env-test-key")
    assert secret_store.get_secret("chatgpt_key") == "sk-custom-env-test-key"
    assert test_env.is_file()
    content = test_env.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-custom-env-test-key" in content

    # 2. Deleting secret removes from custom.env
    secret_store.delete_secret("chatgpt_key")
    assert secret_store.get_secret("chatgpt_key") is None
    content_after = test_env.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in content_after

    # 3. sync_env_and_db_secrets bidirectional test
    test_env.write_text("DEEPSEEK_API_KEY=ds-synced-from-env\n", encoding="utf-8")
    secret_store.sync_env_and_db_secrets()
    assert secret_store.get_secret("deepseek_key") == "ds-synced-from-env"
    assert secret_store.resolve_secret("deepseek_key") == "ds-synced-from-env"

