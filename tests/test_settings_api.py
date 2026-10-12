# -*- coding: utf-8 -*-
import json
import os
import tempfile
import unittest
from pathlib import Path
import httpx
from fastapi import FastAPI

from videotrans.core.db import init_db, set_db_path
from videotrans.core import secret_store, storage_config
from videotrans.api.routes import settings as settings_routes


class TestSettingsAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.tmp_dir.name) / "test_api_settings.db"
        set_db_path(db_path)
        init_db(db_path)
        secret_store.invalidate()

        self._orig_env = {}
        for aliases in secret_store.ENV_KEY_MAP.values():
            for env_name in aliases:
                if env_name in os.environ:
                    self._orig_env[env_name] = os.environ.pop(env_name)

        self.app = FastAPI()
        settings_routes.register_routes(self.app)
        self.transport = httpx.ASGITransport(app=self.app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        if self.tmp_dir:
            self.tmp_dir.cleanup()
        secret_store.invalidate()
        from videotrans.core.db import DEFAULT_DB_PATH
        set_db_path(DEFAULT_DB_PATH)
        for k, v in getattr(self, "_orig_env", {}).items():
            os.environ[k] = v

    async def test_get_settings_snapshot_returns_saved_api_keys(self):
        secret_store.set_secret("deepgram_apikey", "super-secret-key-123")
        resp = await self.client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()

        assert "providers" in data
        assert "storage" in data
        assert "general" in data

        dg = data["providers"]["deepgram"]
        assert dg["configured"] is True
        # Configured API key must be returned to client for persistence and masked display
        assert dg["apiKey"] == "super-secret-key-123"

    async def test_update_provider_settings_write_only(self):
        payload = {
            "apiKey": "sk-test-secret-value-xyz",
            "model": "nova-3",
        }
        resp = await self.client.put("/api/settings/providers/asr/deepgram", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["configured"] is True
        assert "sk-test-secret-value-xyz" not in json.dumps(data)

        # Confirm stored securely
        assert secret_store.get_secret("deepgram_apikey") == "sk-test-secret-value-xyz"

    async def test_probe_connection_success_and_failure(self):
        # 1. Missing key
        resp = await self.client.post("/api/settings/providers/asr/deepgram/test", json={})
        assert resp.status_code == 400

        # 2. Mocked successful probe
        async def mock_success(provider_id, api_key, base_url="", proxy=""):
            return True, "Mock connection succeeded"

        self.app.state.provider_probe = mock_success
        resp = await self.client.post(
            "/api/settings/providers/asr/deepgram/test",
            json={"apiKey": "valid-key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "Mock connection succeeded" in data["message"]

        # 3. Mocked failed probe
        async def mock_fail(provider_id, api_key, base_url="", proxy=""):
            return False, "Authentication failed: 401 Unauthorized"

        self.app.state.provider_probe = mock_fail
        resp = await self.client.post(
            "/api/settings/providers/asr/deepgram/test",
            json={"apiKey": "invalid-key"},
        )
        assert resp.status_code == 400
        text = resp.text
        assert "401" in text

    async def test_storage_settings_get_and_update(self):
        import os

        resp = await self.client.get("/api/settings/storage")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "usage" in data

        # Test path update
        with tempfile.TemporaryDirectory() as custom_dir:
            out_dir = str(Path(custom_dir) / "custom_output")
            up_resp = await self.client.put(
                "/api/settings/storage",
                json={"output_dir": out_dir},
            )
            assert up_resp.status_code == 200
            up_data = up_resp.json()
            assert up_data["ok"] is True

        # Test rejection of root path
        bad_resp = await self.client.put(
            "/api/settings/storage",
            json={"output_dir": "C:\\" if os.name == "nt" else "/"},
        )
        assert bad_resp.status_code == 400

    async def test_clean_temp_cache(self):
        resp = await self.client.post("/api/settings/storage/clean-temp")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "cleaned_bytes" in data

    async def test_update_general_settings(self):
        payload = {
            "proxy": "",
            "defaultSourceLanguage": "ja",
            "defaultTargetLanguage": "en",
            "crf": 21,
            "preset": "fast",
        }
        resp = await self.client.put("/api/settings/general", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

    async def test_options_reflect_secret_and_model_updates(self):
        # 1. Initially without key, deepgram is not configured
        opt_resp1 = await self.client.get("/api/options")
        assert opt_resp1.status_code == 200
        opts1 = opt_resp1.json()
        dg_opt1 = next(p for p in opts1["asrProviders"] if p["id"] == "deepgram")
        qwen_opt = next(p for p in opts1["asrProviders"] if p["id"] == "qwen-asr")
        google_trans = next(p for p in opts1["translationProviders"] if p["id"] == "google")
        openai_trans = next(p for p in opts1["translationProviders"] if p["id"] == "openai")

        assert dg_opt1["configured"] is False
        assert qwen_opt["configured"] is True  # local model requires no key
        assert google_trans["configured"] is True  # free google translate requires no key
        assert openai_trans["configured"] is False

        # 2. Save Deepgram key & custom model
        save_dg = await self.client.put(
            "/api/settings/providers/asr/deepgram",
            json={"apiKey": "dg-test-key-999", "model": "nova-2"},
        )
        assert save_dg.status_code == 200

        # 3. Save OpenAI key & custom model & custom base url
        save_oa = await self.client.put(
            "/api/settings/providers/llm/openai",
            json={"apiKey": "sk-openai-custom-key", "model": "gpt-4o-mini", "baseUrl": "https://api.myproxy.com/v1"},
        )
        assert save_oa.status_code == 200

        # 4. Save General defaults
        save_gen = await self.client.put(
            "/api/settings/general",
            json={"defaultSourceLanguage": "ko", "defaultTargetLanguage": "vi"},
        )
        assert save_gen.status_code == 200

        # 5. Fetch /api/options again - must reflect updated values
        opt_resp2 = await self.client.get("/api/options")
        assert opt_resp2.status_code == 200
        opts2 = opt_resp2.json()

        dg_opt2 = next(p for p in opts2["asrProviders"] if p["id"] == "deepgram")
        assert dg_opt2["configured"] is True
        assert "nova-2" in dg_opt2["models"]

        openai_trans2 = next(p for p in opts2["translationProviders"] if p["id"] == "openai")
        assert openai_trans2["configured"] is True
        assert openai_trans2["model"] == "gpt-4o-mini"
        assert openai_trans2["baseUrl"] == "https://api.myproxy.com/v1"

        # 6. Fetch /api/settings - verify snapshot has new models and configured flags
        settings_resp = await self.client.get("/api/settings")
        assert settings_resp.status_code == 200
        sdata = settings_resp.json()
        assert sdata["providers"]["deepgram"]["configured"] is True
        assert sdata["providers"]["deepgram"]["model"] == "nova-2"
        assert sdata["providers"]["openai"]["configured"] is True
        assert sdata["providers"]["openai"]["model"] == "gpt-4o-mini"
        assert sdata["providers"]["openai"]["baseUrl"] == "https://api.myproxy.com/v1"
        assert sdata["general"]["defaultSourceLanguage"] == "ko"
        assert sdata["general"]["defaultTargetLanguage"] == "vi"

    async def test_fetch_provider_models_endpoint_and_test_connection_returns_models(self):
        # Mock provider_model_fetcher
        async def mock_fetch_models(provider_id, api_key, base_url="", proxy=""):
            if api_key == "bad-key":
                return False, [], "Invalid API key"
            return True, ["model-alpha", "model-beta", "model-gamma"], "Found 3 models"

        self.app.state.provider_model_fetcher = mock_fetch_models

        # 1. Error if no key configured and none passed
        err_resp = await self.client.post("/api/settings/providers/llm/openai/models", json={})
        assert err_resp.status_code == 400
        assert "API key is not configured" in err_resp.text

        # 2. Success with passed key
        succ_resp = await self.client.post(
            "/api/settings/providers/llm/openai/models",
            json={"apiKey": "sk-test-live-key"},
        )
        assert succ_resp.status_code == 200
        s_data = succ_resp.json()
        assert s_data["ok"] is True
        assert s_data["models"] == ["model-alpha", "model-beta", "model-gamma"]

        # 3. Verify options and settings reflect the fetched models
        opt_resp = await self.client.get("/api/options")
        assert opt_resp.status_code == 200
        opt_data = opt_resp.json()
        oa_opt = next(p for p in opt_data["translationProviders"] if p["id"] == "openai")
        assert "model-alpha" in oa_opt["models"]
        assert "model-beta" in oa_opt["models"]

        # 4. Connection test also returns models
        async def mock_probe(provider_id, api_key, base_url="", proxy=""):
            return True, "Connected successfully"

        self.app.state.provider_probe = mock_probe
        test_resp = await self.client.post(
            "/api/settings/providers/llm/openai/test",
            json={"apiKey": "sk-test-live-key"},
        )
        assert test_resp.status_code == 200
        t_data = test_resp.json()
        assert t_data["ok"] is True
        assert t_data["models"] == ["model-alpha", "model-beta", "model-gamma"]


