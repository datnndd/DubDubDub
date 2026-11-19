from __future__ import annotations

import os
import threading
import time
import types as pytypes
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
import requests
from aiohttp.test_utils import TestClient, TestServer
from google.genai import errors as genai_errors

from videotrans import translator
from videotrans.api.app import create_app
from videotrans.configure.excepts import StopTask, TranslateSrtError
from videotrans.task._base import BaseTask
from videotrans.task.orchestrator import (
    CancellationToken,
    EventKind,
    TaskRequest,
    TaskResult,
    TaskStatus,
    run_staged_translation,
)
from videotrans.task.taskcfg import SrtItem
from videotrans.translator import _base as base_mod
from videotrans.translator import _gemini as gemini_mod
from videotrans.translator import _google as google_mod
from videotrans.translator import _openaicompat as oai_mod
from videotrans.util import prompts as prompts_mod


def _make_item(idx: int, text: str, start_ms: int = 0, end_ms: int = 1000) -> SrtItem:
    s_sec, s_rem = divmod(start_ms, 1000)
    e_sec, e_rem = divmod(end_ms, 1000)
    s_raw = f"00:00:{s_sec:02d},{s_rem:03d}"
    e_raw = f"00:00:{e_sec:02d},{e_rem:03d}"
    return SrtItem(
        line=idx,
        time=f"{s_raw} --> {e_raw}",
        text=text,
        start_time=start_ms,
        end_time=end_ms,
        startraw=s_raw,
        endraw=e_raw,
    )


class _FakeSettings(dict):
    def get(self, key, default=None):
        return super().get(key, default)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(base_mod, "TEMP_ROOT", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Cache Key Symmetry & Cancellation Safety
# ---------------------------------------------------------------------------

def test_run_srt_cache_hit_100_percent_compact_and_legacy(isolated_cache, monkeypatch):
    for compact_mode in (True, False):
        fake_settings = _FakeSettings({
            "aisendsrt": True,
            "aitrans_thread": 2,
            "aitrans_concurrency": 1,
            "aitrans_compact": compact_mode,
            "translation_wait": 0,
        })
        monkeypatch.setattr(base_mod, "settings", fake_settings)

        calls = []

        class MockSrtTrans(base_mod.BaseTrans):
            def _item_task(self, data):
                calls.append(data)
                if compact_mode:
                    return "[1] Hola\n[2] Mundo"
                return (
                    "1\n00:00:00,000 --> 00:00:01,000\nHola\n\n"
                    "2\n00:00:01,000 --> 00:00:02,000\nMundo"
                )

        items1 = [
            _make_item(1, "Hello", 0, 1000),
            _make_item(2, "World", 1000, 2000),
        ]
        items2 = [
            _make_item(1, "Hello", 0, 1000),
            _make_item(2, "World", 1000, 2000),
        ]

        t1 = MockSrtTrans(
            translate_type=translator.CHATGPT_INDEX,
            text_list=items1,
            source_code="en",
            target_code="es",
            is_test=False,
        )
        out1 = t1.run()
        assert len(calls) == 1
        assert [x["text"] for x in out1] == ["Hola", "Mundo"]

        t2 = MockSrtTrans(
            translate_type=translator.CHATGPT_INDEX,
            text_list=items2,
            source_code="en",
            target_code="es",
            is_test=False,
        )
        out2 = t2.run()
        assert len(calls) == 1  # 100% cache hit on repeat
        assert [x["text"] for x in out2] == ["Hola", "Mundo"]


def test_run_srt_cancel_does_not_crash_or_poison_cache(isolated_cache, monkeypatch):
    fake_settings = _FakeSettings({
        "aisendsrt": True,
        "aitrans_thread": 2,
        "aitrans_concurrency": 1,
        "aitrans_compact": True,
    })
    monkeypatch.setattr(base_mod, "settings", fake_settings)
    token = CancellationToken()

    class CancellingTrans(base_mod.BaseTrans):
        def _item_task(self, data):
            token.cancel()
            return None

    t = CancellingTrans(
        translate_type=translator.CHATGPT_INDEX,
        text_list=[_make_item(1, "Hello")],
        source_code="en",
        target_code="es",
        is_test=False,
        cancellation_token=token,
    )
    res = t.run()
    assert res == []
    cache_files = list((isolated_cache / "translate_cache").glob("*.txt"))
    assert cache_files == []


# ---------------------------------------------------------------------------
# 2. <think> / Fence / <TRANSLATE_TEXT> Stripping & Plain Fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "<think>\nstep 1\nstep 2\n</think>\n<TRANSLATE_TEXT>\n[1] Hola\n</TRANSLATE_TEXT>",
            "[1] Hola",
        ),
        (
            "<THINK>\nunclosed reasoning\n<TRANSLATE_TEXT>\n[1] Hola\n</TRANSLATE_TEXT>",
            "[1] Hola",
        ),
        ("<think>\nunclosed reasoning only", ""),
        ("orphan reasoning</think>\n[1] Hola", "[1] Hola"),
        ("```srt\n[1] Hola\n[2] Mundo\n```", "[1] Hola\n[2] Mundo"),
        (
            "```xml\n<TRANSLATE_TEXT>\n```srt\n[1] Hola\n```\n</TRANSLATE_TEXT>\n```",
            "[1] Hola",
        ),
        ("<TRANSLATE_TEXT>\n[1] Hola", "[1] Hola"),
        ("[1] Hola\n[2] Mundo", "[1] Hola\n[2] Mundo"),
    ],
)
def test_clean_ai_response_variants(raw, expected):
    assert prompts_mod.clean_ai_response(raw, tag="TRANSLATE_TEXT") == expected


# ---------------------------------------------------------------------------
# 3. text_list=None Guard & Bounded Sliding Window Context
# ---------------------------------------------------------------------------

def test_openaicompat_none_text_list_and_sliding_context_window(monkeypatch):
    fake_settings = _FakeSettings({
        "aisendsrt": True,
        "aitrans_thread": 4,
        "aitrans_context": True,
        "aitrans_context_window": 3,
        "aitrans_concurrency": 4,
        "aitrans_compact": True,
        "translation_wait": 0,
    })
    monkeypatch.setattr(base_mod, "settings", fake_settings)
    monkeypatch.setattr(oai_mod, "settings", fake_settings)

    # 1. text_list=None (used by llm_segment) must not crash
    seg_inst = oai_mod.OpenAICampat(ainame="chatgpt", uuid="seg-1")
    assert seg_inst.run() == []

    # 2. Sliding context window with 12 items -> 3 bounded sequential batches of 4
    seen_requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            seen_requests.append(kwargs["messages"])
            user_msg = kwargs["messages"][1]["content"]
            # Extract lines inside <INPUT>...</INPUT>
            input_block = user_msg.split("<INPUT>")[1].split("</INPUT>")[0].strip()
            count = len([ln for ln in input_block.splitlines() if ln.strip()])
            batch_no = len(seen_requests)
            out_lines = [
                f"[{i}] Trans_B{batch_no}_L{i}" for i in range(1, count + 1)
            ]
            content = "<TRANSLATE_TEXT>\n" + "\n".join(out_lines) + "\n</TRANSLATE_TEXT>"
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def close(self):
            pass

    monkeypatch.setattr(oai_mod, "OpenAI", FakeOpenAI)

    items = [_make_item(i, f"Source line {i}", i * 1000, (i + 1) * 1000) for i in range(1, 13)]
    tr_inst = oai_mod.OpenAICampat(
        translate_type=translator.CHATGPT_INDEX,
        ainame="chatgpt",
        text_list=items,
        source_code="en",
        target_code="es",
        target_language_name="Spanish",
        is_test=True,
    )
    tr_inst.api_key = "sk-test"
    tr_inst.api_url = "https://api.openai.com/v1"
    tr_inst.model_name = "gpt-test"

    result = tr_inst.run()
    assert len(result) == 12
    assert len(seen_requests) == 3  # Bounded batches of 4, NOT all 12 in one call

    # Static system prompt identical across all 3 batches & never contains context
    sys_prompts = {msgs[0]["content"] for msgs in seen_requests}
    assert len(sys_prompts) == 1
    assert "<CONTEXT_PREVIOUS_TRANSLATIONS" not in seen_requests[1][0]["content"]

    # Batch 1 has no context; Batch 2 & 3 carry last 3 translated lines in user
    assert "<CONTEXT_PREVIOUS_TRANSLATIONS" not in seen_requests[0][1]["content"]
    ctx_b2 = seen_requests[1][1]["content"].split("<CONTEXT_PREVIOUS_TRANSLATIONS")[1].split("</CONTEXT_PREVIOUS_TRANSLATIONS>")[0]
    assert "Trans_B1_L2" in ctx_b2
    assert "Trans_B1_L3" in ctx_b2
    assert "Trans_B1_L4" in ctx_b2
    assert "Trans_B1_L1" not in ctx_b2  # Window capped at 3

    ctx_b3 = seen_requests[2][1]["content"].split("<CONTEXT_PREVIOUS_TRANSLATIONS")[1].split("</CONTEXT_PREVIOUS_TRANSLATIONS>")[0]
    assert "Trans_B2_L2" in ctx_b3
    assert "Trans_B2_L3" in ctx_b3
    assert "Trans_B2_L4" in ctx_b3


# ---------------------------------------------------------------------------
# 4. Bounded Concurrency Speedup, Strict Ordering, Zero Sleep & Client Pooling
# ---------------------------------------------------------------------------

def test_bounded_concurrency_speedup_ordering_and_client_pooling(monkeypatch):
    fake_settings = _FakeSettings({
        "aisendsrt": True,
        "aitrans_thread": 2,
        "aitrans_context": False,
        "aitrans_concurrency": 3,
        "aitrans_compact": True,
        "translation_wait": 5.0,  # Must be skipped when concurrency > 1
    })
    monkeypatch.setattr(base_mod, "settings", fake_settings)
    monkeypatch.setattr(oai_mod, "settings", fake_settings)

    ctor_calls = 0
    closed_count = 0
    active_lock = threading.Lock()
    active_now = 0
    max_active = 0

    class FakeCompletions:
        def create(self, **kwargs):
            nonlocal active_now, max_active
            with active_lock:
                active_now += 1
                max_active = max(max_active, active_now)
            try:
                user_msg = kwargs["messages"][1]["content"]
                # Make early batches slower than later batches to verify ordering
                if "Source 1" in user_msg:
                    time.sleep(0.18)
                else:
                    time.sleep(0.06)
                input_block = user_msg.split("<INPUT>")[1].split("</INPUT>")[0].strip()
                lines = [ln.strip() for ln in input_block.splitlines() if ln.strip()]
                out = "\n".join(ln.replace("Source", "Target") for ln in lines)
                return SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=f"<TRANSLATE_TEXT>\n{out}\n</TRANSLATE_TEXT>"))]
                )
            finally:
                with active_lock:
                    active_now -= 1

    class FakeOpenAI:
        def __init__(self, **kwargs):
            nonlocal ctor_calls
            ctor_calls += 1
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def close(self):
            nonlocal closed_count
            closed_count += 1

    monkeypatch.setattr(oai_mod, "OpenAI", FakeOpenAI)

    items = [_make_item(i, f"Source {i}", i * 500, (i + 1) * 500) for i in range(1, 13)]
    inst = oai_mod.OpenAICampat(
        translate_type=translator.CHATGPT_INDEX,
        ainame="chatgpt",
        text_list=items,
        source_code="en",
        target_code="es",
        target_language_name="Spanish",
        is_test=True,
    )
    inst.api_key = "sk-pool"
    inst.api_url = "https://api.openai.com/v1"
    inst.model_name = "gpt-test"

    t0 = time.monotonic()
    res = inst.run()
    elapsed = time.monotonic() - t0

    assert [x["text"] for x in res] == [f"Target {i}" for i in range(1, 13)]
    assert 2 <= max_active <= 3
    assert elapsed < 1.0  # Neither sequential sum nor 5s wait_sec sleep occurred
    assert ctor_calls == 1  # Client reused across all 6 batches
    assert closed_count == 1  # Closed in run() finally


# ---------------------------------------------------------------------------
# 5. OpenAICampat Runtime Retries, APITimeoutError & llm_segment Fixes
# ---------------------------------------------------------------------------

def test_openaicompat_runtime_retries_timeout_and_fail_fast(monkeypatch):
    fake_settings = _FakeSettings({
        "retry_nums": 1,  # Legacy default <= 1 must still retry >= 3 times
        "aitrans_compact": True,
    })
    monkeypatch.setattr(base_mod, "settings", fake_settings)
    monkeypatch.setattr(oai_mod, "settings", fake_settings)

    inst = oai_mod.OpenAICampat(
        translate_type=translator.CHATGPT_INDEX,
        ainame="chatgpt",
        text_list=[_make_item(1, "Hello")],
        source_code="en",
        target_code="es",
        target_language_name="Spanish",
        is_test=True,
    )
    inst.api_key = "sk-test"
    inst.api_url = "https://api.openai.com/v1"
    inst.model_name = "gpt-test"

    sleep_delays = []
    monkeypatch.setattr(inst, "_sleep_interruptible", lambda d: (sleep_delays.append(d), False)[1])

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp429 = httpx.Response(429, request=req, headers={"retry-after": "0.5"})
    resp401 = httpx.Response(401, request=req)

    attempts = 0

    class RetrySuccessClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise openai.APITimeoutError(request=req)
            if attempts == 2:
                raise openai.RateLimitError("rate limit", response=resp429, body=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="<TRANSLATE_TEXT>Hola</TRANSLATE_TEXT>"))]
            )

    monkeypatch.setattr(oai_mod, "OpenAI", RetrySuccessClient)
    assert inst._item_task("Hello") == "Hola"
    assert attempts == 3
    assert len(sleep_delays) == 2

    # Runtime settings update (retry_nums=4) without module reload, all failing with APITimeoutError
    inst._close_clients()
    fake_settings["retry_nums"] = 4
    fail_attempts = 0

    class AlwaysTimeoutClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            nonlocal fail_attempts
            fail_attempts += 1
            raise openai.APITimeoutError(request=req)

    monkeypatch.setattr(oai_mod, "OpenAI", AlwaysTimeoutClient)
    with pytest.raises(openai.APITimeoutError):
        inst._item_task("Hello")
    assert fail_attempts == 5  # 1 initial + 4 retries

    # Auth error fails fast with StopTask in 1 attempt
    inst._close_clients()
    auth_attempts = 0

    class AuthFailClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            nonlocal auth_attempts
            auth_attempts += 1
            raise openai.AuthenticationError("bad key", response=resp401, body={"message": "bad key"})

    monkeypatch.setattr(oai_mod, "OpenAI", AuthFailClient)
    with pytest.raises(StopTask):
        inst._item_task("Hello")
    assert auth_attempts == 1


def test_llm_segment_empty_url_fallback_and_cleaning(monkeypatch):
    fake_params = {
        "chatgpt_api": "",  # Empty URL must fall back to official OpenAI URL
        "chatgpt_key": "sk-seg",
        "chatgpt_model": "gpt-4o-mini",
        "chatgpt_max_token": 8192,
    }
    monkeypatch.setattr(oai_mod, "params", fake_params)
    monkeypatch.setattr(oai_mod, "settings", _FakeSettings({"llm_chunk_size": 10, "retry_nums": 2}))

    seen_base_urls = []

    class FakeSegClient:
        def __init__(self, **kwargs):
            seen_base_urls.append(kwargs.get("base_url"))
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            content = (
                "<think>\ninternal reasoning\n</think>\n"
                "```srt\n"
                "<SRT>\n"
                "1\n00:00:00,000 --> 00:00:01,500\nFixed segment.\n"
                "</SRT>\n"
                "```"
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))]
            )

        def close(self):
            pass

    monkeypatch.setattr(oai_mod, "OpenAI", FakeSegClient)

    inst = oai_mod.OpenAICampat(ainame="chatgpt", uuid="seg-u1")
    res = inst.llm_segment([_make_item(1, "raw seg", 0, 1500)])
    assert seen_base_urls == ["https://api.openai.com/v1"]
    assert len(res) == 1
    assert res[0]["text"] == "Fixed segment."


# ---------------------------------------------------------------------------
# 6. Gemini Key Rotation, Client Pooling, Empty Proxy, Config & Retries
# ---------------------------------------------------------------------------

def test_gemini_concurrency_key_rotation_pooling_proxy_and_config(monkeypatch):
    fake_settings = _FakeSettings({
        "aisendsrt": True,
        "aitrans_thread": 1,
        "aitrans_concurrency": 4,
        "aitrans_compact": True,
        "retry_nums": 1,
    })
    monkeypatch.setattr(base_mod, "settings", fake_settings)
    monkeypatch.setattr(gemini_mod, "settings", fake_settings)
    monkeypatch.setattr(
        gemini_mod,
        "params",
        {
            "gemini_key": " k1 , k2 , k3 ",
            "gemini_model": "gemini-2.0-flash",
            "gemini_api": "",
            "gemini_maxtoken": 4096,
        },
    )

    created_clients = []
    closed_clients = 0
    seen_configs = []
    seen_keys_used = []
    lock = threading.Lock()

    class FakeModels:
        def __init__(self, api_key):
            self.api_key = api_key

        def generate_content_stream(self, *, model, contents, config):
            with lock:
                seen_keys_used.append(self.api_key)
                seen_configs.append(config)
            user_txt = contents[0].parts[0].text
            inp = user_txt.split("<INPUT>")[1].split("</INPUT>")[0].strip()
            if inp == "[1] Line 1":
                text = "[1] Linea 1"
            else:
                text = f"<think>thought\nmultiline</think>\n```srt\n{inp.replace('Line', 'Linea')}\n```"
            return [SimpleNamespace(text=text)]

    class FakeGenaiClient:
        def __init__(self, *, api_key, http_options):
            with lock:
                created_clients.append((api_key, http_options))
            # Verify empty proxy_str does NOT inject client_args={'proxy': ''}
            assert http_options.client_args is None
            self.models = FakeModels(api_key)

        def close(self):
            nonlocal closed_clients
            with lock:
                closed_clients += 1

    monkeypatch.setattr(gemini_mod.genai, "Client", FakeGenaiClient)

    items = [_make_item(i, f"Line {i}", i * 500, (i + 1) * 500) for i in range(1, 13)]
    g = gemini_mod.Gemini(
        translate_type=translator.GEMINI_INDEX,
        text_list=items,
        source_code="en",
        target_code="es",
        target_language_name="Spanish",
        is_test=True,
    )
    g.proxy_str = ""  # Must not crash HttpOptions

    out = g.run()
    assert [x["text"] for x in out] == [f"Linea {i}" for i in range(1, 13)]
    assert len(created_clients) == 3  # Pooled per unique key (k1, k2, k3)
    assert closed_clients == 3
    assert set(seen_keys_used) == {"k1", "k2", "k3"}

    # gemini-2.0-flash must retain system_instruction & safety_settings (not wiped)
    cfg = seen_configs[0]
    assert cfg.system_instruction is not None and len(cfg.system_instruction) == 1
    assert cfg.safety_settings is not None and len(cfg.safety_settings) == 4
    assert cfg.thinking_config is None


def test_gemini_retries_429_500_timeout_and_fails_fast_on_403(monkeypatch):
    fake_settings = _FakeSettings({"retry_nums": 1})
    monkeypatch.setattr(base_mod, "settings", fake_settings)
    monkeypatch.setattr(gemini_mod, "settings", fake_settings)
    monkeypatch.setattr(
        gemini_mod,
        "params",
        {"gemini_key": "k1,k2", "gemini_model": "gemini-2.5-flash", "gemini_api": ""},
    )

    g = gemini_mod.Gemini(
        translate_type=translator.GEMINI_INDEX,
        text_list=[_make_item(1, "Hello")],
        source_code="en",
        target_code="es",
        target_language_name="Spanish",
        is_test=True,
    )
    sleep_calls = []
    monkeypatch.setattr(g, "_sleep_interruptible", lambda d: (sleep_calls.append(d), False)[1])

    call_keys = []

    class FakeGenaiClient:
        def __init__(self, *, api_key, http_options):
            self.api_key = api_key
            self.models = self

        def generate_content_stream(self, **kwargs):
            call_keys.append(self.api_key)
            if len(call_keys) == 1:
                raise genai_errors.APIError(429, {"error": {"message": "rate limit"}})
            if len(call_keys) == 2:
                raise httpx.ConnectTimeout("connect timeout")
            if len(call_keys) == 3:
                raise genai_errors.APIError(500, {"error": {"message": "server error"}})
            return [SimpleNamespace(text="Hola")]

        def close(self):
            pass

    monkeypatch.setattr(gemini_mod.genai, "Client", FakeGenaiClient)
    assert g._item_task("Hello") == "Hola"
    assert call_keys == ["k1", "k2", "k1", "k2"]  # Key rotated across retries
    assert len(sleep_calls) == 3

    # 403 must fail fast with StopTask in 1 attempt
    g._unload()
    fail_calls = 0

    class ForbiddenClient:
        def __init__(self, *, api_key, http_options):
            self.models = self

        def generate_content_stream(self, **kwargs):
            nonlocal fail_calls
            fail_calls += 1
            raise genai_errors.APIError(403, {"error": {"message": "forbidden"}})

        def close(self):
            pass

    monkeypatch.setattr(gemini_mod.genai, "Client", ForbiddenClient)
    with pytest.raises(StopTask):
        g._item_task("Hello")
    assert fail_calls == 1


# ---------------------------------------------------------------------------
# 7. Google URL Encoding, html.unescape, timeout=15, verify=True & Retries
# ---------------------------------------------------------------------------

def test_google_params_unescape_verify_timeout_session_and_retries(monkeypatch):
    fake_settings = _FakeSettings({"retry_nums": 1, "trans_thread": 2, "trans_concurrency": 1})
    monkeypatch.setattr(base_mod, "settings", fake_settings)
    monkeypatch.setattr(google_mod, "settings", fake_settings)

    seen_calls = []
    sessions_created = 0
    sessions_closed = 0

    class FakeResponse:
        def __init__(self, status_code, text):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                err = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
                err.response = self
                raise err

    class FakeSession:
        def __init__(self):
            nonlocal sessions_created
            sessions_created += 1
            self.headers = {}
            self.proxies = {}

        def get(self, url, *, params=None, timeout=None, verify=None, **kwargs):
            seen_calls.append((url, params, timeout, verify))
            if len(seen_calls) == 1:
                raise requests.exceptions.ConnectTimeout("connect timeout")
            if len(seen_calls) == 2:
                return FakeResponse(429, "rate limited")
            return FakeResponse(
                200,
                '<div class="result-container">It&#39;s &quot;A &amp; B = 100% #1&quot;\nline2</div>',
            )

        def close(self):
            nonlocal sessions_closed
            sessions_closed += 1

    monkeypatch.setattr(google_mod.requests, "Session", FakeSession)

    g = google_mod.Google(
        translate_type=translator.GOOGLE_INDEX,
        text_list=[
            _make_item(1, 'It\'s "A & B = 100% #1"'),
            _make_item(2, "line2"),
        ],
        source_code="en",
        target_code="fr",
        target_language_name="fr",
        is_test=True,
    )
    monkeypatch.setattr(g, "_sleep_interruptible", lambda d: False)

    out = g.run()
    assert out[0]["text"] == 'It\'s "A & B = 100% #1"'
    assert out[1]["text"] == "line2"
    assert sessions_created == 1
    assert sessions_closed == 1
    assert len(seen_calls) == 3
    url, params_sent, timeout_sent, verify_sent = seen_calls[-1]
    assert url == "https://translate.google.com/m"
    assert params_sent == {
        "sl": "en",
        "tl": "fr",
        "hl": "fr",
        "q": 'It\'s "A & B = 100% #1"\nline2',
    }
    assert timeout_sent == 15
    assert verify_sent is True


# ---------------------------------------------------------------------------
# 8. Prefix Caching, Glossary Filtering, Cache Invalidation & Token Reduction
# ---------------------------------------------------------------------------

def test_prefix_cache_invariance_glossary_filtering_and_lru_invalidation(tmp_path, monkeypatch):
    # Create temporary ROOT_DIR with prompts and glossary.txt
    srt_dir = tmp_path / "videotrans" / "prompts" / "srt"
    srt_dir.mkdir(parents=True)
    real_srt = Path(__file__).resolve().parents[1] / "videotrans" / "prompts" / "srt" / "chatgpt.txt"
    (srt_dir / "chatgpt.txt").write_text(real_srt.read_text(encoding="utf-8"), encoding="utf-8")
    (srt_dir / "gemini.txt").write_text(real_srt.read_text(encoding="utf-8"), encoding="utf-8")

    glossary_path = tmp_path / "videotrans" / "glossary.txt"
    glossary_path.write_text("AI=Artificial Intelligence\nOpenAI=OpenAI Lab\n神经网络=Neural Net\n", encoding="utf-8")

    monkeypatch.setattr(prompts_mod, "ROOT_DIR", str(tmp_path))
    prompts_mod._read_text_cached.cache_clear()
    prompts_mod._parse_glossary_cached.cache_clear()

    # Check word-boundary protection for short ASCII "AI" vs "said"
    entries = prompts_mod.load_glossary_entries()
    assert prompts_mod.filter_glossary_for_text(entries, "He said hello.") == []
    assert prompts_mod.filter_glossary_for_text(entries, "AI and 神经网络 work.") == [
        ("AI", "Artificial Intelligence"),
        ("神经网络", "Neural Net"),
    ]

    # Verify cache hit on repeated load & invalidation on mtime/content update
    misses_before = prompts_mod._parse_glossary_cached.cache_info().misses
    prompts_mod.load_glossary_entries()
    hits_after = prompts_mod._parse_glossary_cached.cache_info().hits
    assert hits_after >= 1

    time.sleep(0.02)
    glossary_path.write_text("GPU=Graphics Unit\n", encoding="utf-8")
    os.utime(glossary_path, (time.time() + 2, time.time() + 2))
    updated_entries = prompts_mod.load_glossary_entries()
    assert updated_entries == (("GPU", "Graphics Unit"),)
    assert prompts_mod._parse_glossary_cached.cache_info().misses > misses_before

    # Verify static system prompt invariance across OpenAICampat and Gemini
    glossary_path.write_text("Quantum=Cuantico\nNebula=Nebulosa\n", encoding="utf-8")
    os.utime(glossary_path, (time.time() + 4, time.time() + 4))

    monkeypatch.setattr(oai_mod, "settings", _FakeSettings({"aisendsrt": True}))
    monkeypatch.setattr(gemini_mod, "settings", _FakeSettings({"aisendsrt": True}))
    monkeypatch.setattr(gemini_mod, "params", {"gemini_key": "k1", "gemini_model": "gemini-2.5-flash", "gemini_api": ""})

    oai = oai_mod.OpenAICampat(
        translate_type=translator.CHATGPT_INDEX,
        ainame="chatgpt",
        text_list=[_make_item(1, "Quantum"), _make_item(2, "Plain line")],
        source_code="en",
        target_code="es",
        target_language_name="Spanish",
    )
    m1 = oai._build_messages("[1] Quantum physics")
    m2 = oai._build_messages("[1] Plain everyday sentence")
    assert m1[0]["content"] == m2[0]["content"]
    assert "Quantum" not in m1[0]["content"]
    assert "Cuantico" in m1[1]["content"]
    assert "Nebulosa" not in m1[1]["content"]
    assert "Glossary" not in m2[1]["content"]

    gem = gemini_mod.Gemini(
        translate_type=translator.GEMINI_INDEX,
        text_list=[_make_item(1, "Quantum")],
        source_code="en",
        target_code="es",
        target_language_name="Spanish",
    )
    u1 = gem._build_user_message("[1] Quantum physics")
    u2 = gem._build_user_message("[1] Plain everyday sentence")
    assert "Quantum" not in gem.system_prompt
    assert "Cuantico" in u1
    assert "Glossary" not in u2


def test_prompt_and_compact_payload_char_reduction_benchmark():
    """Reproducible benchmark verifying >= 25% input+output char reduction vs legacy SRT prompt."""
    cues = [
        _make_item(i, f"Subtitle cue number {i} spoken naturally.", (i - 1) * 2000, i * 2000)
        for i in range(1, 21)
    ]
    translated_texts = [f"Subtítulo número {i} hablado natural." for i in range(1, 21)]

    # Legacy baseline: 5,956-char template + full SRT blocks in & out
    legacy_template_chars = 5956
    legacy_input_srt = "\n\n".join(
        f"{c['line']}\n{c['time']}\n{c['text']}" for c in cues
    )
    legacy_output_srt = "\n\n".join(
        f"{c['line']}\n{c['time']}\n{t}" for c, t in zip(cues, translated_texts)
    )

    # Optimized: streamlined split prompt + compact [id] text payload in & out
    sys_prompt, usr_template = prompts_mod.build_prompt_parts("chatgpt", aisendsrt=True, lang="Spanish")
    compact_in = base_mod._encode_indexed_batch(cues)
    compact_out = "\n".join(f"[{i}] {t}" for i, t in enumerate(translated_texts, 1))
    opt_user = prompts_mod.render_user_prompt(usr_template, batch_input=compact_in)

    # 1-batch total (system + user + output)
    legacy_1b_total = legacy_template_chars + len(legacy_input_srt) + len(legacy_output_srt)
    opt_1b_total = len(sys_prompt) + len(opt_user) + len(compact_out)
    reduction_1b = 1.0 - (opt_1b_total / legacy_1b_total)
    assert reduction_1b >= 0.25, f"Expected >=25% 1-batch reduction, got {reduction_1b:.2%}"

    # 3-batch total & payload-only comparison
    legacy_3b_total = 3 * legacy_template_chars + len(legacy_input_srt) + len(legacy_output_srt)
    opt_3b_total = 3 * (len(sys_prompt) + len(prompts_mod.render_user_prompt(usr_template, batch_input=""))) + len(compact_in) + len(compact_out)
    reduction_3b = 1.0 - (opt_3b_total / legacy_3b_total)
    assert reduction_3b >= 0.25, f"Expected >=25% 3-batch reduction, got {reduction_3b:.2%}"

    payload_only_reduction = 1.0 - ((len(compact_in) + len(compact_out)) / (len(legacy_input_srt) + len(legacy_output_srt)))
    assert payload_only_reduction >= 0.25


# ---------------------------------------------------------------------------
# 9. Strict ID Reconciliation, Bracketed Dialogue & check_target_sub
# ---------------------------------------------------------------------------

def test_strict_id_reconciliation_targeted_retry_and_check_target_sub(monkeypatch):
    fake_settings = _FakeSettings({
        "aisendsrt": True,
        "aitrans_thread": 3,
        "aitrans_concurrency": 1,
        "aitrans_compact": True,
        "translation_wait": 0,
    })
    monkeypatch.setattr(base_mod, "settings", fake_settings)

    calls = []

    class PartialDropTrans(base_mod.BaseTrans):
        def _item_task(self, data):
            calls.append(data)
            if len(calls) == 1:
                # Drops [2], includes bracketed dialogue [applause] & [2024]
                return "[1] [applause] Bienvenido [2024]\n[3] Adiós"
            # Targeted retry receives only the missing cue re-indexed as [1]
            return "[1] Segunda línea"

    items = [
        _make_item(10, "[applause] Welcome [2024]", 0, 1000),
        _make_item(11, "Second line", 1000, 2000),
        _make_item(12, "Goodbye", 2000, 3000),
    ]
    items[1]["spk"] = "spk_2"

    tr_inst = PartialDropTrans(
        translate_type=translator.CHATGPT_INDEX,
        text_list=items,
        source_code="en",
        target_code="es",
        is_test=True,
    )
    out = tr_inst.run()
    assert len(calls) == 2
    assert "[1] Second line" in calls[1]
    assert [x["text"] for x in out] == [
        "[applause] Bienvenido [2024]",
        "Segunda línea",
        "Adiós",
    ]
    assert out[1]["line"] == 11
    assert out[1]["spk"] == "spk_2"
    assert out[1]["time"] == "00:00:01,000 --> 00:00:02,000"

    # check_target_sub: equal-length keeps immutable source timestamps; mismatch handles <= 50ms drift
    bt = BaseTask()
    src = [
        _make_item(1, "A", 1000, 2000),
        _make_item(2, "B", 2500, 3500),
    ]
    drifted_equal = [
        _make_item(1, "A_es", 1005, 2005),
        _make_item(2, "B_es", 2500, 3500),
    ]
    checked_eq = bt.check_target_sub(src, drifted_equal)
    assert checked_eq[0]["text"] == "A_es"
    assert checked_eq[0]["time"] == src[0]["time"]

    drifted_mismatch = [
        _make_item(1, "A_es", 1020, 2010),  # 20ms drift -> matched
        _make_item(2, "B_es", 2500, 3500),
        _make_item(3, "Extra", 5000, 6000),
    ]
    checked_mm = bt.check_target_sub(src, drifted_mismatch)
    assert [x["text"] for x in checked_mm] == ["A_es", "B_es"]


# ---------------------------------------------------------------------------
# 10. /api/translate Temp Cleanup & run_staged_translation Progress Events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_translate_cleans_temp_artifacts_on_success_and_failure(tmp_path, monkeypatch):
    temp_dir = tmp_path / "app_temp"
    temp_dir.mkdir()
    import videotrans.configure.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "TEMP_DIR", str(temp_dir))

    segs = [{"id": 1, "startSec": 0.0, "endSec": 1.5, "sourceText": "Hello"}]

    # Case 1: success
    def ok_runner(request, event_sink, token):
        return TaskResult("j1", TaskStatus.SUCCEEDED, tmp_path, segments=tuple(segs))

    app = create_app(upload_dir=tmp_path / "uploads", translation_runner=ok_runner)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        res = await client.post("/api/translate", json={"segments": segs})
        assert res.status == 200
        assert list(temp_dir.iterdir()) == []
    finally:
        await client.close()

    # Case 2: failure
    def fail_runner(request, event_sink, token):
        from videotrans.task.orchestrator import TaskFailure
        return TaskResult(
            "j2",
            TaskStatus.FAILED,
            tmp_path,
            failure=TaskFailure("trans_failed", "trans", "boom", "trace"),
        )

    app_fail = create_app(upload_dir=tmp_path / "uploads2", translation_runner=fail_runner)
    client_fail = TestClient(TestServer(app_fail))
    await client_fail.start_server()
    try:
        res = await client_fail.post("/api/translate", json={"segments": segs})
        assert res.status == 400
        assert list(temp_dir.iterdir()) == []
    finally:
        await client_fail.close()


def test_run_staged_translation_emits_per_batch_progress(tmp_path, monkeypatch):
    fake_settings = _FakeSettings({
        "aisendsrt": True,
        "aitrans_thread": 2,
        "aitrans_concurrency": 1,
        "aitrans_compact": True,
        "translation_wait": 0,
    })
    monkeypatch.setattr(base_mod, "settings", fake_settings)

    class BatchProgressTrans(base_mod.BaseTrans):
        def _item_task(self, data):
            lines = [ln for ln in str(data).splitlines() if ln.strip()]
            return "\n".join(ln.replace("Src", "Tgt") for ln in lines)

    def fake_run(**kwargs):
        return BatchProgressTrans(
            translate_type=translator.CHATGPT_INDEX,
            text_list=kwargs["text_list"],
            source_code=kwargs["source_code"],
            target_code=kwargs["target_code"],
            aisendsrt=True,
            is_test=True,
            event_sink=kwargs["event_sink"],
            cancellation_token=kwargs["cancellation_token"],
        ).run()

    monkeypatch.setattr(translator, "run", fake_run)

    media_file = tmp_path / "video.mp4"
    media_file.touch()
    from videotrans.configure.config import TEMP_DIR
    cache_dir = Path(TEMP_DIR) / "cache-progress-test"
    cache_dir.mkdir(parents=True, exist_ok=True)

    segments = [
        {
            "id": i,
            "startSec": float(i - 1),
            "endSec": float(i),
            "sourceText": f"Src line {i}",
        }
        for i in range(1, 7)
    ]
    events = []
    res = run_staged_translation(
        TaskRequest({
            "name": media_file.as_posix(),
            "target_dir": tmp_path.as_posix(),
            "cache_folder": cache_dir.as_posix(),
            "source_language_code": "en",
            "target_language_code": "es",
            "translate_type": translator.CHATGPT_INDEX,
            "aisendsrt": True,
            "segments": segments,
            "uuid": "prog-test",
        }),
        event_sink=events.append,
    )
    assert res.status == TaskStatus.SUCCEEDED
    progress_vals = [e.progress for e in events if e.kind == EventKind.PROGRESS and e.progress is not None]
    # Initial 35.0, then 3 batch updates between 35.0 and 95.0, then 100.0
    assert progress_vals[0] == 35.0
    assert progress_vals[-1] == 100.0
    mid = [p for p in progress_vals if 35.0 < p <= 95.0]
    assert len(mid) == 3
    assert mid == sorted(mid)


# ---------------------------------------------------------------------------
# 11. Iteration 2 Regression Tests: Cache Contention, Sibling Abort, Parser & Time Bounds
# ---------------------------------------------------------------------------

def test_concurrent_same_key_cache_io_and_e2e_run(isolated_cache, monkeypatch):
    fake_settings = _FakeSettings({
        "aisendsrt": True,
        "aitrans_thread": 2,
        "aitrans_concurrency": 4,
        "aitrans_compact": True,
        "translation_wait": 0,
    })
    monkeypatch.setattr(base_mod, "settings", fake_settings)

    # 1. Direct 12-thread _set_cache / _get_cache contention on the exact same key
    probe_trans = base_mod.BaseTrans(
        translate_type=translator.CHATGPT_INDEX,
        text_list=[_make_item(1, "Thank you.")],
        source_code="en",
        target_code="es",
        is_test=False,
    )
    errors = []
    barrier = threading.Barrier(12)

    def _hammer_cache(tid: int):
        try:
            barrier.wait(timeout=5.0)
            for step in range(25):
                payload = "[1] Thank you.\n[2] See you soon."
                probe_trans._set_cache(payload, f"[1] Gracias ({tid}-{step})\n[2] Hasta pronto")
                got = probe_trans._get_cache(payload)
                assert got is not None and "[1] Gracias" in got
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_hammer_cache, args=(i,)) for i in range(12)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=15.0)
    assert errors == []

    # Clear cache before Part 2 so e2e run exercises fresh concurrent _set_cache + _get_cache
    for f in (isolated_cache / "translate_cache").glob("*.txt"):
        f.unlink()

    # 2. End-to-end BaseTrans.run() on 24 identical 2-cue batches with concurrency=4
    class IdenticalBatchTrans(base_mod.BaseTrans):
        def _item_task(self, data):
            time.sleep(0.01)
            return "[1] Gracias.\n[2] Hasta pronto."

    items = []
    for i in range(24):
        base_idx = i * 2 + 1
        items.append(_make_item(base_idx, "Thank you.", i * 2000, i * 2000 + 1000))
        items.append(_make_item(base_idx + 1, "See you soon.", i * 2000 + 1000, (i + 1) * 2000))

    e2e_trans = IdenticalBatchTrans(
        translate_type=translator.CHATGPT_INDEX,
        text_list=items,
        source_code="en",
        target_code="es",
        is_test=False,
    )
    out = e2e_trans.run()
    assert len(out) == 48
    assert [x["text"] for x in out] == ["Gracias.", "Hasta pronto."] * 24
    leftover_tmps = list((isolated_cache / "translate_cache").glob("tr_*.tmp"))
    assert leftover_tmps == []


def test_concurrent_sibling_fail_fast_wakes_sleeping_workers(monkeypatch):
    fake_settings = _FakeSettings({
        "aisendsrt": True,
        "aitrans_thread": 1,
        "aitrans_concurrency": 3,
        "aitrans_compact": True,
        "translation_wait": 0,
    })
    monkeypatch.setattr(base_mod, "settings", fake_settings)

    sleep_entered = threading.Event()
    sleep_results = []
    token = CancellationToken()

    class SiblingAbortTrans(base_mod.BaseTrans):
        def _item_task(self, data):
            if "[1] Fail cue" in str(data):
                # Ensure sibling workers are already inside _sleep_interruptible
                sleep_entered.wait(timeout=2.0)
                raise StopTask("fatal auth error")
            sleep_entered.set()
            aborted = self._sleep_interruptible(10.0)
            sleep_results.append(aborted)
            return None

    items = [
        _make_item(1, "Sleep cue 1", 0, 1000),
        _make_item(2, "Sleep cue 2", 1000, 2000),
        _make_item(3, "Fail cue", 2000, 3000),
    ]
    inst = SiblingAbortTrans(
        translate_type=translator.CHATGPT_INDEX,
        text_list=items,
        source_code="en",
        target_code="es",
        is_test=True,
        cancellation_token=token,
    )
    t0 = time.monotonic()
    with pytest.raises(StopTask, match="fatal auth error"):
        inst.run()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0
    # Wait briefly for any sibling worker unwinding from _sleep_interruptible
    deadline = time.monotonic() + 1.0
    while not sleep_results and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sleep_results and all(r is True for r in sleep_results)


def test_parse_indexed_response_discards_out_of_range_but_keeps_years():
    assert base_mod._parse_indexed_response(
        "[1] Uno\n[2] Dos\n[3] Tres\n[99] Hallucinated extra line", 3
    ) == {1: "Uno", 2: "Dos", 3: "Tres"}
    assert base_mod._parse_indexed_response(
        "[1] Uno\n[99] Extra\nTrailing line\n[2] Dos\n[3] Tres", 3
    ) == {1: "Uno", 2: "Dos", 3: "Tres"}
    assert base_mod._parse_indexed_response(
        "[0] Preamble\n[1] Uno\n[0] Stray\n[2] Dos\n[3] Tres", 3
    ) == {1: "Uno", 2: "Dos", 3: "Tres"}
    assert base_mod._parse_indexed_response(
        "[1] En\n[2024] fue bueno\n[2] Dos\n[3] Tres", 3
    ) == {1: "En [2024] fue bueno", 2: "Dos", 3: "Tres"}


def test_legacy_srt_fallback_in_text_and_srt_batches(monkeypatch):
    fake_settings = _FakeSettings({
        "aisendsrt": False,
        "trans_thread": 2,
        "aitrans_concurrency": 1,
        "aitrans_compact": True,
        "translation_wait": 0,
    })
    monkeypatch.setattr(base_mod, "settings", fake_settings)

    class LegacySrtInTextTrans(base_mod.BaseTrans):
        def _item_task(self, data):
            return (
                "1\n00:00:00,000 --> 00:00:01,000\nUno\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\n[2] Dos"
            )

    text_trans = LegacySrtInTextTrans(
        translate_type=translator.CHATGPT_INDEX,
        text_list=[_make_item(1, "One", 0, 1000), _make_item(2, "Two", 1000, 2000)],
        source_code="en",
        target_code="es",
        aisendsrt=False,
        is_test=True,
    )
    text_out = text_trans.run()
    assert [x["text"] for x in text_out] == ["Uno", "[2] Dos"]

    fake_settings["aisendsrt"] = True
    fake_settings["aitrans_thread"] = 2

    class LegacySrtInCompactSrtTrans(base_mod.BaseTrans):
        def _item_task(self, data):
            return (
                "1\n00:00:00,000 --> 00:00:01,000\nUno\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\n[2] Dos"
            )

    srt_trans = LegacySrtInCompactSrtTrans(
        translate_type=translator.CHATGPT_INDEX,
        text_list=[_make_item(1, "One", 0, 1000), _make_item(2, "Two", 1000, 2000)],
        source_code="en",
        target_code="es",
        aisendsrt=True,
        is_test=True,
    )
    srt_out = srt_trans.run()
    assert [x["text"] for x in srt_out] == ["Uno", "[2] Dos"]


def test_check_target_sub_parses_time_string_when_bounds_default_zero():
    bt = BaseTask()
    src = [
        _make_item(1, "A", 0, 30),
        _make_item(2, "B", 5000, 6000),
    ]
    # Default start_time=0, end_time=0 on SrtItem must NOT override the 5s time string
    tgt = [
        SrtItem(line=99, time="00:00:05,020 --> 00:00:06,020", text="B_es"),
        SrtItem(line=88, time="00:00:09,000 --> 00:00:10,000", text="Extra1"),
        SrtItem(line=77, time="00:00:11,000 --> 00:00:12,000", text="Extra2"),
    ]
    checked = bt.check_target_sub(src, tgt)
    assert [x["text"] for x in checked] == ["", "B_es"]
