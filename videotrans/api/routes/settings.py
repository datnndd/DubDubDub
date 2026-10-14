# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from videotrans import recognition, translator, tts
from videotrans.configure import config as runtime_config
from videotrans.configure.config import settings as global_settings, params as global_params
from videotrans.configure.contants import DEFAULT_GEMINI_MODEL, ELEVENLABS_TTS_MODELS
from videotrans.configure.excepts import get_msg_from_except
from videotrans.core import secret_store, storage_config
from videotrans.util.help_role import role_menu
from videotrans.api.catalog import (
    ASR_BY_ID,
    ASR_PROVIDERS,
    asr_models,
    TIMING_MODES,
    TRANSLATION_BY_ID,
    TRANSLATION_MODES,
    TRANSLATION_PROVIDERS,
    TTS_PROVIDER_ALIASES,
)
from videotrans.api.task_params import _optional_index, _translation_mode
from videotrans.api.provider_helpers import (
    _save_asr_settings,
    _save_translation_settings,
    _translation_snapshot,
)

router = APIRouter()

class AsrSettingsRequest(BaseModel):
    apiKey: Optional[str] = Field(None, alias="api_key")
    model: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class AsrTestRequest(BaseModel):
    apiKey: Optional[str] = Field(None, alias="api_key")
    model: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class TranslationSettingsRequest(BaseModel):
    apiKey: Optional[str] = Field(None, alias="api_key")
    baseUrl: Optional[str] = Field(None, alias="base_url")
    model: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class TranslationTestRequest(BaseModel):
    apiKey: Optional[str] = Field(None, alias="api_key")
    baseUrl: Optional[str] = Field(None, alias="base_url")
    model: Optional[str] = None
    translationMode: Optional[Any] = Field(None, alias="translation_mode")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class ProviderSettingsRequest(BaseModel):
    apiKey: Optional[str] = Field(None, alias="api_key")
    token: Optional[str] = None
    model: Optional[str] = None
    baseUrl: Optional[str] = Field(None, alias="base_url")
    mirrorUrl: Optional[str] = Field(None, alias="mirror_url")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class ProviderTestRequest(BaseModel):
    apiKey: Optional[str] = Field(None, alias="api_key")
    token: Optional[str] = None
    baseUrl: Optional[str] = Field(None, alias="base_url")
    proxy: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class StorageSettingsRequest(BaseModel):
    output_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    models_dir: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class GeneralSettingsRequest(BaseModel):
    proxy: Optional[str] = None
    defaultSourceLanguage: Optional[str] = Field(None, alias="default_source_language")
    defaultTargetLanguage: Optional[str] = Field(None, alias="default_target_language")
    crf: Optional[int] = None
    preset: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


PROVIDER_SECRET_MAP = {
    "deepgram": "deepgram_apikey",
    "openai": "chatgpt_key",
    "chatgpt": "chatgpt_key",
    "deepseek": "deepseek_key",
    "gemini": "gemini_key",
    "elevenlabs": "elevenlabstts_key",
    "huggingface": "hf_token",
    "hf": "hf_token",
}


async def probe_provider_connection(
    provider_id: str,
    api_key: str,
    base_url: str = "",
    proxy: str = "",
    custom_session: Any = None,
) -> tuple[bool, str]:
    pid = provider_id.lower()
    headers: dict[str, str] = {}
    url: str = ""

    if pid == "deepgram":
        url = "https://api.deepgram.com/v1/projects"
        headers["Authorization"] = f"Token {api_key}"
    elif pid in ("openai", "chatgpt"):
        from videotrans.util.network import process_openai_api
        b = process_openai_api(base_url) if base_url else "https://api.openai.com/v1"
        url = f"{b.rstrip('/')}/models"
        headers["Authorization"] = f"Bearer {api_key}"
    elif pid == "deepseek":
        b = base_url.rstrip("/") if base_url else "https://api.deepseek.com"
        url = f"{b}/models"
        headers["Authorization"] = f"Bearer {api_key}"
    elif pid == "gemini":
        b = base_url.rstrip("/") if base_url else "https://generativelanguage.googleapis.com"
        url = f"{b}/v1beta/models?key={api_key}"
    elif pid == "elevenlabs":
        url = "https://api.elevenlabs.io/v1/user"
        headers["xi-api-key"] = api_key
    elif pid in ("huggingface", "hf"):
        url = "https://huggingface.co/api/whoami-v2"
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        return False, f"Unsupported provider probe: {provider_id}"

    client_proxy = proxy or global_settings.proxy or None

    try:
        if custom_session is not None and hasattr(custom_session, "get"):
            # aiohttp session support
            async with custom_session.get(url, headers=headers, proxy=client_proxy) as resp:
                status = resp.status
                if status in (200, 201):
                    return True, "Connection successful"
                if status in (401, 403):
                    return False, f"Authentication failed (HTTP {status}): Invalid API key"
                return False, f"Provider returned HTTP {status}"
        else:
            async with httpx.AsyncClient(proxy=client_proxy, timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code in (200, 201):
                    return True, "Connection successful"
                if resp.status_code in (401, 403):
                    return False, f"Authentication failed (HTTP {resp.status_code}): Invalid API key"
                return False, f"Provider returned HTTP {resp.status_code}"
    except Exception as exc:
        return False, f"Connection failed: {exc}"


async def fetch_remote_provider_models(
    provider_id: str,
    api_key: str,
    base_url: str = "",
    proxy: str = "",
    custom_session: Any = None,
) -> tuple[bool, list[str], str]:
    pid = provider_id.lower()
    headers: dict[str, str] = {}
    url: str = ""

    if pid == "deepgram":
        url = "https://api.deepgram.com/v1/models"
        headers["Authorization"] = f"Token {api_key}"
    elif pid in ("openai", "chatgpt"):
        from videotrans.util.network import process_openai_api
        b = process_openai_api(base_url) if base_url else "https://api.openai.com/v1"
        url = f"{b.rstrip('/')}/models"
        headers["Authorization"] = f"Bearer {api_key}"
    elif pid == "deepseek":
        b = base_url.rstrip("/") if base_url else "https://api.deepseek.com"
        url = f"{b}/models"
        headers["Authorization"] = f"Bearer {api_key}"
    elif pid == "gemini":
        b = base_url.rstrip("/") if base_url else "https://generativelanguage.googleapis.com"
        url = f"{b}/v1beta/models?key={api_key}"
    elif pid == "elevenlabs":
        url = "https://api.elevenlabs.io/v1/models"
        headers["xi-api-key"] = api_key
    else:
        return False, [], f"Provider {provider_id} does not support dynamic model listing"

    client_proxy = proxy or global_settings.proxy or None

    try:
        data: Any = None
        if custom_session is not None and hasattr(custom_session, "get"):
            async with custom_session.get(url, headers=headers, proxy=client_proxy) as resp:
                status = resp.status
                if status in (401, 403):
                    return False, [], f"Authentication failed (HTTP {status}): Invalid API key"
                if status not in (200, 201):
                    return False, [], f"Provider returned HTTP {status}"
                data = await resp.json()
        else:
            async with httpx.AsyncClient(proxy=client_proxy, timeout=12.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code in (401, 403):
                    return False, [], f"Authentication failed (HTTP {resp.status_code}): Invalid API key"
                if resp.status_code not in (200, 201):
                    return False, [], f"Provider returned HTTP {resp.status_code}"
                data = resp.json()

        models: list[str] = []

        if pid in ("openai", "chatgpt", "deepseek"):
            raw_list = data.get("data", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for item in raw_list:
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item["id"]).strip())
                elif isinstance(item, str):
                    models.append(item.strip())
            if pid in ("openai", "chatgpt") and "api.openai.com" in url:
                non_chat_prefixes = ("text-embedding", "dall-e", "tts-1", "whisper", "text-moderation", "babbage", "davinci")
                chat_models = [m for m in models if not any(m.startswith(p) for p in non_chat_prefixes)]
                if chat_models:
                    priority = ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
                    sorted_models = [p for p in priority if p in chat_models]
                    for m in sorted(chat_models):
                        if m not in sorted_models:
                            sorted_models.append(m)
                    models = sorted_models

        elif pid == "gemini":
            raw_models = data.get("models", []) if isinstance(data, dict) else []
            for m in raw_models:
                if isinstance(m, dict):
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods or not methods:
                        name = str(m.get("name", "")).strip()
                        if name.startswith("models/"):
                            name = name[7:]
                        if name and "embedding" not in name and "aqa" not in name:
                            models.append(name)
            if models:
                priority = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-lite-preview-02-05"]
                sorted_models = [p for p in priority if p in models]
                for m in sorted(models):
                    if m not in sorted_models:
                        sorted_models.append(m)
                models = sorted_models

        elif pid == "elevenlabs":
            raw_list = data if isinstance(data, list) else data.get("models", [])
            for item in raw_list:
                if isinstance(item, dict):
                    if item.get("can_do_text_to_speech", True):
                        mid = item.get("model_id")
                        if mid:
                            models.append(str(mid).strip())

        elif pid == "deepgram":
            raw_list = []
            if isinstance(data, dict):
                raw_list = data.get("models", []) or data.get("stt", [])
                if not raw_list:
                    for v in data.values():
                        if isinstance(v, list):
                            raw_list.extend(v)
            elif isinstance(data, list):
                raw_list = data
            for item in raw_list:
                if isinstance(item, dict):
                    name = item.get("canonical_name") or item.get("name") or item.get("architecture")
                    if name:
                        models.append(str(name).strip())
            if not models:
                models = ["nova-3", "nova-2", "nova-2-general", "nova-2-meeting", "enhanced", "base"]

        seen = set()
        deduped = []
        for m in models:
            if m and m not in seen:
                seen.add(m)
                deduped.append(m)

        if not deduped:
            return False, [], "No compatible models returned by provider"

        return True, deduped, f"Found {len(deduped)} models"
    except Exception as exc:
        return False, [], f"Failed to fetch models: {exc}"


@router.get("/api/options")
async def options_handler(request: Request) -> JSONResponse:
    languages = [{"code": code, "name": name} for code, name in translator.LANGNAME_DICT.items()]
    settings_store = getattr(request.app.state, "settings_store", None)
    asr_providers = []
    for provider in ASR_PROVIDERS:
        settings_key = provider.get("settingsKey")
        models = list(provider["models"])
        if provider["id"] == "deepgram":
            deepgram_cfg_models = [m.strip() for m in str(global_settings.get("deepgram_model", "")).split(",") if m.strip()]
            if deepgram_cfg_models:
                models = deepgram_cfg_models
            cur_m = str(
                (settings_store.get("model_name", "") if settings_store else "")
                or global_params.get("model_name", "nova-3")
            ).strip()
            if cur_m and cur_m not in models:
                models.insert(0, cur_m)
        is_configured = True
        if settings_key:
            is_configured = (
                bool(settings_store.get(settings_key))
                if settings_store is not None
                else secret_store.is_secret_configured(settings_key)
            )
        asr_providers.append({
            "id": provider["id"],
            "label": provider["label"],
            "recognType": provider["recognType"],
            "models": models,
            "requiresSettings": bool(settings_key),
            "configured": is_configured,
            "testable": bool(provider.get("thirdParty")),
        })
    return JSONResponse({
        "languages": languages,
        "asrProviders": asr_providers,
        "translationProviders": [
            _translation_snapshot(provider, settings_store)
            for provider in TRANSLATION_PROVIDERS
        ],
        "translationModes": [
            {"id": mode_id, "label": mode["label"], "description": mode["description"]}
            for mode_id, mode in TRANSLATION_MODES.items()
        ],
        "voices": list(enumerate(tts.TTS_NAME_LIST)),
        "defaults": {
            "sourceLanguage": str((settings_store.get("source_language", "") if settings_store else "") or global_params.get("source_language", "en")).strip() or "en",
            "targetLanguage": str((settings_store.get("target_language", "") if settings_store else "") or global_params.get("target_language", "vi")).strip() or "vi",
            "recognType": recognition.Deepgram,
            "modelName": str((settings_store.get("model_name", "") if settings_store else "") or global_params.get("model_name", "nova-3")).strip() or "nova-3",
            "timingMode": "voice",
            "translateType": translator.GOOGLE_INDEX,
            "translationMode": _translation_mode()[0],
            "ttsType": tts.DEFAULT_TTS,
        },
    })


@router.get("/api/voices")
async def voices_handler(request: Request) -> JSONResponse:
    if len(str(request.url.query)) > 8000:
        raise HTTPException(status_code=400, detail="Query line exceeds limit")
    raw_provider = (
        request.query_params.get("ttsType")
        or request.query_params.get("tts_type")
        or request.query_params.get("provider")
    )

    tts_type = 0
    if raw_provider is not None:
        val_str = str(raw_provider).strip().lower()
        if val_str in TTS_PROVIDER_ALIASES:
            tts_type = TTS_PROVIDER_ALIASES[val_str]
        else:
            tts_type = _optional_index(raw_provider, len(tts.TTS_NAME_LIST), default=0)

    language = (
        request.query_params.get("language")
        or request.query_params.get("target_language")
        or request.query_params.get("targetLanguage")
        or ""
    )
    role_provider = getattr(request.app.state, "role_provider", None) or role_menu
    try:
        raw_voices = role_provider(tts_type, langcode=language)
        if raw_voices is None or (isinstance(raw_voices, (list, tuple)) and len(raw_voices) == 0):
            return JSONResponse({"voices": ["No"]})
        voices = raw_voices
    except Exception:
        return JSONResponse({"voices": ["No"]})

    from videotrans.core import voice_store
    items: list[dict[str, Any]] = []
    try:
        custom_voices = voice_store.list_voices(provider=tts_type, active_only=True)
        custom_by_name = {v["name"].casefold(): v for v in custom_voices}
    except Exception:
        custom_voices = []
        custom_by_name = {}

    for v in voices:
        v_str = str(v)
        c_name = v_str[8:] if v_str.startswith("Custom: ") else v_str
        matched_custom = custom_by_name.get(c_name.casefold())
        if matched_custom:
            items.append({
                "id": matched_custom["id"],
                "name": matched_custom["name"],
                "provider": tts_type,
                "kind": "clone",
                "sampleUrl": f"/api/custom-voices/{matched_custom['id']}/audio",
            })
        elif v_str.lower() == "clone":
            items.append({
                "id": "clone",
                "name": "Clone Voice",
                "provider": tts_type,
                "kind": "clone",
            })
        elif v_str == "No":
            items.append({
                "id": "No",
                "name": "No Dubbing (Mute/Retain)",
                "provider": tts_type,
                "kind": "preset",
            })
        else:
            items.append({
                "id": v_str,
                "name": v_str,
                "provider": tts_type,
                "kind": "preset",
            })

    existing_item_ids = {it["id"] for it in items}
    for cv in custom_voices:
        if cv["id"] not in existing_item_ids:
            items.append({
                "id": cv["id"],
                "name": cv["name"],
                "provider": tts_type,
                "kind": "clone",
                "sampleUrl": f"/api/custom-voices/{cv['id']}/audio",
            })
            existing_item_ids.add(cv["id"])

    return JSONResponse({"voices": voices, "items": items})


@router.post("/api/asr-settings/{provider_id}")
async def asr_settings_handler(
    provider_id: str,
    payload: AsrSettingsRequest,
    request: Request,
) -> JSONResponse:
    provider = ASR_BY_ID.get(provider_id)
    if provider is None or not provider.get("settingsKey"):
        raise HTTPException(status_code=404, detail="This ASR provider has no WebUI API settings")
    try:
        snapshot = _save_asr_settings(provider, payload.model_dump(exclude_unset=True), getattr(request.app.state, "settings_store", None))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(snapshot)


@router.post("/api/asr-settings/{provider_id}/test")
async def test_asr_settings_handler(
    provider_id: str,
    payload: AsrTestRequest,
    request: Request,
) -> JSONResponse:
    provider = ASR_BY_ID.get(provider_id)
    if provider is None or not provider.get("thirdParty"):
        raise HTTPException(status_code=404, detail="This ASR provider does not use a third-party connection")

    model_name = str(payload.model or "").strip()
    if model_name not in asr_models(provider):
        raise HTTPException(status_code=400, detail=f"Model {model_name or 'missing'} is not supported by {provider['label']}")
    try:
        settings_store = getattr(request.app.state, "settings_store", None)
        if provider.get("settingsKey"):
            _save_asr_settings(provider, payload.model_dump(exclude_unset=True), settings_store)
        asr_tester = getattr(request.app.state, "asr_tester", None)
        result = await asyncio.to_thread(
            asr_tester, provider["recognType"], model_name
        )
    except Exception as exc:
        runtime_config.logger.exception("ASR connection test failed", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Connection test failed: {get_msg_from_except(exc)}") from exc
    return JSONResponse({
        "ok": True,
        "message": "Connection successful",
        "model": model_name,
        "result": result,
    })


@router.post("/api/translation-settings/{provider_id}")
async def translation_settings_handler(
    provider_id: str,
    payload: TranslationSettingsRequest,
    request: Request,
) -> JSONResponse:
    provider = TRANSLATION_BY_ID.get(provider_id)
    if provider is None or not provider.get("keyKey"):
        raise HTTPException(status_code=404, detail="This translation provider has no WebUI settings")
    try:
        snapshot = _save_translation_settings(provider, payload.model_dump(exclude_unset=True), getattr(request.app.state, "settings_store", None))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(snapshot)


@router.post("/api/translation-settings/{provider_id}/test")
async def test_translation_settings_handler(
    provider_id: str,
    payload: TranslationTestRequest,
    request: Request,
) -> JSONResponse:
    provider = TRANSLATION_BY_ID.get(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail="Unsupported translation provider")
    try:
        _, aisendsrt = _translation_mode(payload.translationMode)
        settings_store = getattr(request.app.state, "settings_store", None)
        if provider.get("keyKey"):
            _save_translation_settings(provider, payload.model_dump(exclude_unset=True), settings_store)
        translation_tester = getattr(request.app.state, "translation_tester", None)
        result = await asyncio.to_thread(
            translation_tester, provider["translateType"], aisendsrt
        )
    except Exception as exc:
        runtime_config.logger.exception("Translation connection test failed", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Connection test failed: {get_msg_from_except(exc)}") from exc
    return JSONResponse({"ok": True, "message": "Connection successful", "result": result})


@router.get("/api/settings")
async def get_settings_handler(request: Request) -> JSONResponse:
    storage_metrics = storage_config.get_storage_metrics()
    settings_store = getattr(request.app.state, "settings_store", None)

    deepgram_model = str((settings_store.get("model_name", "") if settings_store else "") or global_params.get("model_name", "nova-3")).strip()
    deepgram_models = [m.strip() for m in str(global_settings.get("deepgram_model", "nova-3,nova-2")).split(",") if m.strip()]
    if deepgram_model and deepgram_model not in deepgram_models:
        deepgram_models.insert(0, deepgram_model)

    chatgpt_model = str((settings_store.get("chatgpt_model", "") if settings_store else "") or global_params.get("chatgpt_model", "gpt-4o")).strip()
    chatgpt_models = [m.strip() for m in str(global_settings.get("chatgpt_model", "gpt-4o,gpt-4o-mini,gpt-3.5-turbo")).split(",") if m.strip()]
    if chatgpt_model and chatgpt_model not in chatgpt_models:
        chatgpt_models.insert(0, chatgpt_model)

    deepseek_model = str((settings_store.get("deepseek_model", "") if settings_store else "") or global_params.get("deepseek_model", "deepseek-chat")).strip()
    deepseek_models = [m.strip() for m in str(global_settings.get("deepseek_model", "deepseek-chat,deepseek-reasoner")).split(",") if m.strip()]
    if deepseek_model and deepseek_model not in deepseek_models:
        deepseek_models.insert(0, deepseek_model)

    gemini_model = str((settings_store.get("gemini_model", "") if settings_store else "") or global_params.get("gemini_model", "gemini-2.0-flash")).strip()
    gemini_models = [m.strip() for m in str(global_settings.get("gemini_model", "gemini-2.0-flash,gemini-1.5-pro,gemini-1.5-flash")).split(",") if m.strip()]
    if gemini_model and gemini_model not in gemini_models:
        gemini_models.insert(0, gemini_model)

    elevenlabs_model = str((settings_store.get("elevenlabstts_models", "") if settings_store else "") or global_params.get("elevenlabstts_models", "eleven_multilingual_v2")).strip()
    elevenlabs_models = [m.strip() for m in str(global_settings.get("elevenlabstts_models", ELEVENLABS_TTS_MODELS)).split(",") if m.strip()]
    if elevenlabs_model and elevenlabs_model not in elevenlabs_models:
        elevenlabs_models.insert(0, elevenlabs_model)

    def _resolve_provider_key(secret_key: str) -> str:
        if settings_store and settings_store.get(secret_key):
            return str(settings_store.get(secret_key)).strip()
        return (secret_store.resolve_secret(secret_key) or "").strip()

    providers = {
        "deepgram": {
            "id": "deepgram",
            "name": "Deepgram",
            "category": "asr",
            "configured": secret_store.is_secret_configured("deepgram_apikey"),
            "fromEnv": secret_store.is_from_env("deepgram_apikey"),
            "apiKey": _resolve_provider_key("deepgram_apikey"),
            "model": deepgram_model,
            "models": deepgram_models,
        },
        "openai": {
            "id": "openai",
            "name": "OpenAI ChatGPT",
            "category": "llm",
            "configured": secret_store.is_secret_configured("chatgpt_key"),
            "fromEnv": secret_store.is_from_env("chatgpt_key"),
            "apiKey": _resolve_provider_key("chatgpt_key"),
            "model": chatgpt_model,
            "models": chatgpt_models,
            "baseUrl": str((settings_store.get("chatgpt_api", "") if settings_store else "") or global_params.get("chatgpt_api", "")),
        },
        "deepseek": {
            "id": "deepseek",
            "name": "DeepSeek",
            "category": "llm",
            "configured": secret_store.is_secret_configured("deepseek_key"),
            "fromEnv": secret_store.is_from_env("deepseek_key"),
            "apiKey": _resolve_provider_key("deepseek_key"),
            "model": deepseek_model,
            "models": deepseek_models,
            "baseUrl": str((settings_store.get("deepseek_api", "") if settings_store else "") or global_params.get("deepseek_api", "https://api.deepseek.com/v1")),
        },
        "gemini": {
            "id": "gemini",
            "name": "Google Gemini",
            "category": "multimodal",
            "configured": secret_store.is_secret_configured("gemini_key"),
            "fromEnv": secret_store.is_from_env("gemini_key"),
            "apiKey": _resolve_provider_key("gemini_key"),
            "model": gemini_model,
            "models": gemini_models,
            "baseUrl": str((settings_store.get("gemini_api", "") if settings_store else "") or global_params.get("gemini_api", "")),
        },
        "elevenlabs": {
            "id": "elevenlabs",
            "name": "ElevenLabs",
            "category": "tts",
            "configured": secret_store.is_secret_configured("elevenlabstts_key"),
            "fromEnv": secret_store.is_from_env("elevenlabstts_key"),
            "apiKey": _resolve_provider_key("elevenlabstts_key"),
            "model": elevenlabs_model,
            "models": elevenlabs_models,
        },
        "huggingface": {
            "id": "huggingface",
            "name": "Hugging Face",
            "category": "weights",
            "configured": secret_store.is_secret_configured("hf_token"),
            "fromEnv": secret_store.is_from_env("hf_token"),
            "apiKey": _resolve_provider_key("hf_token"),
            "mirrorUrl": os.environ.get("HF_ENDPOINT", ""),
        },
    }

    general = {
        "proxy": global_settings.proxy or os.environ.get("HTTPS_PROXY", ""),
        "defaultSourceLanguage": str((settings_store.get("source_language", "") if settings_store else "") or global_params.get("source_language", "en")),
        "defaultTargetLanguage": str((settings_store.get("target_language", "") if settings_store else "") or global_params.get("target_language", "vi")),
        "crf": global_settings.get("crf", 23),
        "preset": global_settings.get("preset", "slow"),
    }

    return JSONResponse({
        "providers": providers,
        "storage": storage_metrics,
        "general": general,
    })


@router.put("/api/settings/providers/{category}/{provider_id}")
async def update_provider_settings_handler(
    category: str,
    provider_id: str,
    payload: ProviderSettingsRequest,
    request: Request,
) -> JSONResponse:
    provider_id = provider_id.lower()
    secret_key = PROVIDER_SECRET_MAP.get(provider_id)
    if not secret_key:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")

    settings_store = getattr(request.app.state, "settings_store", None)
    raw_key = payload.apiKey if payload.apiKey is not None else payload.token
    if raw_key is not None:
        key_val = str(raw_key).strip()
        secret_store.set_secret(secret_key, key_val)
        if secret_key == "hf_token":
            global_settings["hf_token"] = key_val
            global_settings.save()
        else:
            global_params[secret_key] = key_val
        if settings_store is not None:
            try:
                settings_store[secret_key] = key_val
            except Exception:
                pass

    if payload.model is not None:
        model_str = str(payload.model).strip()
        if provider_id == "deepgram":
            global_params["model_name"] = model_str
            global_params["stt_model_name"] = model_str
            if settings_store is not None:
                try:
                    settings_store["model_name"] = model_str
                    settings_store["stt_model_name"] = model_str
                except Exception:
                    pass
            cur_models = [m.strip() for m in str(global_settings.get("deepgram_model", "")).split(",") if m.strip()]
            if model_str and model_str not in cur_models:
                cur_models.insert(0, model_str)
                global_settings["deepgram_model"] = ",".join(cur_models)
                try:
                    global_settings.save()
                except Exception:
                    pass
        elif provider_id in ("openai", "chatgpt"):
            global_params["chatgpt_model"] = model_str
            if settings_store is not None:
                try:
                    settings_store["chatgpt_model"] = model_str
                except Exception:
                    pass
            cur_models = [m.strip() for m in str(global_settings.get("chatgpt_model", "")).split(",") if m.strip()]
            if model_str and model_str not in cur_models:
                cur_models.insert(0, model_str)
                global_settings["chatgpt_model"] = ",".join(cur_models)
                global_settings.save()
        elif provider_id == "deepseek":
            global_params["deepseek_model"] = model_str
            if settings_store is not None:
                try:
                    settings_store["deepseek_model"] = model_str
                except Exception:
                    pass
            cur_models = [m.strip() for m in str(global_settings.get("deepseek_model", "")).split(",") if m.strip()]
            if model_str and model_str not in cur_models:
                cur_models.insert(0, model_str)
                global_settings["deepseek_model"] = ",".join(cur_models)
                global_settings.save()
        elif provider_id == "gemini":
            global_params["gemini_model"] = model_str
            if settings_store is not None:
                try:
                    settings_store["gemini_model"] = model_str
                except Exception:
                    pass
            cur_models = [m.strip() for m in str(global_settings.get("gemini_model", "")).split(",") if m.strip()]
            if model_str and model_str not in cur_models:
                cur_models.insert(0, model_str)
                global_settings["gemini_model"] = ",".join(cur_models)
                global_settings.save()
        elif provider_id == "elevenlabs":
            global_params["elevenlabstts_models"] = model_str
            if settings_store is not None:
                try:
                    settings_store["elevenlabstts_models"] = model_str
                except Exception:
                    pass

    if payload.baseUrl is not None:
        b_str = str(payload.baseUrl).strip()
        if provider_id in ("openai", "chatgpt"):
            from videotrans.util.network import process_openai_api
            norm_b = process_openai_api(b_str) if b_str else ""
            global_params["chatgpt_api"] = norm_b
            if settings_store is not None:
                try:
                    settings_store["chatgpt_api"] = norm_b
                except Exception:
                    pass
        elif provider_id == "deepseek":
            global_params["deepseek_api"] = b_str
            if settings_store is not None:
                try:
                    settings_store["deepseek_api"] = b_str
                except Exception:
                    pass
        elif provider_id == "gemini":
            global_params["gemini_api"] = b_str
            if settings_store is not None:
                try:
                    settings_store["gemini_api"] = b_str
                except Exception:
                    pass

    if payload.mirrorUrl is not None and provider_id in ("huggingface", "hf"):
        m_str = str(payload.mirrorUrl).strip()
        if m_str:
            os.environ["HF_ENDPOINT"] = m_str
        else:
            os.environ.pop("HF_ENDPOINT", None)

    if provider_id in ("huggingface", "hf"):
        global_settings.save()
    else:
        global_params.save()
    if settings_store is not None and hasattr(settings_store, "save"):
        try:
            settings_store.save()
        except Exception:
            pass

    return JSONResponse({
        "ok": True,
        "providerId": provider_id,
        "configured": secret_store.is_secret_configured(secret_key),
        "message": "Settings saved successfully",
    })


@router.post("/api/settings/providers/{category}/{provider_id}/test")
async def test_provider_connection_handler(
    category: str,
    provider_id: str,
    payload: Optional[ProviderTestRequest] = None,
    request: Request = None,
) -> JSONResponse:
    provider_id = provider_id.lower()
    secret_key = PROVIDER_SECRET_MAP.get(provider_id)
    if not secret_key:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")

    api_key = str((payload.apiKey if payload else None) or (payload.token if payload else None) or "").strip()
    if not api_key:
        api_key = secret_store.resolve_secret(secret_key) or ""

    if not api_key:
        raise HTTPException(status_code=400, detail=f"API key is not configured for {provider_id}")

    base_url = str((payload.baseUrl if payload else None) or "").strip()
    if not base_url:
        if provider_id in ("openai", "chatgpt"):
            base_url = global_params.get("chatgpt_api", "")
        elif provider_id == "deepseek":
            base_url = global_params.get("deepseek_api", "")
        elif provider_id == "gemini":
            base_url = global_params.get("gemini_api", "")

    proxy = str((payload.proxy if payload else None) or global_settings.proxy or "").strip()

    req_app = request.app if request is not None else None
    probe_fn = (getattr(req_app.state, "provider_probe", None) if req_app else None) or probe_provider_connection
    ok, msg = await probe_fn(provider_id, api_key, base_url, proxy)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    models: list[str] = []
    try:
        fetcher_fn = (getattr(req_app.state, "provider_model_fetcher", None) if req_app else None) or fetch_remote_provider_models
        m_ok, m_list, _ = await fetcher_fn(provider_id, api_key, base_url, proxy)
        if m_ok and m_list:
            models = m_list
            models_csv = ",".join(models)
            if provider_id in ("openai", "chatgpt"):
                global_settings["chatgpt_model"] = models_csv
                global_settings.save()
            elif provider_id == "deepseek":
                global_settings["deepseek_model"] = models_csv
                global_settings.save()
            elif provider_id == "gemini":
                global_settings["gemini_model"] = models_csv
                global_settings.save()
            elif provider_id == "deepgram":
                global_settings["deepgram_model"] = models_csv
                global_settings.save()
            elif provider_id == "elevenlabs":
                global_settings["elevenlabstts_models"] = models_csv
                global_settings.save()
    except Exception:
        pass

    return JSONResponse({"ok": True, "message": msg, "models": models})


@router.post("/api/settings/providers/{category}/{provider_id}/models")
async def get_provider_models_handler(
    category: str,
    provider_id: str,
    payload: Optional[ProviderTestRequest] = None,
    request: Request = None,
) -> JSONResponse:
    provider_id = provider_id.lower()
    secret_key = PROVIDER_SECRET_MAP.get(provider_id)
    if not secret_key:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_id}")

    api_key = str((payload.apiKey if payload else None) or (payload.token if payload else None) or "").strip()
    if not api_key:
        api_key = secret_store.resolve_secret(secret_key) or ""

    if not api_key:
        raise HTTPException(status_code=400, detail=f"API key is not configured for {provider_id}")

    base_url = str((payload.baseUrl if payload else None) or "").strip()
    if not base_url:
        if provider_id in ("openai", "chatgpt"):
            base_url = global_params.get("chatgpt_api", "")
        elif provider_id == "deepseek":
            base_url = global_params.get("deepseek_api", "")
        elif provider_id == "gemini":
            base_url = global_params.get("gemini_api", "")

    proxy = str((payload.proxy if payload else None) or global_settings.proxy or "").strip()

    req_app = request.app if request is not None else None
    fetcher_fn = (getattr(req_app.state, "provider_model_fetcher", None) if req_app else None) or fetch_remote_provider_models
    ok, models, msg = await fetcher_fn(provider_id, api_key, base_url, proxy)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    if models:
        models_csv = ",".join(models)
        if provider_id in ("openai", "chatgpt"):
            global_settings["chatgpt_model"] = models_csv
            global_settings.save()
        elif provider_id == "deepseek":
            global_settings["deepseek_model"] = models_csv
            global_settings.save()
        elif provider_id == "gemini":
            global_settings["gemini_model"] = models_csv
            global_settings.save()
        elif provider_id == "deepgram":
            global_settings["deepgram_model"] = models_csv
            global_settings.save()
        elif provider_id == "elevenlabs":
            global_settings["elevenlabstts_models"] = models_csv
            global_settings.save()

    return JSONResponse({
        "ok": True,
        "providerId": provider_id,
        "models": models,
        "message": f"Successfully loaded {len(models)} models from {provider_id}",
    })


@router.get("/api/settings/storage")
async def get_storage_settings_handler() -> JSONResponse:
    return JSONResponse(storage_config.get_storage_metrics())


@router.put("/api/settings/storage")
async def update_storage_settings_handler(
    payload: StorageSettingsRequest,
    request: Request,
) -> JSONResponse:
    try:
        storage_config.update_storage_paths(payload.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse({
        "ok": True,
        "storage": storage_config.get_storage_metrics(),
        "message": "Storage paths updated successfully",
    })


@router.post("/api/settings/storage/clean-temp")
async def clean_temp_storage_handler() -> JSONResponse:
    res = storage_config.clean_temp_cache()
    return JSONResponse(res)


@router.put("/api/settings/general")
async def update_general_settings_handler(
    payload: GeneralSettingsRequest,
    request: Request,
) -> JSONResponse:
    if payload.proxy is not None:
        proxy_val = str(payload.proxy).strip()
        global_settings.proxy = proxy_val
        if proxy_val:
            os.environ["HTTP_PROXY"] = proxy_val
            os.environ["HTTPS_PROXY"] = proxy_val
        else:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)

    if payload.defaultSourceLanguage is not None:
        lang = str(payload.defaultSourceLanguage).strip()
        if lang:
            global_params["source_language"] = lang

    if payload.defaultTargetLanguage is not None:
        lang = str(payload.defaultTargetLanguage).strip()
        if lang:
            global_params["target_language"] = lang

    if payload.crf is not None:
        try:
            crf_val = int(payload.crf)
            if 0 <= crf_val <= 51:
                global_settings["crf"] = crf_val
        except (ValueError, TypeError):
            pass

    if payload.preset is not None:
        preset_val = str(payload.preset).strip()
        if preset_val:
            global_settings["preset"] = preset_val

    global_settings.save()
    global_params.save()

    return JSONResponse({
        "ok": True,
        "message": "General settings saved successfully",
    })


def register_routes(app) -> None:
    app.include_router(router)
