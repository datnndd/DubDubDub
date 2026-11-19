import random
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import httpx
from google import genai
from google.genai import errors, types

from videotrans.configure.config import logger, params, settings
from videotrans.configure.excepts import (
    NO_RETRY_EXCEPT,
    StopRetry,
    StopTask,
    TranslateSrtError,
)
from videotrans.translator._base import BaseTrans
from videotrans.util.help_misc import get_prompt
from videotrans.util.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_prompt_parts,
    clean_ai_response,
    filter_glossary_for_text,
    format_glossary_block,
    load_glossary_entries,
    render_user_prompt,
    split_prompt_template,
)

_ORIGINAL_GET_PROMPT = get_prompt


@dataclass
class Gemini(BaseTrans):
    prompt: str = field(init=False, default="")
    system_prompt: str = field(init=False, default=DEFAULT_SYSTEM_PROMPT)
    user_prompt_template: str = field(init=False, default="{batch_input}")
    api_keys: List[str] = field(init=False, repr=False, default_factory=list)

    _key_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )
    _key_index: int = field(default=0, init=False, repr=False, compare=False)
    _client_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )
    _clients: dict[tuple[str, str, str], genai.Client] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _thread_context: threading.local = field(
        default_factory=threading.local, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        super().__post_init__()
        raw_model = params.get("gemini_model", "gemini-2.0-flash") or "gemini-2.0-flash"
        if raw_model.startswith(("gemini-3.6", "gemini-3.5")):
            raw_model = "gemini-2.0-flash"
        self.model_name = raw_model
        self.api_url = (params.get("gemini_api", "") or "").strip()
        self.api_keys = [
            k.strip()
            for k in (params.get("gemini_key", "") or "").split(",")
            if k.strip()
        ]
        self._key_lock = threading.Lock()
        self._key_index = 0
        self._client_lock = threading.Lock()
        self._clients = {}
        self._thread_context = threading.local()

        if get_prompt is not _ORIGINAL_GET_PROMPT:
            raw_prompt = get_prompt(
                ainame="gemini", aisendsrt=self.aisendsrt
            ).replace("{lang}", self.target_language_name)
            self.prompt = raw_prompt
            self.system_prompt, self.user_prompt_template = split_prompt_template(
                raw_prompt
            )
        else:
            self.system_prompt, self.user_prompt_template = build_prompt_parts(
                "gemini",
                aisendsrt=self.aisendsrt,
                lang=self.target_language_name,
            )
            self.prompt = f"{self.system_prompt}\n\n{self.user_prompt_template}"

    def _next_api_key(self) -> str:
        with self._key_lock:
            if not self.api_keys:
                raise StopTask("Gemini API key is not configured.")
            key = self.api_keys[self._key_index % len(self.api_keys)]
            self._key_index += 1
            return key

    def _get_client(self, api_key: str) -> genai.Client:
        proxy = (self.proxy_str or "").strip() or None
        base_url = (self.api_url or "").strip() or None
        cache_key = (api_key, base_url or "", proxy or "")
        with self._client_lock:
            existing = self._clients.get(cache_key)
            if existing is not None:
                return existing
            http_kwargs: dict[str, Any] = {
                "base_url": base_url,
                "timeout": 120_000,
            }
            if proxy is not None:
                http_kwargs["client_args"] = {"proxy": proxy}
                http_kwargs["async_client_args"] = {"proxy": proxy}
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(**http_kwargs),
            )
            self._clients[cache_key] = client
            return client

    def _unload(self) -> None:
        with self._client_lock:
            for client in self._clients.values():
                close_fn = getattr(client, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
            self._clients.clear()

    def _close_clients(self) -> None:
        self._unload()

    def _translate_batch(
        self,
        data: Union[List[str], str],
        context_lines: Optional[List[str]] = None,
    ) -> Optional[str]:
        self._thread_context.lines = context_lines
        try:
            return self._item_task(data)
        finally:
            self._thread_context.lines = None

    def _build_user_message(
        self, text: str, context_lines: Optional[List[str]] = None
    ) -> str:
        entries = load_glossary_entries()
        matched = filter_glossary_for_text(entries, text)
        glossary_block = format_glossary_block(matched)
        ctx_text = (
            "\n".join(str(ln).strip() for ln in context_lines if str(ln).strip())
            if context_lines
            else ""
        )
        return render_user_prompt(
            self.user_prompt_template,
            batch_input=text,
            glossary_block=glossary_block,
            context_text=ctx_text,
        )

    def _build_config(self, model: str) -> types.GenerateContentConfig:
        model_lower = model.lower()
        config_kwargs: dict[str, Any] = {
            "temperature": float(settings.get("aitrans_temperature", 0.1)),
            "max_output_tokens": int(params.get("gemini_maxtoken", 65530)),
            "safety_settings": [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
            "system_instruction": [
                types.Part.from_text(text=self.system_prompt),
            ],
        }
        if not (model_lower.startswith("gemini-1.") or model_lower.startswith("gemini-2.0")):
            if model_lower.startswith("gemini-2"):
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=24576
                )
            else:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level="HIGH"
                )
        return types.GenerateContentConfig(**config_kwargs)

    def _item_task(
        self,
        data: Union[List[str], str],
        context_lines: Optional[List[str]] = None,
    ) -> Optional[str]:
        if self._exit():
            return None

        if context_lines is None:
            context_lines = getattr(self._thread_context, "lines", None)

        text = "\n".join(i.strip() for i in data) if isinstance(data, list) else data
        model = (
            self.model_name or params.get("gemini_model", "gemini-2.5-flash")
        ).strip()
        user_message = self._build_user_message(text, context_lines=context_lines)
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)],
            ),
        ]
        generate_content_config = self._build_config(model)

        raw_retry = int(settings.get("retry_nums", 3) or 3)
        max_retries = 3 if raw_retry <= 1 else raw_retry
        total_attempts = 1 + max_retries

        last_err: Optional[BaseException] = None
        for attempt in range(total_attempts):
            if self._exit():
                return None
            api_key = self._next_api_key()
            client = self._get_client(api_key)
            try:
                result = ""
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if self._exit():
                        return None
                    result += chunk.text if chunk.text else ""

                logger.debug(f"{result=}")
                cleaned = clean_ai_response(result, tag="TRANSLATE_TEXT")
                if not cleaned:
                    logger.warning("[gemini]请求失败: empty response")
                    raise TranslateSrtError("[Gemini] result is empty")
                return cleaned
            except (StopTask, StopRetry):
                raise
            except errors.APIError as e:
                logger.warning(f"{e=}")
                code = getattr(e, "code", None)
                msg = getattr(e, "message", None) or str(e)
                msg_lower = msg.lower()
                is_quota = any(term in msg_lower for term in ("resource_exhausted", "quota", "rate limit", "rate_limit", "ratelimit", "too many requests", "429"))
                if (
                    code in (408, 429)
                    or (code is not None and 500 <= code < 600)
                    or isinstance(e, errors.ServerError)
                    or is_quota
                ):
                    last_err = e
                elif code in (400, 401, 403, 404) or (
                    code is not None and 400 <= code < 500
                ):
                    if is_quota:
                        last_err = e
                    else:
                        raise StopTask(msg) from e
                else:
                    last_err = e
            except (
                httpx.TimeoutException,
                httpx.RemoteProtocolError,
                TimeoutError,
                TranslateSrtError,
            ) as e:
                last_err = e
            except NO_RETRY_EXCEPT:
                raise

            if attempt < max_retries:
                delay = min(30.0, 1.0 * (2**attempt)) * random.uniform(0.5, 1.5)
                logger.warning(
                    f"[Gemini] attempt {attempt + 1}/{total_attempts} failed ({type(last_err).__name__}: {last_err}), retrying in {delay:.2f}s"
                )
                if self._sleep_interruptible(delay):
                    return None

        if isinstance(last_err, TranslateSrtError):
            raise last_err
        raise TranslateSrtError(str(last_err) if last_err else "[Gemini] unknown error") from last_err
