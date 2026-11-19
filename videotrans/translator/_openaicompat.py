# -*- coding: utf-8 -*-
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, TypeVar, Union

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from videotrans.configure.config import logger, params, settings, tr
from videotrans.configure.excepts import (
    NO_RETRY_EXCEPT,
    LLMSegmentError,
    StopRetry,
    StopTask,
    TranslateSrtError,
)
from videotrans.task.taskcfg import SrtItem
from videotrans.translator._base import BaseTrans
from videotrans.util._srt_parse import get_subtitle_from_srt, ms_to_time_string
from videotrans.util.help_misc import get_prompt, get_tanslate_type
from videotrans.util.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_prompt_parts,
    clean_ai_response,
    filter_glossary_for_text,
    format_glossary_block,
    load_glossary_entries,
    load_resegment_template,
    render_user_prompt,
    split_prompt_template,
)

_ORIGINAL_GET_PROMPT = get_prompt
_T = TypeVar("_T")


@dataclass
class OpenAICampat(BaseTrans):
    ainame: Optional[str] = None
    prompt: str = field(init=False, default="")
    system_prompt: str = field(init=False, default=DEFAULT_SYSTEM_PROMPT)
    user_prompt_template: str = field(init=False, default="{batch_input}")
    api_key: str = field(init=False, default="")
    api_url: str = field(init=False, default="")
    temperature: float = 1.0
    max_tokens: int = 8192
    reasoning_effort: Optional[str] = None
    extra_body: Union[dict, None] = None

    _client_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )
    _clients: dict[tuple[str, str, str], OpenAI] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _http_clients: dict[tuple[str, str, str], httpx.Client] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _thread_context: threading.local = field(
        default_factory=threading.local, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        super().__post_init__()
        self.temperature = float(settings.get("aitrans_temperature", 1.0))
        self._client_lock = threading.Lock()
        self._clients = {}
        self._http_clients = {}
        self._thread_context = threading.local()
        self._key_lock = threading.Lock()
        self._key_index = 0

        if self.ainame:
            if get_prompt is not _ORIGINAL_GET_PROMPT:
                raw_prompt = get_prompt(
                    ainame=self.ainame, aisendsrt=self.aisendsrt
                ).replace("{lang}", self.target_language_name)
                self.prompt = raw_prompt
                self.system_prompt, self.user_prompt_template = split_prompt_template(
                    raw_prompt
                )
            else:
                self.system_prompt, self.user_prompt_template = build_prompt_parts(
                    self.ainame,
                    aisendsrt=self.aisendsrt,
                    lang=self.target_language_name,
                )
                self.prompt = f"{self.system_prompt}\n\n{self.user_prompt_template}"
        else:
            self.prompt = ""
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
            self.user_prompt_template = "{batch_input}"

        try:
            self.max_tokens = int(self.max_tokens)
        except (ValueError, TypeError) as e:
            logger.error(
                f"当前渠道{self.ainame}设置的最大输出tokens错误，应填写整数，实际填写的是`{self.max_tokens}`\n{e}"
            )
            self.max_tokens = 8192

    def _get_client(self, api_key: str, api_url: str) -> OpenAI:
        proxy = (self.proxy_str or "").strip() or None
        key = (api_key or "", api_url or "", proxy or "")
        with self._client_lock:
            existing = self._clients.get(key)
            if existing is not None:
                return existing
            http_client = httpx.Client(
                proxy=proxy,
                limits=httpx.Limits(
                    max_connections=32,
                    max_keepalive_connections=16,
                    keepalive_expiry=30.0,
                ),
                timeout=httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=15.0),
            )
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url=api_url,
                    http_client=http_client,
                    max_retries=0,
                )
            except TypeError:
                http_client.close()
                client = OpenAI(api_key=api_key, base_url=api_url)
                http_client = None
            self._clients[key] = client
            if http_client is not None:
                self._http_clients[key] = http_client
            return client

    def _get_api_keys(self) -> list[str]:
        raw = self.api_key or ""
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        return keys or ([raw.strip()] if raw.strip() else [])

    def _next_api_key(self) -> str:
        with self._key_lock:
            keys = self._get_api_keys()
            if not keys:
                return self.api_key or ""
            key = keys[self._key_index % len(keys)]
            self._key_index += 1
            return key

    def _close_clients(self) -> None:
        with self._client_lock:
            for hc in self._http_clients.values():
                try:
                    hc.close()
                except Exception:
                    pass
            for client in self._clients.values():
                close_fn = getattr(client, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
            self._http_clients.clear()
            self._clients.clear()

    def _unload(self) -> None:
        self._close_clients()

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

    def _build_messages(
        self,
        text: str,
        context_lines: Optional[List[str]] = None,
    ) -> list[dict[str, str]]:
        entries = load_glossary_entries()
        matched = filter_glossary_for_text(entries, text)
        glossary_block = format_glossary_block(matched)
        ctx_text = (
            "\n".join(str(ln).strip() for ln in context_lines if str(ln).strip())
            if context_lines
            else ""
        )
        user_content = render_user_prompt(
            self.user_prompt_template,
            batch_input=text,
            glossary_block=glossary_block,
            context_text=ctx_text,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _extract_retry_after(exc: Exception) -> Optional[float]:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None)
        if not headers:
            return None
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw is None:
            return None
        try:
            return max(0.5, min(60.0, float(raw)))
        except (TypeError, ValueError):
            return None

    def _call_with_retry(
        self,
        fn: Callable[[], _T],
        *,
        api_url: Optional[str] = None,
        req_meta: Optional[dict[str, Any]] = None,
        retryable_extra: tuple[type[BaseException], ...] = (),
    ) -> Optional[_T]:
        raw_retry = int(settings.get("retry_nums", 3) or 3)
        max_retries = 3 if raw_retry <= 1 else raw_retry
        total_attempts = 1 + max_retries
        target_url = api_url if api_url is not None else self.api_url
        safe_meta = dict(req_meta or {})
        safe_meta.pop("messages", None)

        last_err: Optional[BaseException] = None
        for attempt in range(total_attempts):
            if self._exit():
                return None
            try:
                return fn()
            except (StopTask, StopRetry, LengthFinishReasonError) as e:
                raise
            except (
                NotFoundError,
                AuthenticationError,
                PermissionDeniedError,
                BadRequestError,
            ) as e:
                body = getattr(e, "body", None)
                msg = (
                    body.get("message")
                    if isinstance(body, dict) and body.get("message")
                    else getattr(e, "message", str(e))
                )
                raise StopTask(f"{msg}\n{target_url}\n{safe_meta}") from e
            except (
                APITimeoutError,
                httpx.TimeoutException,
                httpx.RemoteProtocolError,
                TimeoutError,
            ) as e:
                last_err = e
            except APIConnectionError as e:
                cause = getattr(e, "__cause__", None)
                if isinstance(
                    cause, (httpx.TimeoutException, httpx.RemoteProtocolError, TimeoutError)
                ):
                    last_err = e
                else:
                    raise StopTask(
                        f'[{self.ainame}] {tr("Unable to connect to API", target_url)}\n{getattr(e, "message", e)}'
                    ) from e
            except (RateLimitError, InternalServerError) as e:
                last_err = e
            except APIStatusError as e:
                msg = str(getattr(e, "message", e) or e)
                if re.search(r"insufficient.*?balance", msg, flags=re.I):
                    raise StopTask(
                        tr(
                            "The server returned an error message: Insufficient balance",
                            get_tanslate_type(self.translate_type),
                            target_url,
                        )
                    ) from e
                status_code = getattr(e, "status_code", None)
                msg_lower = msg.lower()
                is_quota = any(term in msg_lower for term in ("quota", "rate limit", "rate_limit", "ratelimit", "too many requests", "429"))
                if (
                    status_code in (408, 409, 429)
                    or (status_code is not None and status_code >= 500)
                    or is_quota
                ):
                    last_err = e
                else:
                    raise StopTask(f"{msg}\n{target_url}\n{safe_meta}") from e
            except APIError as e:
                msg = str(getattr(e, "message", e) or e)
                if re.search(r"insufficient.*?balance", msg, flags=re.I):
                    raise StopTask(
                        tr(
                            "The server returned an error message: Insufficient balance",
                            get_tanslate_type(self.translate_type),
                            target_url,
                        )
                    ) from e
                last_err = e
            except NO_RETRY_EXCEPT:
                raise
            except (TranslateSrtError, *retryable_extra) as e:
                last_err = e

            if attempt < max_retries:
                delay = self._extract_retry_after(last_err)
                if delay is None:
                    delay = min(30.0, 1.0 * (2**attempt)) * random.uniform(0.5, 1.5)
                logger.warning(
                    f"[{self.ainame}] attempt {attempt + 1}/{total_attempts} failed ({type(last_err).__name__}: {last_err}), retrying in {delay:.2f}s"
                )
                if self._sleep_interruptible(delay):
                    return None

        if last_err is not None:
            raise last_err
        return None

    def _item_task(
        self,
        data: Union[List[str], str],
        context_lines: Optional[List[str]] = None,
    ) -> Optional[str]:
        if self._exit():
            return None
        if len(self.api_url) < 10:
            raise StopTask(f"API URL is error: {self.api_url}")

        if context_lines is None:
            context_lines = getattr(self._thread_context, "lines", None)

        text = "\n".join(i.strip() for i in data) if isinstance(data, list) else data
        messages = self._build_messages(text, context_lines=context_lines)

        base_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "timeout": 300,
            "temperature": float(self.temperature),
        }
        if "api.openai.com" in self.api_url or (
            self.ainame == "chatgpt"
            and re.match(r"^(gpt|o\d)", self.model_name, flags=re.I)
        ):
            base_kwargs["max_completion_tokens"] = int(self.max_tokens)
        else:
            base_kwargs["max_tokens"] = int(self.max_tokens)

        if self.reasoning_effort:
            base_kwargs["reasoning_effort"] = self.reasoning_effort

        logger.debug(f"字幕翻译:[{self.ainame=},{base_kwargs=}]")

        def _invoke() -> str:
            current_key = self._next_api_key() if self.api_key else ""
            client = self._get_client(api_key=current_key, api_url=self.api_url)
            req_kwargs = {**base_kwargs, "messages": messages}
            response = client.chat.completions.create(
                **req_kwargs, extra_body=self.extra_body
            )
            if not hasattr(response, "choices") or not response.choices:
                raise TranslateSrtError(str(response))
            if response.choices[0].finish_reason == "length":
                raise LengthFinishReasonError(completion=response)
            if not response.choices[0].message.content:
                logger.warning(f"[{self.ainame}]请求失败:{response=}")
                raise TranslateSrtError(
                    f"[{self.ainame}] {self.api_url} {response.choices[0].finish_reason}:{response}"
                )
            return response.choices[0].message.content

        raw_content = self._call_with_retry(
            _invoke, api_url=self.api_url, req_meta=base_kwargs
        )
        if raw_content is None:
            return None
        return clean_ai_response(raw_content, tag="TRANSLATE_TEXT")

    def llm_segment(self, srt_list, step="") -> List[SrtItem]:
        _st = time.time()
        if self.ainame != "deepseek":
            api_url = (
                params.get("chatgpt_api") or "https://api.openai.com/v1"
            ).strip()
        else:
            api_url = (
                params.get("deepseek_api") or "https://api.deepseek.com/v1"
            ).strip()
        if len(api_url) < 10:
            raise StopTask(f"API URL is error: {api_url}")

        prompts_template = load_resegment_template(step)
        prompts_template = prompts_template.replace(
            "{max_speech_s}",
            str(int(int(settings.get("max_speech_duration_s", 6)) * 1.7)),
        )
        chunk_size = max(1, int(settings.get("llm_chunk_size", 20)))
        model_name = params.get(f"{self.ainame}_model") or ""
        max_tokens = max(
            8192, int(float(params.get(f"{self.ainame}_max_token", 8192)))
        )
        temperature = float(settings.get("aitrans_temperature", 1.0))
        api_key = params.get(f"{self.ainame}_key") or ""
        reasoning_effort = "high" if self.ainame == "deepseek" else None

        if reasoning_effort is None:
            _reason = params.get("chatgpt_reasoning_effort")
            reasoning_effort = None if not _reason or _reason == "No" else _reason

        base_kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "timeout": 600,
        }
        if reasoning_effort:
            base_kwargs["reasoning_effort"] = reasoning_effort
        if "api.openai.com" in api_url or (
            self.ainame == "chatgpt"
            and re.match(r"^(gpt|o\d)", model_name, flags=re.I)
        ):
            base_kwargs["max_completion_tokens"] = int(max_tokens)
        else:
            base_kwargs["max_tokens"] = int(max_tokens)
        logger.debug(f"LLM Re-segmenting:{self.ainame=},{base_kwargs=}")

        def _send(srt: str) -> str:
            message = [
                {"role": "system", "content": prompts_template},
                {"role": "user", "content": f"```srt\n{srt}\n```"},
            ]

            def _invoke() -> str:
                client = self._get_client(api_key=api_key, api_url=api_url)
                req_kwargs = {**base_kwargs, "messages": message}
                response = client.chat.completions.create(
                    **req_kwargs,
                    extra_body={"thinking": {"type": "enabled"}}
                    if self.ainame == "deepseek"
                    else None,
                )
                if not response or not hasattr(response, "choices") or not response.choices:
                    logger.warning(f"[{self.ainame}]重新断句失败:{response=}")
                    raise LLMSegmentError(f"[{self.ainame}]{response}")
                if response.choices[0].finish_reason == "length":
                    raise LLMSegmentError(f"[{self.ainame}] Please increase max_token")
                if not response.choices[0].message.content:
                    logger.warning(f"[{self.ainame}]重新断句失败:{response=}")
                    raise LLMSegmentError(f"[{self.ainame}] {response}")
                return response.choices[0].message.content

            raw = self._call_with_retry(
                _invoke,
                api_url=api_url,
                req_meta=base_kwargs,
                retryable_extra=(LLMSegmentError,),
            )
            if raw is None:
                return ""
            return clean_ai_response(raw, tag="SRT")

        logger.debug(f"LLM断句:{self.ainame=},{model_name=},{api_url=}")
        new_sublist: list[str] = []
        try:
            for idx in range(0, len(srt_list), chunk_size):
                if self._exit():
                    return []
                self.signal(
                    text=f"[{idx}] {self.ainame} " + tr("Re-segmenting...")
                )
                srt_str = "\n\n".join(
                    [
                        f"{line + 1}\n{it['time']}\n{it['text']}"
                        for line, it in enumerate(srt_list[idx : idx + chunk_size])
                    ]
                )
                seg_out = _send(srt_str)
                if self._exit():
                    return []
                if seg_out:
                    new_sublist.append(seg_out)
        finally:
            self._close_clients()

        _srtlist = get_subtitle_from_srt("\n\n".join(new_sublist), is_file=False)
        _len = len(_srtlist)
        for i, it in enumerate(_srtlist):
            _had_edit = False
            if i < _len - 1 and it["end_time"] > _srtlist[i + 1]["start_time"]:
                it["end_time"] = _srtlist[i + 1]["start_time"]
                _had_edit = True
            if it["start_time"] > it["end_time"]:
                it["start_time"] = it["end_time"]
                _had_edit = True
            if _had_edit:
                it["startraw"] = ms_to_time_string(ms=it["start_time"])
                it["endraw"] = ms_to_time_string(ms=it["end_time"])
                it["time"] = f"{it['startraw']} --> {it['endraw']}"
        logger.debug(
            f'{"【二次识别后】" if step else ""}LLM重新断句完成,原始字幕行:{len(srt_list)}, 新字幕行:{len(_srtlist)}, 用时:{time.time()-_st}s'
        )
        return _srtlist
