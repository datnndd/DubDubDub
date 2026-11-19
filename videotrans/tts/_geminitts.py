from dataclasses import dataclass, field
import mimetypes
from pathlib import Path
import random
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

from google import genai
from google.genai import errors, types
import httpx

from videotrans.configure.config import params, logger, settings
from videotrans.configure.excepts import StopTask
from videotrans.tts._base import BaseTTS
from videotrans.util.help_misc import vail_file


@dataclass
class GEMINITTS(BaseTTS):
    is_cloud_provider: bool = True
    api_keys: List[str] = field(default_factory=list, init=False, repr=False)
    _key_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _key_index: int = field(default=0, init=False, repr=False)
    _client_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _clients: Dict[Tuple[str, str], genai.Client] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        raw_keys = str(params.get("gemini_key", "") or "")
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    def _next_api_key(self) -> str:
        with self._key_lock:
            if not self.api_keys:
                raw_keys = str(params.get("gemini_key", "") or "")
                self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
            if not self.api_keys:
                raise StopTask("Gemini API key is empty")
            key = self.api_keys[self._key_index % len(self.api_keys)]
            self._key_index += 1
            return key

    def _build_http_options(self) -> Optional[types.HttpOptions]:
        proxy = (getattr(self, "proxy_str", None) or "").strip()
        kwargs: Dict[str, object] = {"timeout": 120_000}
        if proxy:
            kwargs["client_args"] = {"proxy": proxy}
            kwargs["async_client_args"] = {"proxy": proxy}
        try:
            return types.HttpOptions(**kwargs)
        except Exception:
            return None

    def _get_client(self, api_key: str) -> genai.Client:
        proxy = (getattr(self, "proxy_str", None) or "").strip()
        cache_key = (api_key, proxy)
        with self._client_lock:
            client = self._clients.get(cache_key)
            if client is None:
                http_options = self._build_http_options()
                try:
                    if http_options is not None:
                        client = genai.Client(api_key=api_key, http_options=http_options)
                    else:
                        client = genai.Client(api_key=api_key)
                except TypeError:
                    client = genai.Client(api_key=api_key)
                self._clients[cache_key] = client
            return client

    def _close_clients(self) -> None:
        with self._client_lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for c in clients:
            close_fn = getattr(c, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    def _exec(self) -> None:
        try:
            super()._exec()
        finally:
            self._close_clients()

    def _run(self, data_item: Union[Dict, List, None], idx: int = -1) -> Union[str, None]:
        if vail_file(data_item["filename"]):
            return
        role = data_item["role"]
        tmp_wav = data_item["filename"] + ".wav"
        max_attempts = max(2, int(settings.get("retry_nums", 1) or 1))
        last_exc: Optional[Exception] = None
        raw_tts_model = (params.get("gemini_ttsmodel", "") or "gemini-2.5-flash-preview-tts").strip()
        if raw_tts_model.startswith("gemini-3.1"):
            raw_tts_model = "gemini-2.5-flash-preview-tts"

        for attempt in range(max_attempts):
            if self._exit():
                return
            api_key = self._next_api_key()
            try:
                self.generate_tts_segment(
                    data_item["text"],
                    role,
                    raw_tts_model,
                    tmp_wav,
                    api_key=api_key,
                )
                try:
                    self.convert_to_wav(tmp_wav, data_item["filename"])
                finally:
                    Path(tmp_wav).unlink(missing_ok=True)
                return
            except errors.APIError as e:
                Path(tmp_wav).unlink(missing_ok=True)
                code = getattr(e, "code", None)
                msg = getattr(e, "message", str(e)) or ""
                msg_lower = msg.lower()
                is_quota = any(term in msg_lower for term in ("resource_exhausted", "quota", "rate limit", "rate_limit", "ratelimit", "too many requests", "429"))
                if (
                    code in (408, 429)
                    or (code is not None and 500 <= code < 600)
                    or is_quota
                ):
                    last_exc = e
                elif code in (400, 401, 403, 404):
                    if is_quota:
                        last_exc = e
                    else:
                        raise StopTask(msg)
                else:
                    last_exc = e
                if attempt + 1 < max_attempts and not self._exit():
                    delay = min(10.0, (2 ** attempt) * (0.5 + random.random() * 0.5))
                    logger.warning(
                        "Gemini TTS APIError %s (attempt %d/%d); retrying in %.2fs",
                        code,
                        attempt + 1,
                        max_attempts,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise
            except (httpx.TimeoutException, httpx.RemoteProtocolError, httpx.TransportError) as e:
                Path(tmp_wav).unlink(missing_ok=True)
                last_exc = e
                if attempt + 1 < max_attempts and not self._exit():
                    delay = min(10.0, (2 ** attempt) * (0.5 + random.random() * 0.5))
                    time.sleep(delay)
                    continue
                raise
            except Exception:
                Path(tmp_wav).unlink(missing_ok=True)
                raise

        if last_exc is not None:
            raise last_exc

    def generate_tts_segment(self, text, voice, model, file_name, api_key: Optional[str] = None):
        def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
            parameters = parse_audio_mime_type(mime_type)
            bits_per_sample = parameters["bits_per_sample"]
            sample_rate = parameters["rate"]
            num_channels = 1
            data_size = len(audio_data)
            bytes_per_sample = bits_per_sample // 8
            block_align = num_channels * bytes_per_sample
            byte_rate = sample_rate * block_align
            chunk_size = 36 + data_size

            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                chunk_size,
                b"WAVE",
                b"fmt ",
                16,
                1,
                num_channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
                b"data",
                data_size,
            )
            return header + audio_data

        def parse_audio_mime_type(mime_type: str):
            bits_per_sample = 16
            rate = 24000
            parts = mime_type.split(";")
            for param in parts:
                param = param.strip()
                if param.lower().startswith("rate="):
                    try:
                        rate = int(param.split("=", 1)[1])
                    except (ValueError, IndexError, TypeError):
                        pass
                elif param.startswith("audio/L"):
                    try:
                        bits_per_sample = int(param.split("L", 1)[1])
                    except (ValueError, IndexError, TypeError):
                        pass
            return {"bits_per_sample": bits_per_sample, "rate": rate}

        def save_binary_file(out_file, data):
            with open(out_file, "wb") as f:
                f.write(data)

        key = api_key or self._next_api_key()
        client = self._get_client(key)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=text),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=[
                "audio",
            ],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                )
            ),
        )
        audio_chunks = []
        mime_type = None
        if hasattr(client.models, "generate_content_stream"):
            chunk_iter = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
        else:
            chunk_iter = [
                client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
            ]

        for chunk in chunk_iter:
            candidates = getattr(chunk, "candidates", None)
            if not candidates:
                continue
            cand0 = candidates[0]
            content = getattr(cand0, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not parts:
                continue
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    if getattr(inline_data, "mime_type", None):
                        mime_type = inline_data.mime_type
                    audio_chunks.append(inline_data.data)

        if not audio_chunks:
            raise RuntimeError("Gemini TTS returned no audio chunks")

        audio_data = b"".join(audio_chunks)
        file_extension = mimetypes.guess_extension(mime_type) if mime_type else None
        if file_extension is None:
            audio_data = convert_to_wav(audio_data, mime_type or "audio/L16;rate=24000")
        save_binary_file(file_name, audio_data)
