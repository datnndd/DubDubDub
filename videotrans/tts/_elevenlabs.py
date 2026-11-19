from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import random
import threading
import time
from typing import Any, Dict, List, Optional, Union

from elevenlabs import ElevenLabs, VoiceSettings, UnauthorizedError
import httpx

from videotrans.configure.config import params, logger, ROOT_DIR, settings
from videotrans.configure.excepts import StopTask
from videotrans.tts._base import BaseTTS
from videotrans.util.help_misc import vail_file


@lru_cache(maxsize=4)
def _load_elevenlabs_voice_map(json_path: str, mtime_ns: int, size: int) -> Dict[str, str]:
    try:
        raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {
                str(name): str(info.get("voice_id"))
                for name, info in raw.items()
                if isinstance(info, dict) and info.get("voice_id")
            }
    except Exception:
        pass
    return {}


@dataclass
class ElevenLabsC(BaseTTS):
    is_cloud_provider: bool = True
    _client: Any = field(default=None, init=False, repr=False)
    _http_client: Optional[httpx.Client] = field(default=None, init=False, repr=False)
    _client_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _voice_id_cache: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _voice_cache_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.speed = self.get_speed()

    def _get_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                api_key = params.get("elevenlabstts_key", "")
                proxy = (getattr(self, "proxy_str", None) or "").strip() or None
                try:
                    self._http_client = httpx.Client(
                        proxy=proxy,
                        timeout=httpx.Timeout(120.0, connect=30.0),
                    )
                    self._client = ElevenLabs(
                        api_key=api_key,
                        timeout=120.0,
                        httpx_client=self._http_client,
                    )
                except TypeError:
                    # Fallback if ElevenLabs is monkeypatched by a strict test stub
                    if self._http_client is not None:
                        try:
                            self._http_client.close()
                        except Exception:
                            pass
                        self._http_client = None
                    self._client = ElevenLabs(api_key=api_key)
            return self._client

    def _close_client(self) -> None:
        with self._client_lock:
            self._client = None
            if self._http_client is not None:
                try:
                    self._http_client.close()
                except Exception:
                    pass
                self._http_client = None

    def _resolve_voice_id(self, role: str) -> str:
        role_str = str(role or "").strip()
        with self._voice_cache_lock:
            cached = self._voice_id_cache.get(role_str)
            if cached is not None:
                return cached

        voice_id = None
        json_file = Path(ROOT_DIR) / "videotrans" / "voicejson" / "elevenlabs.json"
        try:
            st = json_file.stat()
            vmap = _load_elevenlabs_voice_map(str(json_file), st.st_mtime_ns, st.st_size)
            voice_id = vmap.get(role_str)
        except OSError:
            pass

        if not voice_id:
            try:
                from videotrans.core import voice_store
                from videotrans import tts
                v = None
                if role_str.startswith("voice_"):
                    v = voice_store.get_voice(role_str)
                if not v:
                    v = voice_store.find_voice_by_name(role_str, provider=tts.ELEVENLABS_TTS)
                if not v:
                    v = voice_store.get_voice(role_str)
                if v and v.get("external_voice_id"):
                    voice_id = v["external_voice_id"]
            except Exception:
                pass

        resolved = voice_id or role_str
        with self._voice_cache_lock:
            self._voice_id_cache[role_str] = resolved
        return resolved

    def _exec(self) -> None:
        try:
            super()._exec()
        finally:
            self._close_client()

    def _run(self, data_item: Union[Dict, List, None], idx: int = -1) -> Union[str, None]:
        if vail_file(data_item["filename"]):
            return
        voice_id = self._resolve_voice_id(data_item["role"])
        max_attempts = max(2, int(settings.get("retry_nums", 1) or 1))
        mp3_path = data_item["filename"] + ".mp3"
        last_exc: Optional[Exception] = None
        model_id = (params.get("elevenlabstts_models") or "eleven_multilingual_v2").strip()
        if model_id.startswith("eleven_v3"):
            model_id = "eleven_multilingual_v2"

        for attempt in range(max_attempts):
            if self._exit():
                return
            try:
                client = self._get_client()
                response = client.text_to_speech.convert(
                    text=data_item["text"],
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format="mp3_44100_128",
                    apply_text_normalization="auto",
                    voice_settings=VoiceSettings(
                        speed=self.speed,
                        stability=0.8,
                        similarity_boost=1,
                        style=0,
                        use_speaker_boost=False,
                    ),
                )
                with open(mp3_path, "wb") as f:
                    for chunk in response:
                        if chunk:
                            f.write(chunk)
                try:
                    self.convert_to_wav(mp3_path, data_item["filename"])
                finally:
                    Path(mp3_path).unlink(missing_ok=True)
                return
            except UnauthorizedError as e:
                Path(mp3_path).unlink(missing_ok=True)
                raise StopTask(getattr(e, "body", str(e)))
            except Exception as e:
                Path(mp3_path).unlink(missing_ok=True)
                status_code = getattr(e, "status_code", None)
                msg_lower = str(e).lower()
                is_quota = any(term in msg_lower for term in ("quota", "rate limit", "rate_limit", "ratelimit", "too many requests", "429"))
                if (
                    status_code in (408, 429)
                    or (status_code is not None and status_code >= 500)
                    or is_quota
                ):
                    last_exc = e
                elif status_code in (400, 401, 403, 404):
                    if is_quota:
                        last_exc = e
                    else:
                        raise StopTask(str(e))
                else:
                    last_exc = e
                if attempt + 1 < max_attempts and not self._exit():
                    delay = min(10.0, (2 ** attempt) * (0.5 + random.random() * 0.5))
                    logger.warning(
                        "ElevenLabs TTS transient failure (attempt %d/%d): %s; retrying in %.2fs",
                        attempt + 1,
                        max_attempts,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise

        if last_exc is not None:
            raise last_exc
