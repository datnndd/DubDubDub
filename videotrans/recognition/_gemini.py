import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from google import genai
from google.genai import types, errors

from videotrans.configure.excepts import StopRetry, SpeechToTextError
from videotrans.configure.config import params, logger, ROOT_DIR, settings, tr
from videotrans.recognition._base import BaseRecogn
from videotrans.recognition._retry import (
    build_httpx_retry_types,
    extract_status_code,
    is_retryable_status,
    is_retryable_error,
    run_with_retry,
)
from videotrans.task.taskcfg import SrtItem
from videotrans.util._srt_parse import ms_to_time_string


_RETRYABLE_GEMINI_NET = build_httpx_retry_types()


def _should_retry_gemini(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        return is_retryable_error(exc)
    return is_retryable_error(exc) or isinstance(exc, _RETRYABLE_GEMINI_NET)


@dataclass
class GeminiRecogn(BaseRecogn):
    api_keys: List[str] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.api_keys = [k.strip() for k in (params.get('gemini_key', '') or '').split(',') if k.strip()]

    def _build_http_options(self):
        opts_kwargs = {'timeout': 180_000}
        if self.proxy_str:
            proxy_args = {'proxy': self.proxy_str}
            opts_kwargs['client_args'] = proxy_args
            opts_kwargs['async_client_args'] = proxy_args
        return types.HttpOptions(**opts_kwargs)

    def _req(self, seg_group, api_key, prompt, client=None):
        res_text = ""
        try:
            if client is None:
                client = genai.Client(
                    api_key=api_key,
                    http_options=self._build_http_options(),
                )
            parts = []
            for f in seg_group:
                parts.append(
                    types.Part.from_bytes(
                        mime_type="audio/wav",
                        data=Path(f['filename']).read_bytes()
                    )
                )
            parts.append(types.Part.from_text(text=prompt))

            logger.debug(f'发送音频到Gemini:prompt={prompt},{seg_group=}')
            generate_content_config = types.GenerateContentConfig(
                max_output_tokens=65536,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=1,
                ),
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",  # Block most
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",  # Block most
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",  # Block most
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",  # Block none
                    ),
                ],
            )
            contents = [
                types.Content(
                    role="user",
                    parts=parts
                )
            ]

            for chunk in client.models.generate_content_stream(
                    model='gemini-flash-latest',
                    contents=contents,
                    config=generate_content_config,
            ):
                if chunk.text is None:
                    continue
                res_text += chunk.text
            return res_text
        except errors.APIError as e:
            if is_retryable_error(e):
                raise
            raise StopRetry(getattr(e, 'message', None) or str(e)) from e

    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit():
            return
        keys = [k.strip() for k in (self.api_keys or []) if k and k.strip()]
        if not keys:
            keys = [k.strip() for k in (params.get('gemini_key', '') or '').split(',') if k.strip()]
        if not keys:
            raise SpeechToTextError(tr("Please configure the API information of the selected recognition provider first."))
        self.api_keys = keys

        clip_dir = None
        clients_by_key = {}
        try:
            seg_list = self.cut_audio()
            if seg_list:
                candidate = Path(seg_list[0]['filename']).parent
                if candidate.name.startswith("clip_"):
                    clip_dir = candidate
            if len(seg_list) < 1:
                raise SpeechToTextError('VAD error')
            nums = int(settings.get('gemini_recogn_chunk', 50))
            seg_list = [seg_list[i:i + nums] for i in range(0, len(seg_list), nums)]
            srt_str_list = []

            prompt = Path(ROOT_DIR + '/videotrans/prompts/recogn/gemini_recogn.txt').read_text(encoding='utf-8')
            # Save speakers
            speaker_list = []

            def _get_client(key: str):
                if key not in clients_by_key:
                    clients_by_key[key] = genai.Client(
                        api_key=key,
                        http_options=self._build_http_options(),
                    )
                return clients_by_key[key]

            for seg_group in seg_list:
                if self._exit():
                    return None

                def _call_chunk():
                    api_key = self.api_keys.pop(0)
                    self.api_keys.append(api_key)
                    return self._req(seg_group, api_key, prompt, client=_get_client(api_key))

                res_text = run_with_retry(
                    _call_chunk,
                    should_retry=_should_retry_gemini,
                    is_cancelled=self._exit,
                    label="GeminiRecogn",
                )
                logger.debug(f'gemini返回结果:{res_text=}')
                m = re.findall(r'<audio_text[^>]*?>(.*?)</audio_text>', res_text.strip(), re.I | re.S)
                if len(m) < 1:
                    continue
                str_s = []
                for i, f in enumerate(seg_group):
                    if i < len(m):
                        startraw = ms_to_time_string(ms=f['start_time'])
                        endraw = ms_to_time_string(ms=f['end_time'])
                        text = m[i].strip()
                        if not text:
                            continue
                        mt = re.match(r'^\[(spk\d+)]', text, re.I)

                        if mt:
                            speaker_list.append(mt.group(1))
                            text = re.sub(r'^\[spk\d+]', '', text, flags=re.I | re.S)
                        srt = SrtItem(
                            line=len(srt_str_list) + 1,
                            start_time=f['start_time'],
                            end_time=f['end_time'],
                            startraw=startraw,
                            endraw=endraw,
                            text=text
                        )
                        srt_str_list.append(srt)
                        str_s.append(f'{srt["line"]}\n{startraw} --> {endraw}\n{srt["text"]}')
                self.signal(
                    text=('\n\n'.join(str_s)) + "\n\n",
                    type='subtitle'
                )
                if self.asr_wait > 0:
                    time.sleep(self.asr_wait)

            if len(srt_str_list) < 1:
                raise SpeechToTextError('No result:The return format may not meet the requirements')
            if speaker_list and self.cache_folder:
                Path(self.cache_folder).mkdir(parents=True, exist_ok=True)
                Path(f'{self.cache_folder}/speaker.json').write_text(json.dumps(speaker_list), encoding='utf-8')
            return srt_str_list
        finally:
            for c in clients_by_key.values():
                close_fn = getattr(c, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
            if clip_dir is not None:
                shutil.rmtree(clip_dir, ignore_errors=True)
            elif getattr(self, "_last_cut_dir", None):
                shutil.rmtree(self._last_cut_dir, ignore_errors=True)


