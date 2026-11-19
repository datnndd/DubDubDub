from dataclasses import dataclass
from typing import List, Union

import httpx
from elevenlabs import ElevenLabs
from elevenlabs.core import ApiError as ElevenLabsApiError
from videotrans.configure.config import params, logger
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


_RETRYABLE_ELEVENLABS_NET = build_httpx_retry_types()


def _should_retry_elevenlabs(exc: BaseException) -> bool:
    if isinstance(exc, ElevenLabsApiError):
        return is_retryable_error(exc)
    return is_retryable_error(exc) or isinstance(exc, _RETRYABLE_ELEVENLABS_NET)


@dataclass
class ElevenLabsRecogn(BaseRecogn):

    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit():
            return

        with open(self.audio_file, 'rb') as file:
            file_object = file.read()

        raw_lang = (self.detect_language or '').strip().lower()
        language_code = '' if raw_lang in ('', 'auto', '-') else raw_lang[:2]
        logger.debug(f'{language_code=}')

        model_id = self.model_name or "scribe_v2"
        call_kwargs = {
            "model_id": model_id,
            "file": file_object,
            "diarize": True,
        }
        if language_code:
            call_kwargs["language_code"] = language_code

        timeout_cfg = (
            httpx.Timeout(240.0, connect=30.0)
            if callable(getattr(httpx, "Timeout", None))
            else 240.0
        )
        with httpx.Client(proxy=self.proxy_str or None, timeout=timeout_cfg) as http_client:
            client = ElevenLabs(
                api_key=params.get('elevenlabstts_key', ''),
                timeout=240.0,
                httpx_client=http_client,
            )
            res = run_with_retry(
                lambda: client.speech_to_text.convert(**call_kwargs),
                should_retry=_should_retry_elevenlabs,
                is_cancelled=self._exit,
                label="ElevenLabsRecogn",
            )

        raws = []
        last_tmp = None
        logger.debug(f'elevenlabs{res=}\n')
        for it in res.words:
            if it.type == 'audio_event':
                continue
            text = it.text
            if not text: continue
            isflag = text[0] in self.flag or text[-1] in self.flag
            spk = (it.speaker_id or '').replace('speaker_', '')

            st = int(it.start * 1000)
            end = int(it.end * 1000)

            if not last_tmp:
                if not text.strip():
                    continue
                last_tmp = SrtItem(
                    line=len(raws) + 1,
                    text=text,
                    start_time=st,
                    end_time=end,
                    spk=spk
                )
                continue

            # If silence exceeds 200ms and sentence duration exceeds 500ms, and punctuation is present, split sentence
            diff_prev = st - last_tmp['end_time']
            segment_time = last_tmp['end_time'] - last_tmp['start_time']

            logger.debug(f'\n{text=},{isflag=},{spk=},{diff_prev=},{segment_time=}\n')

            # Different speaker, force sentence split

            if spk != last_tmp['spk']:
                last_tmp['time'] = ms_to_time_string(
                    ms=last_tmp['start_time']) + ' --> ' + ms_to_time_string(ms=last_tmp['end_time'])
                logger.debug(f'segments-spk:{last_tmp=}')
                if last_tmp['text'].strip():
                    raws.append(last_tmp)
                last_tmp = SrtItem(
                    line=len(raws) + 1,
                    text=text,
                    start_time=st,
                    end_time=end,
                    spk=spk
                )
                continue

            if (diff_prev >= 200 or segment_time >= 500) and isflag:
                # If punctuation is at start, assign word to next segment; otherwise assign to current
                if text[0] in self.flag:
                    last_tmp['text'] += text[0]
                    last_tmp['time'] = ms_to_time_string(
                        ms=last_tmp['start_time']) + ' --> ' + ms_to_time_string(ms=last_tmp['end_time'])
                    logger.debug(f'segments-flag0:{last_tmp=}')
                    if last_tmp['text'].strip():
                        raws.append(last_tmp)
                    last_tmp = SrtItem(
                        line=len(raws) + 1,
                        text=text[1:],
                        start_time=st,
                        end_time=end,
                        spk=spk
                    )
                    continue

                last_tmp['end_time'] = end
                last_tmp['text'] += text
                last_tmp['time'] = ms_to_time_string(
                    ms=last_tmp['start_time']) + ' --> ' + ms_to_time_string(ms=end)
                logger.debug(f'segments-flag1:{last_tmp=}')
                if last_tmp['text'].strip():
                    raws.append(last_tmp)
                last_tmp = None
            else:
                last_tmp['end_time'] = end
                last_tmp['text'] += text

        if last_tmp and last_tmp['text'].strip():
            last_tmp['time'] = ms_to_time_string(ms=last_tmp['start_time']) + ' --> ' + ms_to_time_string(
                ms=last_tmp['end_time'])
            logger.debug(f'segments:{last_tmp=}')

            raws.append(last_tmp)
        return raws
