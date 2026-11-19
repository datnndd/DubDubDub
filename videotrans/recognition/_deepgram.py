import os
import re
import time
from dataclasses import dataclass
from typing import Any, List, Union
from pathlib import Path
import json
import httpx
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from deepgram.clients.common.v1.errors import DeepgramApiError
from deepgram_captions import DeepgramConverter, srt
from videotrans.configure import config, contants
from videotrans.configure.config import tr, params, settings, logger
from videotrans.recognition._base import BaseRecogn
from videotrans.recognition._retry import (
    build_httpx_retry_types,
    extract_status_code,
    is_retryable_status,
    is_retryable_error,
    run_with_retry,
)
from videotrans.task.taskcfg import SrtItem
from videotrans.util._ffmpeg_runner import runffmpeg
from videotrans.util._srt_parse import ms_to_time_string, get_subtitle_from_srt


_RETRYABLE_DEEPGRAM_NET = build_httpx_retry_types()


def _should_retry_deepgram(exc: BaseException) -> bool:
    if isinstance(exc, DeepgramApiError):
        return is_retryable_error(exc)
    return is_retryable_error(exc) or isinstance(exc, _RETRYABLE_DEEPGRAM_NET)


def parse_deepgram_options(raw: Any) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw or not isinstance(raw, str):
        return {}
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except Exception:
            pass
    parsed = {}
    for part in raw.replace("\n", "&").split("&"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.lower() == "true":
            parsed[k] = True
        elif v.lower() == "false":
            parsed[k] = False
        else:
            try:
                if "." in v:
                    parsed[k] = float(v)
                else:
                    parsed[k] = int(v)
            except ValueError:
                parsed[k] = v
    return parsed


@dataclass
class DeepgramRecogn(BaseRecogn):

    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit():
            return

        audio_path = self.audio_file
        tmp_mp3 = None
        try:
            if os.path.getsize(audio_path) > 52428800:
                tmp_dir = Path(self.cache_folder) if self.cache_folder else Path(config.TEMP_DIR)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_mp3 = tmp_dir / f"deepgram-tmp-{time.time_ns()}.mp3"
                runffmpeg(['-y', '-i', audio_path, '-ac', '1', '-ar', '16000', str(tmp_mp3)])
                audio_path = str(tmp_mp3)
            with open(audio_path, "rb") as file:
                buffer_data = file.read()
        finally:
            if tmp_mp3 is not None:
                try:
                    tmp_mp3.unlink(missing_ok=True)
                except OSError:
                    pass

        self.signal(
            text=tr("Recognition may take a while, please be patient"))

        payload: FileSource = {
            "buffer": buffer_data,
        }

        raw_lang = (self.detect_language or "").strip().lower()
        diarize = self.max_speakers > -1
        preset_options: dict[str, Any] = {
            "model": self.model_name or "nova-3",
            "smart_format": True,
            "punctuate": True,
            "paragraphs": True,
            "utterances": True,
            "diarize_model": "latest",
            "utt_split": 0.8,
        }
        if raw_lang == "auto":
            preset_options["detect_language"] = True
        else:
            preset_options["language"] = raw_lang[:2] if raw_lang and raw_lang != "-" else "zh"

        user_raw = (
            getattr(self, "deepgram_options", None)
            or params.get("deepgram_options")
            or settings.get("deepgram_options")
            or {}
        )
        user_opts = parse_deepgram_options(user_raw)
        if isinstance(user_opts, dict):
            extra = user_opts.pop("extra", None)
            if extra:
                user_opts.update(parse_deepgram_options(extra))
            for k, v in user_opts.items():
                if v is not None and v != "":
                    preset_options[k] = v
            if raw_lang == "auto" and user_opts.get("language"):
                preset_options.pop("detect_language", None)

        lang_prefix = str(preset_options.get("language") or "").strip().lower()[:2]

        if diarize:
            preset_options["utterances"] = True
            if not preset_options.get("diarize_model"):
                preset_options["diarize_model"] = "latest"

        # Deepgram API rule: diarize_model cannot be used together with diarize or diarize_version
        if preset_options.get("diarize_model"):
            preset_options.pop("diarize", None)
            preset_options.pop("diarize_version", None)

        valid_fields = set(getattr(PrerecordedOptions, "__dataclass_fields__", {}).keys())
        standard_opts = {k: v for k, v in preset_options.items() if k in valid_fields}
        extra_opts = {k: v for k, v in preset_options.items() if k not in valid_fields}

        options = PrerecordedOptions(**standard_opts)
        for k, v in extra_opts.items():
            setattr(options, k, v)

        orig_to_dict = getattr(options, "to_dict", None)
        if callable(orig_to_dict):
            def _to_dict(encode_json=False):
                d = {**orig_to_dict(encode_json=encode_json), **extra_opts}
                if d.get("diarize_model"):
                    d.pop("diarize", None)
                    d.pop("diarize_version", None)
                return d
            options.to_dict = _to_dict

        deepgram = DeepgramClient(params.get('deepgram_apikey'))
        transport = httpx.HTTPTransport(proxy=self.proxy_str) if self.proxy_str else None
        request_started = time.perf_counter()
        logger.info(
            "[Deepgram] prerecorded request start model=%s language=%s diarize=%s payload=%.2f MiB",
            self.model_name,
            lang_prefix or ("auto" if raw_lang == "auto" else "zh"),
            diarize,
            len(buffer_data) / (1024 * 1024),
        )
        try:
            def _do_transcribe():
                rest_client = deepgram.listen.rest.v("1")
                if transport is not None:
                    try:
                        return rest_client.transcribe_file(payload, options, timeout=600, transport=transport)
                    except TypeError:
                        return rest_client.transcribe_file(payload, options, timeout=600)
                return rest_client.transcribe_file(payload, options, timeout=600)

            res = run_with_retry(
                _do_transcribe,
                should_retry=_should_retry_deepgram,
                is_cancelled=self._exit,
                label="DeepgramRecogn",
            )
        finally:
            if transport is not None and callable(getattr(transport, "close", None)):
                try:
                    transport.close()
                except Exception:
                    pass

        request_duration = round(time.perf_counter() - request_started, 2)
        logger.info(
            "[Deepgram] prerecorded request completed in %.2fs",
            request_duration,
        )
        self.signal(
            text=f"Deepgram ASR completed in {request_duration}s",
            type="asr_timing",
            duration=request_duration,
        )

        if not lang_prefix:
            detected = None
            try:
                ch0 = res.results.channels[0]
                detected = getattr(ch0, "detected_language", None)
            except Exception:
                pass
            if not isinstance(detected, str):
                try:
                    detected = res["results"]["channels"][0].get("detected_language")
                except Exception:
                    detected = None
            if isinstance(detected, str) and detected.strip():
                lang_prefix = detected.strip().lower()[:2]

        zhconv = None
        if lang_prefix == 'zh':
            import zhconv as _zhconv
            zhconv = _zhconv

        # 1. Extract utterances option
        raw_utterances = []
        try:
            if hasattr(res, "results") and hasattr(res.results, "utterances") and res.results.utterances:
                raw_utterances = res.results.utterances
            elif hasattr(res, "__getitem__"):
                try:
                    raw_utterances = res["results"]["utterances"] or []
                except Exception:
                    pass
            if not raw_utterances and isinstance(res, dict):
                raw_utterances = res.get("results", {}).get("utterances", [])
        except Exception as e:
            logger.debug(f"Error accessing utterances from Deepgram response: {e}")

        utterances_list = []
        speaker_list = []
        for it in raw_utterances or []:
            text = (
                getattr(it, "transcript", None)
                if hasattr(it, "transcript")
                else it.get("transcript", "") if isinstance(it, dict) else ""
            )
            if not text or not str(text).strip():
                continue
            start_sec = getattr(it, "start", 0) if hasattr(it, "start") else (it.get("start", 0) if isinstance(it, dict) else 0)
            end_sec = getattr(it, "end", 0) if hasattr(it, "end") else (it.get("end", 0) if isinstance(it, dict) else 0)
            spk = getattr(it, "speaker", None) if hasattr(it, "speaker") else (it.get("speaker") if isinstance(it, dict) else None)

            spk_label = f"[spk{spk}]" if spk is not None else ""
            if spk_label:
                speaker_list.append(spk_label)

            start_ms = int(round(float(start_sec) * 1000))
            end_ms = int(round(float(end_sec) * 1000))
            clean_text = str(text).strip()
            if lang_prefix in contants.CJK_LANG:
                clean_text = re.sub(r'\s| ', '', clean_text, flags=re.I | re.S)
                if lang_prefix == 'zh' and zhconv is not None:
                    clean_text = zhconv.convert(clean_text, 'zh-hans')

            startraw = ms_to_time_string(ms=start_ms)
            endraw = ms_to_time_string(ms=end_ms)
            utterances_list.append({
                "line": len(utterances_list) + 1,
                "start_time": start_ms,
                "end_time": end_ms,
                "text": clean_text,
                "spk": spk_label,
                "startraw": startraw,
                "endraw": endraw,
                "time": f"{startraw} --> {endraw}",
            })

        # 2. Extract paragraphs option
        raw_paragraphs = []
        paragraphs_transcript = ""
        try:
            if hasattr(res, "results") and hasattr(res.results, "channels") and res.results.channels:
                ch = res.results.channels[0]
                if hasattr(ch, "alternatives") and ch.alternatives:
                    alt = ch.alternatives[0]
                    if hasattr(alt, "paragraphs") and alt.paragraphs:
                        pt = getattr(alt.paragraphs, "transcript", "")
                        paragraphs_transcript = pt if isinstance(pt, str) else ""
                        if hasattr(alt.paragraphs, "paragraphs") and isinstance(alt.paragraphs.paragraphs, list):
                            raw_paragraphs = alt.paragraphs.paragraphs
            if not raw_paragraphs and hasattr(res, "__getitem__"):
                try:
                    p_obj = res["results"]["channels"][0]["alternatives"][0]["paragraphs"]
                    pt = p_obj.get("transcript", "") if isinstance(p_obj, dict) else getattr(p_obj, "transcript", "")
                    if isinstance(pt, str) and not paragraphs_transcript:
                        paragraphs_transcript = pt
                    rp = p_obj.get("paragraphs", []) if isinstance(p_obj, dict) else getattr(p_obj, "paragraphs", [])
                    if isinstance(rp, list):
                        raw_paragraphs = rp
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Error accessing paragraphs from Deepgram response: {e}")

        paragraphs_list = []
        for para in raw_paragraphs or []:
            text = ""
            if hasattr(para, "transcript") and getattr(para, "transcript"):
                text = getattr(para, "transcript")
            elif isinstance(para, dict) and para.get("transcript"):
                text = para.get("transcript")
            elif hasattr(para, "text") and getattr(para, "text"):
                text = getattr(para, "text")
            elif isinstance(para, dict) and para.get("text"):
                text = para.get("text")
            elif hasattr(para, "sentences") and getattr(para, "sentences"):
                s_list = getattr(para, "sentences")
                text = " ".join((getattr(s, "text", "") if hasattr(s, "text") else s.get("text", "")).strip() for s in s_list if (getattr(s, "text", "") if hasattr(s, "text") else s.get("text", "")).strip())
            elif isinstance(para, dict) and para.get("sentences"):
                s_list = para.get("sentences", [])
                text = " ".join((s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")).strip() for s in s_list if (s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")).strip())

            if not text or not str(text).strip():
                continue

            start_sec = getattr(para, "start", 0) if hasattr(para, "start") else (para.get("start", 0) if isinstance(para, dict) else 0)
            end_sec = getattr(para, "end", 0) if hasattr(para, "end") else (para.get("end", 0) if isinstance(para, dict) else 0)
            if start_sec == 0 and end_sec == 0:
                sentences = getattr(para, "sentences", None) or (para.get("sentences", []) if isinstance(para, dict) else [])
                if sentences:
                    first_s = sentences[0]
                    last_s = sentences[-1]
                    start_sec = getattr(first_s, "start", 0) if hasattr(first_s, "start") else (first_s.get("start", 0) if isinstance(first_s, dict) else 0)
                    end_sec = getattr(last_s, "end", 0) if hasattr(last_s, "end") else (last_s.get("end", 0) if isinstance(last_s, dict) else 0)

            spk = getattr(para, "speaker", None) if hasattr(para, "speaker") else (para.get("speaker") if isinstance(para, dict) else None)
            spk_label = f"[spk{spk}]" if spk is not None else ""

            start_ms = int(round(float(start_sec) * 1000))
            end_ms = int(round(float(end_sec) * 1000))
            clean_text = str(text).strip()
            if lang_prefix in contants.CJK_LANG:
                clean_text = re.sub(r'\s| ', '', clean_text, flags=re.I | re.S)
                if lang_prefix == 'zh' and zhconv is not None:
                    clean_text = zhconv.convert(clean_text, 'zh-hans')

            startraw = ms_to_time_string(ms=start_ms)
            endraw = ms_to_time_string(ms=end_ms)
            paragraphs_list.append({
                "line": len(paragraphs_list) + 1,
                "start_time": start_ms,
                "end_time": end_ms,
                "text": clean_text,
                "spk": spk_label,
                "startraw": startraw,
                "endraw": endraw,
                "time": f"{startraw} --> {endraw}",
            })

        # Save speaker list if diarized
        if speaker_list and self.cache_folder:
            Path(self.cache_folder).mkdir(parents=True, exist_ok=True)
            Path(f'{self.cache_folder}/speaker.json').write_text(json.dumps(speaker_list), encoding='utf-8')

        # Persist both options to transcript_options.json for downstream choice
        transcript_options = {}
        if utterances_list:
            transcript_options["utterances"] = utterances_list
        if paragraphs_list:
            transcript_options["paragraphs"] = paragraphs_list
        elif paragraphs_transcript and utterances_list:
            # Fallback single paragraph covering the speech span
            clean_pt = paragraphs_transcript.strip()
            if lang_prefix in contants.CJK_LANG:
                clean_pt = re.sub(r'\s| ', '', clean_pt, flags=re.I | re.S)
                if lang_prefix == 'zh' and zhconv is not None:
                    clean_pt = zhconv.convert(clean_pt, 'zh-hans')
            transcript_options["paragraphs"] = [{
                "line": 1,
                "start_time": utterances_list[0]["start_time"],
                "end_time": utterances_list[-1]["end_time"],
                "text": clean_pt,
                "spk": utterances_list[0].get("spk", ""),
                "startraw": utterances_list[0]["startraw"],
                "endraw": utterances_list[-1]["endraw"],
                "time": f"{utterances_list[0]['startraw']} --> {utterances_list[-1]['endraw']}",
            }]

        if transcript_options:
            if self.cache_folder:
                Path(self.cache_folder).mkdir(parents=True, exist_ok=True)
                Path(f"{self.cache_folder}/transcript_options.json").write_text(
                    json.dumps(transcript_options, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
            self.transcript_options = transcript_options

        # Choose default raws
        segment_pref = (user_opts.get("segment_by") if isinstance(user_opts, dict) else None) or "utterances"
        if segment_pref == "paragraphs" and paragraphs_list:
            raws = paragraphs_list
        elif utterances_list:
            raws = utterances_list
        elif paragraphs_list:
            raws = paragraphs_list
        else:
            try:
                transcription = DeepgramConverter(res)
                srt_str = srt(transcription,
                              line_length=int(settings.get('cjk_len', 20) if lang_prefix in ['zh', 'ja', 'ko'] else settings.get('other_len', 60)))
                raws = get_subtitle_from_srt(srt_str, is_file=False)
                if lang_prefix in contants.CJK_LANG:
                    for i, it in enumerate(raws):
                        if lang_prefix == 'zh' and zhconv is not None:
                            it['text'] = zhconv.convert(it['text'], 'zh-hans')
                        raws[i]['text'] = it['text'].replace(' ', '')
            except Exception as e:
                logger.debug(f"DeepgramConverter fallback failed: {e}")
                raws = []

        return raws
