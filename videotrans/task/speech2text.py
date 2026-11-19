import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from videotrans.configure import contants
from videotrans.configure.config import ROOT_DIR, tr, settings, logger, HOME_DIR
from videotrans.configure import config
from videotrans.configure.contants import BUILTINT_URL_MS, PUNC_RESTORE_MS, DENOISE_URL_MS, BUILTINT_URL_HF, \
    DENOISE_URL_HF, PUNC_RESTORE_HF
from videotrans.recognition import run
from videotrans.task._base import BaseTask
from videotrans.task.taskcfg import TaskCfgSTT
from videotrans.util.help_misc import is_connect_hf

"""
Speech recognition only
"""


@dataclass
class SpeechToText(BaseTask):
    cfg: TaskCfgSTT = field(default_factory=TaskCfgSTT, repr=False)
    # Output subtitle format after recognition, e.g. srt, txt, etc.
    out_format: str = field(init=True, default='srt')
    # In this subclass, should_recogn is always True.
    should_recogn: bool = True
    # Whether to copy the generated subtitle to the original video directory and rename it to match the video name for easy soft subtitle autoloading
    copysrt_rawvideo: bool = field(default=False, init=True)
    # Stores original language subtitles
    source_srt_list: List = field(default_factory=list)
    # Insert speaker prefix at the beginning of subtitle
    spk_insert: bool = False

    def __post_init__(self):
        super().__post_init__()
        # -1=disable speaker diarization, 0=enable with unlimited speakers, >0+1 is maximum speaker count
        self.max_speakers = self.cfg.nums_diariz if self.cfg.enable_diariz else -1
        if self.max_speakers > 0:
            self.max_speakers += 1
        # Target destination folder
        if not self.cfg.target_dir:
            self.cfg.target_dir = HOME_DIR + f"/recogn"
        # Target subtitle file after transcription; first unified to srt, then converted to other formats via ffmpeg if needed
        self.cfg.target_sub = self.cfg.target_dir + '/' + self.cfg.noextname + '.srt'
        # Temporary folder
        self.cfg.cache_folder = config.TEMP_DIR + f'/{self.uuid}'
        # Converted to 16k mono wav audio for model recognition
        self.cfg.shibie_audio = self.cfg.cache_folder + f'/{self.cfg.noextname}-{time.time()}.wav'
        self.signal(text=tr("Speech Recognition to Word Processing"))

    # Pre-processing
    def prepare(self):
        if self._exit(): return
        Path(self.cfg.target_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.cache_folder).mkdir(parents=True, exist_ok=True)
        if getattr(self.cfg, "subtitle_source", "audio_asr") == "video_ocr":
            return
        from videotrans.util.help_ffmpeg import conver_to_16k
        conver_to_16k(self.cfg.name, self.cfg.shibie_audio)

    def recogn(self):
        if getattr(self.cfg, "subtitle_source", "audio_asr") == "video_ocr":
            from videotrans.configure.excepts import SpeechToTextError
            from videotrans.core.proc_registry import get_current_job_id
            from videotrans.ocr import scan_video_to_srt
            from videotrans.util.help_ffmpeg import get_video_info

            if not getattr(self.cfg, "ocr_roi_confirmed", False):
                raise SpeechToTextError(tr("ROI must be confirmed in the OCR preview dialog before starting"))
            roi = getattr(self.cfg, "ocr_roi", None)
            if not roi:
                raise SpeechToTextError(tr("No ROI configured for video OCR"))

            duration_ms = None
            try:
                info = get_video_info(self.cfg.name)
                duration_ms = info.get("time") if isinstance(info, dict) else None
            except Exception:
                duration_ms = None

            ocr_opts = getattr(self.cfg, "ocr_options", None) or {}
            coarse_interval_ms = int(ocr_opts.get("coarse_interval_ms", 500) or 500)
            use_angle_cls = bool(ocr_opts.get("use_angle_cls", False))
            paragraph_mode = str(ocr_opts.get("paragraph_mode") or "preserve_lines")
            job_id = get_current_job_id() or self.uuid

            def _emit_ocr_progress(p: dict) -> None:
                self.signal(
                    text=p.get("latest_text", ""),
                    type="ocr_progress",
                    progress=p.get("percent"),
                    percent=p.get("percent"),
                    timestamp_ms=p.get("timestamp_ms"),
                    duration_ms=p.get("duration_ms"),
                    segments=p.get("segments"),
                    ocr_calls=p.get("ocr_calls"),
                    skipped_frames=p.get("skipped_frames"),
                    device=p.get("device"),
                )

            srt_items, _segments = scan_video_to_srt(
                self.cfg.name,
                roi=roi,
                language=self.cfg.detect_language,
                device="cuda" if self.cfg.is_cuda else "cpu",
                checkpoint_path=f"{self.cfg.cache_folder}/ocr_ckpt.json",
                progress_callback=_emit_ocr_progress,
                stop_event=self._exit,
                job_id=job_id,
                duration_ms=duration_ms,
                coarse_interval_ms=coarse_interval_ms,
                use_angle_cls=use_angle_cls,
                paragraph_mode=paragraph_mode,
            )
            if self._exit(): return
            if not srt_items:
                raise SpeechToTextError(self.cfg.basename + tr('recogn result is empty'))
            self.source_srt_list = srt_items
            self._save_srt_target(self.source_srt_list, self.cfg.target_sub)
            self.signal(text=Path(self.cfg.target_sub).read_text(encoding='utf-8'), type='replace_subtitle')
            return

        while 1:
            if self._exit(): return
            # Not yet generated
            if Path(self.cfg.shibie_audio).exists():
                break
            time.sleep(0.5)

        from videotrans.util.help_down import down_file_from_hf
        from videotrans.configure.excepts import SpeechToTextError

        # Noise reduction needed
        if self.cfg.remove_noise:
            logger.debug('开始降噪')
            try:
                from videotrans.process.prepare_audio import remove_noise
                title = tr('Starting to process speech noise reduction, which may take a long time, please be patient')
                down_file_from_hf(f'{ROOT_DIR}/models/onnx', urls=DENOISE_URL_MS if not is_connect_hf() else DENOISE_URL_HF,
                                            callback=self._process_callback)
                _noise_wav = f"{config.TEMP_DIR}/{self.cfg.noextname}-{os.path.getsize(self.cfg.name)}-removed_noise.wav"
                kw = {
                        "input_file": self.cfg.shibie_audio,
                        "output_file": _noise_wav,
                        "is_cuda": self.cfg.is_cuda
                    }
                _rs = self._new_process(callback=remove_noise, title=title, is_cuda=self.cfg.is_cuda, kwargs=kw)
                if _rs:
                    self.cfg.shibie_audio = _noise_wav
                self.signal(text='remove noise end')
            except Exception as e:
                logger.exception(f'降噪失败，跳过 {e}', exc_info=True)

        if self._exit(): return
        raw_subtitles = run(
            recogn_type=self.cfg.recogn_type,
            uuid=self.uuid,
            model_name=self.cfg.model_name,
            audio_file=self.cfg.shibie_audio,
            detect_language=self.cfg.detect_language,
            cache_folder=self.cfg.cache_folder,
            is_cuda=self.cfg.is_cuda,
            subtitle_type=0,
            max_speakers=self.max_speakers,
            llm_post=self.cfg.rephrase == 1
        )
        if not raw_subtitles or len(raw_subtitles) < 1:
            raise SpeechToTextError(self.cfg.basename + tr('recogn result is empty'))
        self.source_srt_list = raw_subtitles
        self._save_srt_target(self.source_srt_list, self.cfg.target_sub)


        # Chinese/English punctuation restoration
        _lang_prefix = (self.cfg.detect_language or "")[:2].lower()
        if self.cfg.detect_language != 'auto' and self.cfg.fix_punc==1 and _lang_prefix in ['zh', 'en']:

            try:
                from videotrans.process.prepare_audio import fix_punc
                down_file_from_hf(f'{ROOT_DIR}/models/puntc', PUNC_RESTORE_MS if not is_connect_hf() else PUNC_RESTORE_HF, callback=self._process_callback)
                text_dict = {f'{it["line"]}': re.sub(r'[,.?!，。？！]', ' ', it["text"]) for it in self.source_srt_list}
                # Pass file path after serialization
                text_dict_file=f'{self.cfg.cache_folder}/text_dict_file_{time.time()}.json'
                Path(text_dict_file).write_text(json.dumps(text_dict),encoding="utf-8")
                kw = {"text_dict_file": text_dict_file, "is_cuda": self.cfg.is_cuda}
                _rs = self._new_process(callback=fix_punc, title=tr("Restoring punct"), kwargs=kw)
                if _rs:
                    text_dict_obj=json.loads(Path(text_dict_file).read_text(encoding='utf-8'))
                    for it in self.source_srt_list:
                        it['text'] = text_dict_obj.get(f'{it["line"]}', it['text'])
                        if _lang_prefix == 'en':
                            it['text'] = it['text'].replace('，', ',').replace('。', '. ').replace('？', '?').replace(
                                '！', '!')
                    self._save_srt_target(self.source_srt_list, self.cfg.target_sub)
                else:
                    logger.error('标点恢复出错')
            except Exception as e:
                logger.exception(f'恢复标点出错，跳过{e}', exc_info=True)

        # If speaker diarization is already present, do not re-segment
        self.signal(text=Path(self.cfg.target_sub).read_text(encoding='utf-8'), type='replace_subtitle')
        if Path(self.cfg.cache_folder + "/speaker.json").exists(): return
        
        # If speaker diarization is selected, do not use LLM re-segmentation
        if not self.cfg.enable_diariz and self.cfg.rephrase == 1:
            # LLM re-segmentation
            try:
                from videotrans.translator._openaicompat import OpenAICampat
                ob = OpenAICampat(
                    ainame='chatgpt' if settings.get('llm_ai_type', 'chatgpt') != 'deepseek' else 'deepseek',
                    uuid=self.uuid)
                self.signal(text=tr("Re-segmenting..."))
                srt_list = ob.llm_segment(self.source_srt_list)
                if srt_list and len(srt_list) > len(self.source_srt_list) / 2:
                    self.source_srt_list = srt_list
                    self._save_srt_target(self.source_srt_list, self.cfg.target_sub)
                else:
                    logger.error(f'重新断句失败，已恢复原样,原始字幕行:{len(self.source_srt_list)}, 重新断句后字幕行:{len(srt_list)}\n断句结果:\n{srt_list=}')
            except Exception as e:
                self.signal(text=tr("Re-segmenting Error"))
                logger.exception(f"重新断句失败已恢复原样 {e}", exc_info=True)

    def diariz(self):
        if (
            self._exit()
            or not self.cfg.enable_diariz
            or getattr(self.cfg, "subtitle_source", "audio_asr") == "video_ocr"
            or Path(self.cfg.cache_folder + "/speaker.json").exists()
        ):
            return
        from videotrans.util.help_down import down_file_from_hf, check_and_down_ms
        speaker_type = settings.get('speaker_type', 'built')
        hf_token = settings.get('hf_token')
        if speaker_type == 'built' and (self.cfg.detect_language or "")[:2].lower() not in ['zh', 'en']:
            logger.error(f'当前选择 built 说话人分离模型，但不支持当前语言:{self.cfg.detect_language}')
            return
        if speaker_type in ['pyannote', 'reverb'] and not hf_token:
            logger.error(f'当前选择 pyannote 说话人分离模型，但未设置 huggingface.co 的token: {self.cfg.detect_language}')
            return

        self.precent += 3
        ishf=is_connect_hf()
        title = tr(f'Speaker classification') + f':{speaker_type}'
        subtitles_file=f'{self.cfg.cache_folder}/diariz-{time.time()}.json'
        Path(subtitles_file).write_text(json.dumps([[it['start_time'], it['end_time']] for it in self.source_srt_list]),encoding='utf-8')
        kw = {
            "input_file": self.cfg.shibie_audio,
            "subtitles_file": subtitles_file,
            "speak_file":self.cfg.cache_folder + "/speaker.json",
            "num_speakers": self.max_speakers,
            "is_cuda": self.cfg.is_cuda
        }
        if speaker_type == 'built':
            down_file_from_hf(f'{ROOT_DIR}/models/onnx', BUILTINT_URL_MS if not ishf else BUILTINT_URL_HF, callback=self._process_callback)
            from videotrans.process.prepare_audio import built_speakers as _run_speakers
            del kw['is_cuda']
            kw['num_speakers'] = -1 if self.max_speakers < 1 else self.max_speakers
            kw['language'] = self.cfg.detect_language
        elif speaker_type == 'ali_CAM':
            check_and_down_ms(
                model_id='iic/speech_campplus_speaker-diarization_common', 
                local_dir=f"{ROOT_DIR}/models/speech_campplus_speaker-diarization_common",
                callback=self._process_callback)
            from videotrans.process.prepare_audio import cam_speakers as _run_speakers
        elif speaker_type == 'pyannote':
            from videotrans.process.prepare_audio import pyannote_speakers as _run_speakers
        elif speaker_type == 'reverb':
            from videotrans.process.prepare_audio import reverb_speakers as _run_speakers
        else:
            logger.error(f'当前所选说话人分离模型不支持:{speaker_type=}')
            return
        try:
            if speaker_type in ['pyannote', 'reverb']:
                #self.signal(text='Downloading speakers models')
                #from huggingface_hub import snapshot_download
                #snapshot_download(
                #    repo_id="pyannote/speaker-diarization-3.1" if speaker_type == 'pyannote' else "Revai/reverb-diarization-v1",
                #    token=hf_token,
                #    local_dir=f'{ROOT_DIR}/models/models--'+("pyannote--speaker-diarization-3.1" if speaker_type == 'pyannote' else "Revai--reverb-diarization-v1"),
                #    endpoint=os.environ.get('HF_ENDPOINT')
                #)
                from videotrans.util.help_down import check_and_down_hf
                check_and_down_hf(
                    speaker_type, 
                    "pyannote/speaker-diarization-3.1" if speaker_type == 'pyannote' else "Revai/reverb-diarization-v1", 
                    f'{ROOT_DIR}/models/models--'+("pyannote--speaker-diarization-3.1" if speaker_type == 'pyannote' else "Revai--reverb-diarization-v1"), 
                    self._process_callback, 
                    None,
                    hf_token)
            _rs = self._new_process(callback=_run_speakers, title=title,
                                         is_cuda=self.cfg.is_cuda and speaker_type != 'built', kwargs=kw)

            logger.debug('分离说话人成功完成' if _rs else '分离失败说话人失败')
            self.signal(text=tr('separating speakers end'))
        except Exception as e:
            logger.exception(f'说话人分离失败，跳过 {e}', exc_info=True)
        self.signal(text=tr('separating speakers end'))

    def task_done(self):
        if self._exit(): return
        from videotrans.util.help_srt import simple_wrap
        if self.cfg.detect_language and self.cfg.detect_language != 'auto':
            # Handle line wrapping
            maxlen = int(
                settings.get('cjk_len', 15) if self.cfg.detect_language[:2] in contants.CJK_LANG else
                settings.get('other_len', 60))
            for i, it in enumerate(self.source_srt_list):
                it['text'] = simple_wrap(it['text'], maxlen, self.cfg.detect_language)

        # Remove punctuation marks
        if self.cfg.fix_punc==2:
            from videotrans.util.help_srt import delete_punc
            for i, it in enumerate(self.source_srt_list):
                it['text'] = delete_punc(it['text'])


        if self.cfg.enable_diariz and self.spk_insert and Path(
                self.cfg.cache_folder + "/speaker.json").exists():
            speakers = json.loads(Path(self.cfg.cache_folder + "/speaker.json").read_text(encoding='utf-8'))
            if speakers:
                speakers_len = len(speakers)
                for i, it in enumerate(self.source_srt_list):
                    if i < speakers_len and speakers[i]:
                        it['text'] = f'[{speakers[i]}]{it["text"]}'

        self._save_srt_target(self.source_srt_list, self.cfg.target_sub)
        if self.out_format == 'txt':
            self.cfg.target_sub = self.cfg.target_sub[:-3] + 'txt'
            Path(self.cfg.target_sub).write_text("\r\n".join([it["text"] for it in self.source_srt_list]),
                                                 encoding='utf-8')
        elif self.out_format != 'srt':
            from videotrans.util.help_ffmpeg import runffmpeg
            runffmpeg(['-y', '-i', self.cfg.target_sub, self.cfg.target_sub[:-3] + self.out_format])
            Path(self.cfg.target_sub).unlink(missing_ok=True)
            self.cfg.target_sub = self.cfg.target_sub[:-3] + self.out_format

        if self.copysrt_rawvideo:
            p = Path(self.cfg.name)
            try:
                shutil.copy2(self.cfg.target_sub, f'{p.parent.as_posix()}/{p.stem}.{self.out_format}')
            except shutil.SameFileError:
                pass
        self.set_end(True)
