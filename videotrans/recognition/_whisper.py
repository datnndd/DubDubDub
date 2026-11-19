import time, json, datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, List

from videotrans.configure._paths import TEMP_ROOT
from videotrans.configure.config import settings, ROOT_DIR, logger
from videotrans.configure import config
from videotrans.recognition._base import BaseRecogn
from videotrans.task.taskcfg import SrtItem

from videotrans.util.help_down import check_and_down_hf
from videotrans.util.help_srt import get_subtitle_from_srt


from videotrans.configure.contants import FASTER_MODELS_DICT
from videotrans.configure.excepts import SttTimeoutError


def get_audio_duration_ms(audio_path: Optional[Union[str, Path]]) -> int:
    """Return audio duration in milliseconds via header-only metadata probe."""
    if not audio_path:
        return 0
    path_obj = Path(audio_path)
    if not path_obj.is_file():
        return 0
    try:
        import soundfile as sf
        info = sf.info(str(path_obj))
        if info.samplerate > 0:
            return int(round(info.frames * 1000 / info.samplerate))
    except Exception:
        pass
    try:
        import wave
        with wave.open(str(path_obj), "rb") as wf:
            sr = wf.getframerate()
            if sr > 0:
                return int(round(wf.getnframes() * 1000 / sr))
    except Exception:
        pass
    try:
        from videotrans.util._ffprobe import get_audio_time
        ms = get_audio_time(str(path_obj))
        return int(round(ms)) if ms else 0
    except Exception:
        return 0


@dataclass
class FasterAll(BaseRecogn):
    def __post_init__(self):
        super().__post_init__()
        local_dir = f'{ROOT_DIR}/models/models--'
        model_key = self.model_name or ''
        if model_key in FASTER_MODELS_DICT:
            local_dir += FASTER_MODELS_DICT[model_key].replace('/', '--')
        else:
            local_dir += model_key.replace('/', '--')
        self.local_dir = local_dir
        self.audio_duration = get_audio_duration_ms(self.audio_file)
        self.speech_timestamps_file = None


    def _exec(self)->Union[List[SrtItem], None]:
        if self._exit(): return
        self.error = ''
        self.signal(text="STT starting, hold on...")
        return self._faster()

    def _download(self):
        repo_id = FASTER_MODELS_DICT.get(self.model_name, self.model_name)
        check_and_down_hf(self.model_name, repo_id, self.local_dir, callback=self._process_callback)
        # Pre-segment using VAD when batching
        # Otherwise post-segmentation processing

        if settings.get('whisper_prepare'):
            self._vad_split()
            self.speech_timestamps_file=f'{self.cache_folder}/speech_timestamps_{time.time()}.json'
            Path(self.speech_timestamps_file).write_text(json.dumps(self.speech_timestamps),encoding='utf-8')


    def _faster(self)->Union[List[SrtItem], None]:
        title=f"Model: {self.model_name}"
        self.signal(text=title)
        logs_file = f'{config.TEMP_DIR}/{self.uuid}/faster-{self.detect_language}-{time.time()}.log'
        Path(logs_file).parent.mkdir(parents=True, exist_ok=True)
        _max_speech=max(int(float(settings.get('max_speech_duration_s', 5)) * 1000),2000)
        if self.recogn2pass:
            # Second-pass recognition, generates brief subtitles
            _max_speech = max(int(float(settings.get('max_speech_duration_s2', 2)) * 1000),500)
        
        base_tmp = self.cache_folder or TEMP_ROOT
        Path(base_tmp).mkdir(parents=True, exist_ok=True)
        subtitle_srt = f'{base_tmp}/faster-{datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")}-{time.time_ns()}.srt'
        kwargs = {
            "detect_language": self.detect_language,
            "model_name": self.model_name,
            "logs_file": logs_file,
            "is_cuda": self.is_cuda,
            "no_speech_threshold": float(settings.get('no_speech_threshold', 0.6)),
            "condition_on_previous_text": settings.get('condition_on_previous_text', False),
            "speech_timestamps": self.speech_timestamps_file,
            "audio_file": self.audio_file,
            "local_dir": self.local_dir,
            "compute_type": settings.get('cuda_com_type', 'int8'),
            "jianfan": self.jianfan,
            "audio_duration":self.audio_duration,
            "hotwords":settings.get('hotwords'),
            "prompt": settings.get(f'initial_prompt_{self.detect_language}') if self.detect_language != 'auto' else None,
            "beam_size": int(settings.get('beam_size', 5)),
            "best_of": int(settings.get('best_of', 5)),
            "temperature":settings.get('temperature'),
            "repetition_penalty":float(settings.get('repetition_penalty',1.0)),
            "compression_ratio_threshold":float(settings.get('compression_ratio_threshold',2.2)),
            "max_speech_ms":_max_speech,
            "subtitle_srt":subtitle_srt
        }
        try:
            from videotrans.process.stt_faster import  faster_whisper
            raws=self._new_process(callback=faster_whisper,title=title,is_cuda=self.is_cuda,kwargs=kwargs)
            return raws
        except SttTimeoutError:
            if not Path(subtitle_srt).exists():
                raise
            logger.debug(f'捕获到强制抛出的 SttTimeoutError, 使用已识别的文件 {subtitle_srt}')
            return get_subtitle_from_srt(subtitle_srt, is_file=True)
        finally:
            try:
                Path(subtitle_srt).unlink(missing_ok=True)
            except OSError:
                pass
