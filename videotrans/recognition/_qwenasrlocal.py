import json
import shutil
from dataclasses import dataclass, asdict
from typing import List, Union

from pathlib import Path
import time

from videotrans.configure.config import logger, ROOT_DIR, settings
from videotrans.configure import config

from videotrans.recognition._base import BaseRecogn
from videotrans.task.taskcfg import SrtItem
from videotrans.util.help_down import check_and_down_ms, check_and_down_hf
from videotrans.util.help_misc import is_connect_hf


@dataclass
class QwenasrlocalRecogn(BaseRecogn):
    def __post_init__(self):
        super().__post_init__()
        self.local_dir=f'{ROOT_DIR}/models/models--Qwen--Qwen3-ASR-{self.model_name}'
    
    def _download(self):

        if not is_connect_hf():
            check_and_down_ms(f'Qwen/Qwen3-ASR-{self.model_name}', callback=self._process_callback,
                                    local_dir=self.local_dir)
        else:
            check_and_down_hf(model_id=f'Qwen3-ASR-{self.model_name}',
                                    repo_id=f'Qwen/Qwen3-ASR-{self.model_name}',
                                    local_dir=self.local_dir,
                                    callback=self._process_callback)

    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit(): return

        logs_file = f'{config.TEMP_DIR}/{self.uuid}/qwen3tts-{time.time()}.log'
        Path(logs_file).parent.mkdir(parents=True, exist_ok=True)
        title = "Qwen3-ASR"
        cut_audio_list_file = f'{config.TEMP_DIR}/{self.uuid}/cut_audio_list_{time.time_ns()}.json'
        clip_dir = None
        try:
            seg_list = self.cut_audio()
            if seg_list:
                candidate = Path(seg_list[0]['filename']).parent
                if candidate.name.startswith("clip_"):
                    clip_dir = candidate
            Path(cut_audio_list_file).write_text(
                json.dumps([asdict(item) for item in seg_list]),
                encoding='utf-8'
            )
            kwargs = {
                "cut_audio_list": cut_audio_list_file,
                "logs_file": logs_file,
                "is_cuda": self.is_cuda,
                "audio_file": self.audio_file,
                "model_name": self.model_name,
                "hotword": settings.get('hotwords'),
                "detect_language": self.detect_language,
            }
            from videotrans.process.stt_qwen import qwen3asr_fun
            jsdata = self._new_process(callback=qwen3asr_fun, title=title, is_cuda=self.is_cuda, kwargs=kwargs)
            logger.debug(f'Qwen-asr返回的字词时间戳数据:{jsdata=}')
            return jsdata
        finally:
            if clip_dir is not None:
                shutil.rmtree(clip_dir, ignore_errors=True)
            elif getattr(self, "_last_cut_dir", None):
                shutil.rmtree(self._last_cut_dir, ignore_errors=True)
            try:
                Path(cut_audio_list_file).unlink(missing_ok=True)
            except OSError:
                pass
