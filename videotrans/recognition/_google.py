import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from videotrans.configure import config
from videotrans.configure.config import tr, settings, logger
from videotrans.recognition._base import BaseRecogn
from videotrans.recognition._retry import run_with_retry
from videotrans.task.taskcfg import SrtItem
from videotrans.util._srt_parse import ms_to_time_string
from videotrans.util.help_misc import vail_file


@dataclass
class GoogleRecogn(BaseRecogn):

    def _exec(self) -> Union[List[SrtItem], None]:
        if self._exit():
            return

        has_cache = bool(self.cache_folder)
        if has_cache:
            tmp_dir = Path(f'{self.cache_folder}/{Path(self.audio_file).name}_tmp')
        else:
            tmp_dir = Path(config.TEMP_DIR) / f'{Path(self.audio_file).name}_{time.time_ns()}_tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir.as_posix()

        raw_lang = (self.detect_language or '').strip()
        lang = raw_lang if raw_lang and raw_lang.lower() not in ('auto', '-') else 'en-US'

        self.raws = []
        try:
            normalized_sound = AudioSegment.from_wav(self.audio_file)  # -20.0
            nonslient_file = f'{tmp_path}/detected_voice.json'
            if vail_file(nonslient_file):
                with open(nonslient_file, 'r') as f:
                    nonsilent_data = json.load(f)
            else:
                nonsilent_data = self._shorten_voice_old(normalized_sound)
                with open(nonslient_file, 'w') as f:
                    f.write(json.dumps(nonsilent_data))

            total_length = len(nonsilent_data)
            recognizer = sr.Recognizer()
            recognizer.operation_timeout = 60

            for i, duration in enumerate(nonsilent_data):
                if self._exit():
                    return
                start_time, end_time, buffered = duration
                if start_time == end_time:
                    end_time += int(settings.get('voice_silence', 140))

                chunk_filename = tmp_path + f"/c{i}_{start_time // 1000}_{end_time // 1000}.wav"
                audio_chunk = normalized_sound[start_time:end_time]
                audio_chunk.export(chunk_filename, format="wav")

                try:
                    with sr.AudioFile(chunk_filename) as source:
                        audio_data = recognizer.record(source)
                finally:
                    try:
                        Path(chunk_filename).unlink(missing_ok=True)
                    except OSError:
                        pass

                def _recognize_chunk():
                    try:
                        return recognizer.recognize_google(audio_data, language=lang)
                    except sr.UnknownValueError:
                        return ""

                text = run_with_retry(
                    _recognize_chunk,
                    should_retry=lambda exc: isinstance(exc, (sr.RequestError, TimeoutError)),
                    is_cancelled=self._exit,
                    label="GoogleRecogn",
                )

                text = re.sub(r'&#\d+;', '', f"{text.capitalize()}. ".replace('&#39;', "'"), flags=re.I | re.S).strip()
                if not text or re.match(r'^[，。、？‘’“”；：（｛｝【】）:;"\'\s \d`!@#$%^&*()_+=.,?/\\-]*$', text):
                    continue
                start = ms_to_time_string(ms=start_time)
                end = ms_to_time_string(ms=end_time)
                srt_line = SrtItem(
                    line=len(self.raws) + 1,
                    time=f"{start} --> {end}",
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    startraw=start,
                    endraw=end
                )
                self.raws.append(srt_line)
                self.signal(text=f"{tr('yuyinshibiejindu')} {srt_line['line']}/{total_length}")
                self.signal(text=f"{srt_line['text']}\n", type='subtitle')
                if self.asr_wait > 0:
                    time.sleep(self.asr_wait)
            return self.raws
        finally:
            if not has_cache:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def match_target_amplitude(self, sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    def _shorten_voice_old(self, normalized_sound):
        normalized_sound = self.match_target_amplitude(normalized_sound, -20.0)
        max_interval = int(settings.get('max_speech_duration_s',5)) * 1000
        buffer = int(settings.get('min_silence_duration_ms',140))
        nonsilent_data = []
        audio_chunks = detect_nonsilent(normalized_sound, min_silence_len=int(settings.get('min_silence_duration_ms',140)),
                                        silence_thresh=-20 - 25)
        for i, chunk in enumerate(audio_chunks):
            start_time, end_time = chunk
            n = 0
            while end_time - start_time >= max_interval:
                n += 1
                new_end = start_time + max_interval + buffer
                new_start = start_time
                nonsilent_data.append((new_start, new_end, True))
                start_time += max_interval
            nonsilent_data.append((start_time, end_time, False))
        return nonsilent_data
