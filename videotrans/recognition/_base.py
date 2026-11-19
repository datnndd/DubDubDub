import re, time,os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from videotrans.configure.config import tr, settings, logger
from videotrans.configure import config
from videotrans.configure.base import BaseCon
from videotrans.task.taskcfg import SrtItem
from videotrans.configure import contants
from videotrans.util.help_srt import ms_to_time_string
from tenacity import RetryError


@dataclass
class BaseRecogn(BaseCon):
    recogn_type: int = 0  # Speech recognition type
    # Subtitle detection language
    detect_language: str = None

    # Model name
    model_name: Optional[str] = None
    # 16k wav to be recognized
    audio_file: Optional[str] = None
    # Temporary directory
    cache_folder: Optional[str] = None

    # Task ID
    uuid: Optional[str] = None
    # Enable CUDA acceleration
    is_cuda: bool = False

    # Subtitle embedding type: 0, 1, 2, 3, 4
    subtitle_type: int = 0
    # Whether task has finished
    has_done: bool = field(default=False, init=False)
    # Error message
    error: str = field(default='', init=False)
    # Recognition API URL
    api_url: str = field(default='', init=False)
    # Device type: cpu, cuda
    device: str = field(init=False, default='cpu')
    # Punctuation marks
    flag: List[str] = field(init=False, default_factory=list)
    # Stores returned subtitle list
    raws: List = field(default_factory=list, init=False)
    # Word joining character: direct concatenation for CJK/Cantonese, space for others
    join_word_flag: str = field(init=False, default=' ')
    # Whether conversion to Simplified Chinese is needed
    jianfan: bool = False
    # Max characters per subtitle line
    maxlen: int = 20
    audio_duration: int = 0
    max_speakers: int = -1  # Speakers: -1 disables diarization, 0=unlimited, >0 maximum speaker count
    llm_post: bool = False  # Whether to perform LLM resegmentation; if so, skip simple post-fix after recognition
    speech_timestamps: List = field(default_factory=list)  # VAD split segments
    recogn2pass: bool = False
    asr_wait: float = float(settings.get('asr_wait', 0))
    local_dir: str = None
    deepgram_options: Optional[Union[dict, str]] = None
    transcript_options: Optional[dict] = None

    def __post_init__(self):
        super().__post_init__()
        self.device = 'cuda' if self.is_cuda else 'cpu'
        # Common punctuation
        self.flag = contants.PUNC_FLAGS
        # Soft punctuation like comma
        self.half_flag = contants.PUNC_FLAGS_HALF
        # Sentence terminating punctuation
        self.end_flag = contants.PUNC_FLAGS_END
        # Join character: CJK, Cantonese, Khmer, Thai connect directly without spaces; other languages connect with space
        self.join_word_flag = " "
        # CJK characters
        self.is_cjk = False

        if self.detect_language and self.detect_language[:2].lower() in contants.CJK_LANG:
            self.maxlen = int(float(settings.get('cjk_len', 20)))
            self.jianfan = True if self.detect_language[:2] == 'zh' and settings.get('zh_hant_s') else False
            self.flag.append(" ")
            self.join_word_flag = ""
            self.is_cjk = True
        else:
            self.maxlen = int(float(settings.get('other_len', 60)))
            self.jianfan = False

    # run->_exec
    def run(self) -> Union[List[SrtItem], None]:
        try:
            if hasattr(self, '_download'):
                self.signal(text=tr("check or download models"))
                self._download()
            self.signal(text=tr("Transcription in progress, please wait"))
            res = self._exec()
            if res:
                return self._post_fix(res)
            from videotrans.configure.excepts import SpeechToTextError
            raise SpeechToTextError(
                tr('No speech was detected, please make sure there is human speech in the selected audio/video and that the language is the same as the selected one.'))
        except RetryError as e:
            raise e.last_attempt.exception()
        except (OSError,FileNotFoundError) as e:
            _e=str(e)
            if self.local_dir and ("no file named model.safetensors" in _e or os.path.basename(self.local_dir) in _e):
               from videotrans.configure.excepts import DownloadModelsError
               raise  DownloadModelsError(tr('model incomplete error',self.local_dir,tr('Help document')))
            raise


    # Simple post-processing on transcription results
    def _post_fix(self, res: List[SrtItem]) -> List[SrtItem]:
        srt_list = []
        logger.debug('移除无效字幕行')
        for i, it in enumerate(res):
            text = it['text'].strip()
            # Remove invalid subtitle lines composed entirely of symbols
            if text and not re.match(contants.NON_WORD, text):
                it['line'] = len(srt_list) + 1
                srt_list.append(it)
            else:
                logger.warning(f'移除无效字幕行,全部由符号组成的行：{i=},{text=}')

        if not srt_list:
            return []

        # Fix timestamp overlap
        logger.debug('修正重叠时间轴')
        for i, it in enumerate(srt_list):
            if i > 0 and srt_list[i - 1]['end_time'] > it['start_time']:
                logger.warning(
                    f'修正字幕时间轴重叠：将前面字幕 end_time={srt_list[i - 1]["end_time"]} 改为当前字幕 start_time, {it=}')
                srt_list[i - 1]['end_time'] = it['start_time']
                srt_list[i - 1]['endraw'] = ms_to_time_string(ms=it['start_time'])
                srt_list[i - 1]['time'] = f"{srt_list[i - 1]['startraw']} --> {srt_list[i - 1]['endraw']}"
        
        
        
        # If not LLM resegmentation and merge short subtitles is selected, perform merge
        if not self.recogn2pass and not self.llm_post and settings.get('merge_short_sub', True):
            logger.debug('开始合并邻近短字幕')
            srt_list=self._merge_sub(srt_list)

        if settings.get('del_end_punc'):
            logger.debug(f'开始移除每条字幕末尾标点')
            for it in srt_list:
                # Remove trailing punctuation
                it['text'] = it['text'].strip('。，？！,.?!').strip()
        return srt_list

    def _exec(self) -> Union[List[SrtItem], None]:
        raise NotImplemented()

    # Some recognition channels require pre-cutting audio into appropriate length clips using VAD, then recognizing each clip as a subtitle
    # Whisper models without pre-split selected do not need cutting
    def _vad_split(self):
        _st = time.time()
        _vad_type = settings.get('vad_type', 'tenvad')
        title = f'VAD:{_vad_type} split audio...'
        self.signal(text=title)

        _threshold = float(settings.get('threshold', 0.5))
        _min_speech = max(int(float(settings.get('min_speech_duration_ms', 1000))), 0)
        
        # Ten-vad minimum segment cannot be less than 500ms
        if _vad_type == 'tenvad':
            _min_speech = max(_min_speech, 500)

        # Maximum segment cannot exceed 30s, and cannot be less than _min_speech
        _max_speech = max(min(int(float(settings.get('max_speech_duration_s', 6)) * 1000), 30000), _min_speech + 1000)
        
        # Silence threshold cannot be less than 25ms
        _min_silence = max(int(settings.get('min_silence_duration_ms', 600)), 25)
        if self.recogn2pass:
            # Second-pass recognition, generates brief subtitles
            _min_speech = max( int(float(settings.get('min_speech_duration_ms2', 1000))), 500)
            _max_speech = max( min( int(float(settings.get('max_speech_duration_s2', 2)) * 1000), 4000), _min_speech + 500)
            logger.debug(f'[当前是二次语音识别]{_vad_type},{_min_speech=}ms,{_max_speech=}ms,{_min_silence=}ms')

        kw = {
            "input_wav": self.audio_file,
            "threshold": _threshold,
            "min_speech_duration_ms": _min_speech,
            "max_speech_duration_ms": _max_speech,
            "min_silent_duration_ms": _min_silence
        }

        try:
            from videotrans.process.vad import get_speech_timestamp, get_speech_timestamp_silero
            self.speech_timestamps = self._new_process(
                callback=get_speech_timestamp if _vad_type == 'tenvad' else get_speech_timestamp_silero,
                title=title,
                kwargs=kw)
        except Exception as e:
            logger.exception(f'VAD 处理失败 {e}', exc_info=True)
            if not self.recogn2pass:
                raise
        self.signal(text=f'[VAD] ended {int(time.time() - _st)}s')


    def cut_audio(self) -> List[SrtItem]:
        import numpy as np

        base_dir = Path(self.cache_folder) if self.cache_folder else Path(config.TEMP_DIR)
        dir_name = str(base_dir / f"clip_{time.time_ns()}")
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        self._last_cut_dir = dir_name

        if not self.speech_timestamps:
            self._vad_split()

        # Load audio (16k mono) without eager full-file float32/abs copies
        try:
            import soundfile as sf
            samples, samplerate = sf.read(self.audio_file, dtype='int16', always_2d=False)
            if samples.ndim > 1:
                samples = np.mean(samples, axis=1).astype(np.int16)
        except Exception:
            import wave
            with wave.open(self.audio_file, 'rb') as wf:
                samplerate = wf.getframerate()
                n_channels = wf.getnchannels()
                raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16)
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

        # Minimum segment duration (at least 1000ms, at most 25000ms)
        min_speech_duration_ms = min(25000, max(int(settings.get('min_speech_duration_ms', 1000)), 1000))
        max_speech_duration_ms = 30000

        # Deep copy
        segs = [seg[:] for seg in self.speech_timestamps]
        segs = [[max(0, s), max(0, e)] for s, e in segs if e > s]

        # Short segment merge (stack-based algorithm)
        merged = []
        for seg in segs:
            if not merged:
                merged.append(seg)
                continue
            # If previous segment is too short, merge forward into current segment
            while merged and (merged[-1][1] - merged[-1][0]) < min_speech_duration_ms:
                prev = merged.pop()
                seg[0] = prev[0]   # Current segment absorbs previous one
            # If current segment itself is still too short, try merging into top of stack (if exists)
            if (seg[1] - seg[0]) < min_speech_duration_ms:
                if merged:
                    merged[-1][1] = seg[1]
                else:
                    merged.append(seg)   # Keep isolated short segment (cannot merge further)
            else:
                merged.append(seg)
        segs = merged

        # Truncate overly long segments (compute energy only for slices exceeding max_speech_duration_ms)
        final_segs = []
        for s, e in segs:
            dur = e - s
            if dur <= max_speech_duration_ms:
                final_segs.append([s, e])
                continue

            # Search for lowest energy point in the 70%~100% time region as a safe cut point
            start_sample = max(0, int(s * samplerate / 1000))
            end_sample = min(len(samples), int(e * samplerate / 1000))
            segment_energy = np.abs(samples[start_sample:end_sample].astype(np.float32))
            search_start = int(len(segment_energy) * 0.7)
            search_end = len(segment_energy) - 1
            if search_end > search_start:
                min_idx = search_start + int(np.argmin(segment_energy[search_start:search_end]))
            else:
                min_idx = len(segment_energy) // 2   # Fallback: split in half

            cut_ms = s + int(min_idx * 1000 / samplerate)
            # Avoid creating extremely short segments (keep at least 1 second)
            cut_ms = max(s + 1000, min(cut_ms, e - 1000))
            final_segs.append([s, cut_ms])
            final_segs.append([cut_ms, e])

        # Add 500ms silence head/tail to each segment and export
        pad_samples = int(samplerate * 0.5)
        silence_pad = np.zeros(pad_samples, dtype=np.int16)

        data = []
        for i, (start_ms, end_ms) in enumerate(final_segs):
            startraw = ms_to_time_string(ms=start_ms)
            endraw = ms_to_time_string(ms=end_ms)
            s_idx = max(0, int(start_ms * samplerate / 1000))
            e_idx = min(len(samples), int(end_ms * samplerate / 1000))
            padded = np.concatenate([silence_pad, samples[s_idx:e_idx], silence_pad])
            file_name = f"{dir_name}/audio_{i}.wav"
            try:
                import soundfile as sf
                sf.write(file_name, padded, samplerate, subtype='PCM_16')
            except Exception:
                import wave
                with wave.open(file_name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(samplerate)
                    wf.writeframes(padded.tobytes())
            data.append(SrtItem(
                line=i + 1,
                text="",
                start_time=start_ms,
                end_time=end_ms,
                startraw=startraw,
                endraw=endraw,
                time=f'{startraw} --> {endraw}',
                filename=file_name
            ))

        logger.debug(f'切分为 {len(data)} 个音频片段')
        return data
    

    def _merge_sub(self, srt_list: List[SrtItem]) -> List[SrtItem]:
        """Merge overly short subtitles and redistribute fragments by punctuation."""
        def _finalize(items: List[SrtItem]) -> List[SrtItem]:
            out = [it for it in items if it['text'].strip()]
            for idx, item in enumerate(out):
                item['line'] = idx + 1
            return out

        post_srt_raws = []
        min_speech = max(300, int(float(settings.get('min_speech_duration_ms', 1000))))
        max_speech = int(1000*float(settings.get('max_speech_duration_s', 5)))
        logger.debug(f'对识别出的字幕进行简单合并与修正，{min_speech=}ms,{max_speech=}ms')

        # Phase 1: iterate and merge overly short items
        post_srt_raws = self._phase1_merge_short(srt_list, min_speech, post_srt_raws,max_speech)

        if len(post_srt_raws) < 2:
            return _finalize(post_srt_raws)

        # Phase 2: handle first item if too short
        post_srt_raws = self._phase2_merge_first(post_srt_raws, min_speech)
        if len(post_srt_raws) < 2:
            return _finalize(post_srt_raws)

        # Phase 3: handle last item if too short
        post_srt_raws = self._phase3_merge_last(post_srt_raws, min_speech)
        if len(post_srt_raws) < 2:
            return _finalize(post_srt_raws)

        # Phase 4: redistribute punctuation fragments forward
        post_srt_raws = self._phase4_redistribute_by_punct(post_srt_raws, forward=True)
        # Phase 5: redistribute punctuation fragments backward
        post_srt_raws = self._phase4_redistribute_by_punct(post_srt_raws, forward=False)

        # Phase 6: clean trailing punctuation, remove blank subtitles
        if settings.get('del_end_punc'):
            for it in post_srt_raws:
                # Remove trailing punctuation
                it['text'] = it['text'].strip('。,.').strip()
        return _finalize(post_srt_raws)

    def _phase1_merge_short(self, srt_list, min_speech, post_srt_raws,max_speech=5000):
        """Iterate over original list, merging short subtitles into adjacent neighbors."""
        for idx, it in enumerate(srt_list):
            if not it['text'].strip():
                continue
            
            _words_len=len(it['text'].strip()) if self.is_cjk else len(it['text'].strip().split(' '))
            if idx == 0 or idx == len(srt_list) - 1 or not post_srt_raws or (it['end_time'] - it['start_time'] >= min_speech and _words_len>1):
                post_srt_raws.append(it)
                continue

            prev_diff = it['start_time'] - post_srt_raws[-1]['end_time']
            next_diff = srt_list[idx + 1]['start_time'] - it['end_time']
            prev_text = post_srt_raws[-1]['text'].strip()
            cur_text = it['text'].strip()
            merge_forward = (
                    (bool(prev_text) and prev_text[-1] not in self.flag and bool(cur_text) and cur_text[-1] in self.flag)
                    or (bool(prev_text) and prev_text[-1] in self.half_flag and bool(cur_text) and cur_text[-1] in self.end_flag)
                    or prev_diff <= next_diff
            )
            # If should merge forward, but previous item length exceeds max allowed duration and difference <= 2s, merge backward instead; otherwise merge forward
            if merge_forward and (prev_diff+2000>next_diff) and (post_srt_raws[-1]['end_time']-post_srt_raws[-1]['start_time'] >max_speech):
                merge_forward=False
                logger.warning(f'应合并到前边字幕，但已过长，因此强制合并进后个字幕')

            # If scheduled to merge forward, but only 1-2 characters and time is contiguous, merge backward instead
            if merge_forward and idx < len(srt_list) - 1 and _words_len<3 and next_diff==0:
                merge_forward=False
                logger.warning(f'已是要求合并到前边，但是只有1-2个字符，并且前后时间相连，则合并到后边,{next_diff=},{it["text"]=},{idx=}')
            
            
            if merge_forward:
                self._log_merge('前', it, post_srt_raws[-1], prev_diff, next_diff)
                post_srt_raws[-1]['end_time'] = it['end_time']
                post_srt_raws[-1]['endraw'] = ms_to_time_string(ms=it['end_time'])
                post_srt_raws[-1]['time'] = f"{post_srt_raws[-1]['startraw']} --> {post_srt_raws[-1]['endraw']}"
                post_srt_raws[-1]['text'] += self.join_word_flag + it['text']
            else:
                self._log_merge('后', it, srt_list[idx + 1], prev_diff, next_diff)
                srt_list[idx + 1]['text'] = it['text'] + self.join_word_flag + srt_list[idx + 1]['text']
                srt_list[idx + 1]['start_time'] = it['start_time']
                srt_list[idx + 1]['startraw'] = ms_to_time_string(ms=it['start_time'])
                srt_list[idx + 1]['time'] = f"{srt_list[idx + 1]['startraw']} --> {srt_list[idx + 1]['endraw']}"
        return post_srt_raws

    def _phase2_merge_first(self, post_srt_raws, min_speech):
        """First item duration < min_speech and gap with next item < 2s -> merge."""
        if (post_srt_raws[0]['end_time'] - post_srt_raws[0]['start_time'] < min_speech
                and post_srt_raws[1]['start_time'] - post_srt_raws[0]['end_time'] < 2000) or len(post_srt_raws[0]['text'].strip())<2:
            post_srt_raws[1]['start_time'] = post_srt_raws[0]['start_time']
            startraw = ms_to_time_string(ms=post_srt_raws[1]['start_time'])
            endraw = ms_to_time_string(ms=post_srt_raws[1]['end_time'])
            post_srt_raws[1]['startraw'] = startraw
            post_srt_raws[1]['endraw'] = endraw
            post_srt_raws[1]['time'] = f"{startraw} --> {endraw}"
            post_srt_raws[1]['text'] = post_srt_raws[0]['text'] + self.join_word_flag + post_srt_raws[1]['text']
            post_srt_raws.pop(0)
        return post_srt_raws

    def _phase3_merge_last(self, post_srt_raws, min_speech):
        """Last item duration < min_speech and gap with previous item < 2s -> merge."""
        if (post_srt_raws[-1]['end_time'] - post_srt_raws[-1]['start_time'] < min_speech
                and post_srt_raws[-1]['start_time'] - post_srt_raws[-2]['end_time'] < 2000) or len(post_srt_raws[-1]['text'].strip())<2:
            post_srt_raws[-2]['end_time'] = post_srt_raws[-1]['end_time']
            startraw = ms_to_time_string(ms=post_srt_raws[-2]['start_time'])
            endraw = ms_to_time_string(ms=post_srt_raws[-2]['end_time'])
            post_srt_raws[-2]['startraw'] = startraw
            post_srt_raws[-2]['endraw'] = endraw
            post_srt_raws[-2]['time'] = f"{startraw} --> {endraw}"
            post_srt_raws[-2]['text'] += self.join_word_flag + post_srt_raws[-1]['text']
            post_srt_raws.pop(-1)
        return post_srt_raws

    def _phase4_redistribute_by_punct(self, post_srt_raws, forward):
        """Redistribute short fragments from current subtitle to previous/next neighbor based on punctuation."""
        for i, it in enumerate(post_srt_raws):
            if i == 0 or i == len(post_srt_raws) - 1:
                continue
            if not it['text'].strip():
                continue
            neighbour = i - 1 if forward else i + 1
            if post_srt_raws[neighbour]['end_time' if forward else 'start_time'] != it[
                'start_time' if forward else 'end_time']:
                continue

            fragments = [t for t in re.split(r'[,.，。]', it['text']) if t.strip()]
            if not fragments:
                it['text'] = ''
                continue
            if len(fragments) == 1:
                continue

            target_fragment = fragments[0] if forward else fragments[-1]
            # Check if fragment is too long
            if self.is_cjk:
                if len(target_fragment.strip()) > 3:
                    continue
            else:
                if len(target_fragment.strip().split(' ')) > 3:
                    continue

            # Skip if neighbor ends/starts with terminal punctuation or is empty
            prev_text = post_srt_raws[i - 1]['text'].strip()
            cur_text = it['text'].strip()
            if forward and (not prev_text or prev_text[-1] in self.flag):
                continue
            if not forward and (not cur_text or cur_text[-1] in self.flag):
                continue

            if forward:
                cut_len = len(fragments[0]) + 1
                moved_text = it['text'][:cut_len].strip()
                post_srt_raws[i - 1]['text'] = (
                    post_srt_raws[i - 1]['text'].rstrip() + self.join_word_flag + moved_text
                ).strip()
                it['text'] = it['text'][cut_len:].strip()
            else:
                moved_text = fragments[-1].strip()
                post_srt_raws[i + 1]['text'] = (
                    moved_text + self.join_word_flag + post_srt_raws[i + 1]['text'].lstrip()
                ).strip()
                it['text'] = it['text'][:-len(fragments[-1])].strip()

            logger.warning(f'该字幕原始文字={it["text"]}, 合并进{"前" if forward else "后"}条字幕的文字={moved_text}')
        return post_srt_raws

    @staticmethod
    def _log_merge(direction, current, neighbour, prev_diff, next_diff):
        logger.warning(
            f'\n[P]字幕时长过短，合并进 [{direction}面] 字幕,{prev_diff=},{next_diff=}\n当前被合并字幕={current}\n合并到的字幕={neighbour}')
