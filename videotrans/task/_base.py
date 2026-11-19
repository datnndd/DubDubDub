import copy
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union
from videotrans.configure.config import tr, app_cfg, logger
from videotrans.configure.base import BaseCon
from videotrans.task.taskcfg import TaskCfgBase, SrtItem

@dataclass
class BaseTask(BaseCon):
    # Configuration info, such as translation, dubbing, recognition channels, etc.
    cfg: TaskCfgBase = field(default_factory=TaskCfgBase, repr=False)
    # Progress tracking
    precent: int = 1
    # Original subtitle information requiring dubbing: List[dict]
    queue_tts: List = field(default_factory=list, repr=False)
    # Whether the task has ended
    hasend: bool = False
    # Whether speech recognition is needed
    should_recogn: bool = False
    # Whether subtitle translation is needed
    should_trans: bool = False
    # Whether dubbing is needed
    should_dubbing: bool = False
    # Whether vocal separation is needed
    should_separate: bool = False
    # Whether embedding dubbing or subtitles is needed
    should_hebing: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.cfg.uuid:
            self.uuid = self.cfg.uuid

    # Pre-processing, such as extracting audio from video, vocal/background separation, transcoding, etc.
    def prepare(self):
        pass

    # Speech recognition to create original language subtitles
    def recogn(self):
        pass

    # Speaker diarization: except for Funasr/Doubao ASR LLM/Deepgram, check if speakers already exist; Gemini/openai gpt4-dia generate speakers
    def diariz(self):
        pass

    # Translate original language subtitles to target language subtitles
    def trans(self):
        pass

    # Perform dubbing according to queue_tts
    def dubbing(self):
        pass

    # Dubbing speedup, video slowdown alignment
    def align(self):
        pass

    # Merge video, audio, and subtitles to generate result file
    def assembling(self):
        pass

    # Delete temporary files, move or copy, send success message
    def task_done(self):
        pass

    # Delete invalid files with size 0
    def _unlink_size0(self, file: Union[str, List[str]]):
        if not file: return
        files = [file] if isinstance(file, str) else file
        for f in files:
            p = Path(f)
            if p.exists() and p.stat().st_size == 0:
                p.unlink(missing_ok=True)

    # Save subtitle file to target folder
    def _save_srt_target(self, srt_list: List[SrtItem], file: str):
        from videotrans.util.help_srt import get_srt_from_list
        try:
            txt = get_srt_from_list(srt_list)
            with open(file, "w", encoding="utf-8", errors="ignore") as f:
                f.write(txt)
        except Exception as e:
            from videotrans.configure.excepts import VideoTransError
            raise VideoTransError(f'保存字幕前格式化srt失败:{file=}') from e

        self.signal(text=Path(file).read_text(encoding='utf-8', errors="ignore"), type='replace_subtitle')
        return True

    @staticmethod
    def _norm_srt_time(time_str: str) -> str:
        return " ".join((time_str or "").split())

    @staticmethod
    def _parse_time_bounds_ms(item: dict) -> tuple[int, int] | None:
        st = item.get("start_time")
        et = item.get("end_time")
        if (
            isinstance(st, (int, float))
            and isinstance(et, (int, float))
            and (et > 0 or st > 0 or "-->" not in str(item.get("time") or ""))
        ):
            return int(st), int(et)
        raw_time = str(item.get("time") or "")
        if "-->" not in raw_time:
            return None
        left, _, right = raw_time.partition("-->")
        def _to_ms(token: str) -> int | None:
            parts = token.strip().replace(".", ",").split(":")
            if len(parts) != 3:
                return None
            try:
                hh = int(parts[0])
                mm = int(parts[1])
                sec_part = parts[2].split(",")
                ss = int(sec_part[0])
                ms = int((sec_part[1] + "000")[:3]) if len(sec_part) > 1 else 0
                return ((hh * 60 + mm) * 60 + ss) * 1000 + ms
            except (TypeError, ValueError):
                return None
        s_ms = _to_ms(left)
        e_ms = _to_ms(right)
        if s_ms is None or e_ms is None:
            return None
        return s_ms, e_ms

    # If LLM resegmentation is enabled, skip this step; timelines change after LLM resegmentation and cannot align with original subtitles
    def check_target_sub(self, source_srt_list: List[SrtItem], target_srt_list: List[SrtItem]) -> List[SrtItem]:
        if not source_srt_list:
            return []
        _source = copy.deepcopy(source_srt_list)
        if not target_srt_list:
            for it in _source:
                it['text'] = ''
            return _source

        source_len = len(source_srt_list)
        target_len = len(target_srt_list)
        if source_len == target_len:
            logger.debug(f'原始语言字幕和目标语言字幕行数一致，均为 {source_len=}')
            for idx, it in enumerate(_source):
                it['text'] = target_srt_list[idx].get('text', '')
            return _source

        logger.warning(f'翻译结果行数{target_len}，原始字幕行数{source_len}，不一致,根据原始字幕时间轴获取对应目标字幕文本')
        used: set[int] = set()
        exact_map: dict[str, list[int]] = {}
        target_bounds: list[tuple[int, int] | None] = []
        line_map: dict[object, list[int]] = {}
        for t_idx, tgt in enumerate(target_srt_list):
            norm_t = self._norm_srt_time(str(tgt.get('time', '')))
            if norm_t:
                exact_map.setdefault(norm_t, []).append(t_idx)
            target_bounds.append(self._parse_time_bounds_ms(tgt))
            t_line = tgt.get('line')
            if t_line is not None:
                line_map.setdefault(t_line, []).append(t_idx)

        matched_texts: list[str | None] = [None] * source_len
        # Pass 1: exact normalized timestamp match
        for s_idx, src in enumerate(_source):
            norm_s = self._norm_srt_time(str(src.get('time', '')))
            candidates = exact_map.get(norm_s) or []
            for t_idx in candidates:
                if t_idx not in used:
                    used.add(t_idx)
                    matched_texts[s_idx] = target_srt_list[t_idx].get('text', '')
                    break

        # Pass 2: tolerant timestamp match (<= 50ms drift)
        for s_idx, src in enumerate(_source):
            if matched_texts[s_idx] is not None:
                continue
            s_bounds = self._parse_time_bounds_ms(src)
            if s_bounds is None:
                continue
            s_start, s_end = s_bounds
            best_t_idx: int | None = None
            best_diff = 51
            for t_idx, t_bounds in enumerate(target_bounds):
                if t_idx in used or t_bounds is None:
                    continue
                diff = max(abs(s_start - t_bounds[0]), abs(s_end - t_bounds[1]))
                if diff <= 50 and diff < best_diff:
                    best_diff = diff
                    best_t_idx = t_idx
            if best_t_idx is not None:
                used.add(best_t_idx)
                matched_texts[s_idx] = target_srt_list[best_t_idx].get('text', '')

        # Pass 3: line-number fallback for remaining unmatched items
        for s_idx, src in enumerate(_source):
            if matched_texts[s_idx] is not None:
                continue
            s_line = src.get('line')
            for t_idx in (line_map.get(s_line) or []):
                if t_idx not in used:
                    used.add(t_idx)
                    matched_texts[s_idx] = target_srt_list[t_idx].get('text', '')
                    break

        for s_idx, it in enumerate(_source):
            it['text'] = matched_texts[s_idx] if matched_texts[s_idx] is not None else ''
        return _source

    # Manually set as ended, on successful completion or error
    def set_end(self, succeed=False):
        self.hasend = True
        if succeed:
            self.precent = 100
            if self.cancellation_token is None and self.uuid in app_cfg.stoped_uuid_set:
                return
            self.signal(text=f"{self.cfg.name}", type='succeed')
            if self.event_sink is None:
                if app_cfg.exec_mode=="cli":
                    print(f'Save to:[ {self.cfg.target_dir} ]')
                else:
                    from videotrans.util.help_ffmpeg import send_notification
                    send_notification(tr('Succeed'), f"{self.cfg.basename}")
            # Clean up temporary files
            try:
                if self.cfg.cache_folder:
                    shutil.rmtree(self.cfg.cache_folder, ignore_errors=True)
            except Exception as e:
                logger.exception(f'任务结束后清理临时文件失败，跳过,{e}:{self.cfg.cache_folder=}', exc_info=True)
        if self.cancellation_token is None:
            app_cfg.stoped_uuid_set.add(self.uuid)
