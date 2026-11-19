import asyncio,os
import concurrent
import copy
import inspect
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from tenacity import RetryError
from videotrans.configure.base import BaseCon
from videotrans.configure.config import tr, settings, logger, ROOT_DIR
from videotrans.configure import config
from videotrans.util.help_misc import vail_file,pygameaudio,get_tts_type

"""Shared execution for retained TTS providers."""


@dataclass
class BaseTTS(BaseCon):
    # Dubbing channel
    tts_type: int = 2
    # Queue storing subtitle information, extended SrtItem
    queue_tts: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    # Reference audio or role dictionary
    roledict: Dict[str, Any] = field(default_factory=dict, repr=False)
    # queue_tts count
    len: int = field(init=False)
    # Language code
    language: Optional[str] = None
    # Unique uid
    uuid: Optional[str] = None
    # Whether to play immediately
    play: bool = False
    # Whether testing
    is_test: bool = False

    # Volume, speech rate and pitch use percentage/Hz string
    volume: Union[float, str] = field(default='+0%', init=False)
    rate: Union[float, str] = field(default='+0%', init=False)
    pitch: Union[float, str] = field(default='+0Hz', init=False)

    # Whether finished
    has_done: int = field(default=0, init=False)

    # Pause duration after each task (resolved dynamically in __post_init__)
    wait_sec: Optional[float] = None
    # Concurrent thread count (resolved dynamically in __post_init__)
    dub_nums: Optional[int] = None
    # Cloud provider flag for auto-scaling concurrency
    is_cloud_provider: bool = False
    # Stores message
    error: Union[str, Exception, None] = None
    # Dubbing API URL
    api_url: str = field(default='', init=False)
    # Enable CUDA (used on demand by local channels)
    is_cuda: bool = False
    local_dir: str = None
    # In single video mode, heavy local processes can wait for new tasks during dubbing proofreading
    # is_redubb is True indicates triggered from dubbing proofreading panel
    is_redubb:bool=False

    def __post_init__(self):
        super().__post_init__()
        is_cloud = bool(self.is_cloud_provider or self.tts_type in (0, 3))
        if self.dub_nums is None:
            cfg_threads = int(float(settings.get('dubbing_thread', 1) or 1))
            if is_cloud and cfg_threads <= 1:
                self.dub_nums = max(1, min(4, len(self.queue_tts)))
            else:
                self.dub_nums = max(1, cfg_threads)
        if self.wait_sec is None:
            cfg_wait = float(settings.get('dubbing_wait', 0) or 0)
            if is_cloud and cfg_wait == 1.0:
                self.wait_sec = 0.0
            else:
                self.wait_sec = max(0.0, cfg_wait)
        Path(f'{config.TEMP_DIR}/{self.uuid}').mkdir(parents=True, exist_ok=True)
        self.queue_tts = copy.deepcopy(self.queue_tts)
        self.len = len(self.queue_tts)
        self._cleantts()

    # Subclass did not override _exec() method: run() -> _exec() -> _local_mul_thread() -> _item_task() -> _run()
    # Subclass overrides _exec() method: run() -> _exec()
    def run(self) -> None:
        if self._exit(): return
        from videotrans.configure.excepts import DubbingSrtError
        _tts_name=get_tts_type(self.tts_type)
        logger.debug(f'当前使用配音渠道：{_tts_name}')
        self.signal(text=f"{_tts_name} starting: [len={self.len}]")
        loop = None
        try:
            if hasattr(self, '_download'):
                self.signal(text=tr("check or download models"))
                self._download()
                self.signal(text=tr('Dubbing'))
            # Support channels implemented as async coroutine functions
            if inspect.iscoroutinefunction(self._exec):
                try:
                    # Check whether the current thread has a running event loop
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # If the current thread already has a running loop (e.g. web framework main thread),
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(asyncio.run, self._exec())
                        future.result()
                else:
                    # If there is no running loop, directly use asyncio.run
                    asyncio.run(self._exec())
            else:
                # May invoke multithreading
                self._exec()
        except RetryError as e:
            raise e.last_attempt.exception()
        except (OSError,FileNotFoundError) as e:
            _e=str(e)
            if self.local_dir and ("no file named model.safetensors" in _e or os.path.basename(self.local_dir) in _e):
               from videotrans.configure.excepts import DownloadModelsError
               raise  DownloadModelsError(tr('model incomplete error',self.local_dir,tr('Help document')))
            raise
        except RuntimeError as e:
            logger.warning(f'TTS 线程运行时发生错误: {e}')
            if 'Event loop' in str(e):
                logger.warning("捕获到 'Event loop is closed' 错误，这通常是关闭时序问题。")
            else:
                raise


        # Play during preview or test
        if self.play:
            if vail_file(self.queue_tts[0]['filename']):
                return pygameaudio(self.queue_tts[0]['filename'])

            logger.error(f'试听配音时发生错误{self.error}')
            if isinstance(self.error, RetryError):
                raise self.error.last_attempt.exception()
            raise self.error if isinstance(self.error, Exception) else DubbingSrtError(str(self.error))

        # Record success count
        succeed_nums = 0
        for it in self.queue_tts:
            if self._exit(): return
            if not it['text'].strip() or vail_file(it['filename']):
                succeed_nums += 1
        # Only considered failure if all dubbings failed
        if succeed_nums < 1:
            if self._exit(): return
            logger.error(f'本次配音全部失败：{self.error}')
            if isinstance(self.error, Exception):
                raise self.error.last_attempt.exception() if isinstance(self.error, RetryError) else self.error

            raise DubbingSrtError(tr("Dubbing failed") + str(self.error))
        logger.debug(f'本次 {_tts_name} 配音成功 {succeed_nums} 个，失败 {self.len - succeed_nums} 个')
        self.signal(text=tr("Dubbing succeeded {}，failed {}", succeed_nums, self.len - succeed_nums))

    # If subclass does not override _exec(), this method is called by default
    # Checks if returned error is StopTask type; if so, terminates task directly
    def _local_mul_thread(self) -> None:
        if self._exit(): return
        from videotrans.configure.excepts import StopTask
        # Single subtitle line, no need for multithreading
        if len(self.queue_tts) == 1 or self.dub_nums == 1:
            logger.debug(f'设定最大配音线程: {self.dub_nums},实际 单线程配音, 待配音字幕长度: {self.len}, 配音后暂停{self.wait_sec}s')
            for k, item in enumerate(self.queue_tts):
                if self._exit(): return
                if not item.get('text', '').strip() or vail_file(item['filename']):
                    continue
                error = self._item_task(item, k)
                if error:
                    self.error = error
                    if isinstance(error, StopTask):
                        raise error

                self.signal(text=f'{tr("Dubbing")} [{k + 1}/{self.len}]')
                if self.wait_sec > 0 and k < len(self.queue_tts) - 1:
                    time.sleep(self.wait_sec)
            return

        all_task = []
        _wok=max(min(self.dub_nums, len(self.queue_tts)),2)
        pool = ThreadPoolExecutor(max_workers=_wok)
        logger.debug(f'设定配音最大线程数: {self.dub_nums},实际 {_wok} 线程配音, 待配音字幕长度: {self.len}')
        try:
            completed_tasks = 0
            for k, item in enumerate(self.queue_tts):
                if self._exit():
                    for f in all_task:
                        f.cancel()
                    return
                if not item.get('text', '').strip() or vail_file(item['filename']):
                    completed_tasks += 1
                    continue
                future = pool.submit(self._item_task, item, k)
                all_task.append(future)

            if all_task:
                for task in as_completed(all_task):
                    if self._exit():
                        for f in all_task:
                            f.cancel()
                        return
                    error = task.result()
                    if error:
                        self.error = error
                        if isinstance(error, StopTask):
                            for f in all_task:
                                f.cancel()
                            raise error
                    completed_tasks += 1
                    self.signal(text=f"{tr('Dubbing')}: [{completed_tasks}/{self.len}] ...")
            self.signal(text=f"TTS ended ...")
        finally:
            for f in all_task:
                f.cancel()
            pool.shutdown(wait=False)

    # Called by run(); subclasses may override to implement batch dubbing for entire queue_tts
    # If not overridden, enters multithreading, calling _item_task() per subtitle; subclass must implement _run()
    def _exec(self) -> None:
        self._local_mul_thread()

    # Each subtitle task, called in multiple threads by _local_mul_thread
    # data_item is each element in queue_tts
    # If subclass does not override _exec(), it must implement _run()
    # return: None on success; returns error message or raises exception on failure
    def _item_task(self, data_item: Union[Dict, List, None], idx: int = -1) -> Union[str, Exception, None]:
        if self._exit() or not data_item.get('text', '').strip() or vail_file(data_item.get('filename')):
            return
        # For unrecoverable errors like 404, invalid SK, unauthorized, send error signal directly without continuing other threads
        try:
            self.signal(text=f'{tr("Dubbing")} {idx}/{self.len}')
            return self._run(data_item,idx)
        except RetryError as e:
            logger.exception(f'\n第{idx}条字幕配音失败,字幕文本:{data_item}\n{e}', exc_info=True)
            return e.last_attempt.exception()
        except Exception as e:
            logger.exception(f'\n第{idx}条字幕配音失败,字幕文本:{data_item}\n{e}', exc_info=True)
            return e

    # If subclass did not override _exec, it must implement this method
    def _run(self, data_item: Union[Dict, List, None], idx: int = -1) -> Union[str, None]:
        raise NotImplemented

    # Text normalization and parameter cleanup (volume, etc.)
    def _cleantts(self) -> None:
        if not self.queue_tts:
            return
        normalizer = None
        lang_prefix = (self.language or "")[:2].lower()
        if settings.get('normal_text'):
            if lang_prefix == 'zh':
                from videotrans.util.cn_tn import TextNorm
                normalizer = TextNorm(to_banjiao=True)
            elif lang_prefix == 'en':
                from videotrans.util.en_tn import EnglishNormalizer
                normalizer = EnglishNormalizer()

        for i, it in enumerate(self.queue_tts):
            if it['text'].strip() and normalizer:
                try:
                    self.queue_tts[i]['text'] = normalizer(it['text'])
                except Exception as e:
                    logger.exception(f'文本规范化失败，忽略:{it["text"]=},{e}', exc_info=True)

        volume = str(self.queue_tts[0].get('volume') or '+0%')
        volume = f'+{volume}' if re.match(r'^\d+(\.\d+)?%$', volume) else volume
        self.volume = '+0%' if not re.match(r'^[+-]\d+(\.\d+)?%$', volume) else volume

        rate = str(self.queue_tts[0].get('rate') or '+0%')
        rate = f'+{rate}' if re.match(r'^\d+(\.\d+)?%$', rate) else rate
        self.rate = '+0%' if not re.match(r'^[+-]\d+(\.\d+)?%$', rate) else rate

        pitch = str(self.queue_tts[0].get('pitch') or '+0Hz').replace('hz', 'Hz')
        pitch = f'+{pitch}' if re.match(r'^\d+(\.\d+)?Hz$', pitch, re.I) else pitch
        self.pitch = '+0Hz' if not re.match(r'^[+-]\d+(\.\d+)?Hz$', pitch, flags=re.I) else pitch

        logger.debug(f'{self.volume=}, {self.rate=}, {self.pitch=}')

    # Convert percentage rate/volume to decimal float
    def get_speed(self) -> float:
        speed = 1.0
        try:
            speed = round(1 + float(self.rate.replace('%', '')) / 100, 1)
        except (TypeError,ValueError):
            pass
        return speed

    def get_volume(self) -> float:
        volume = 1.0
        try:
            volume = round(1 + float(self.volume.replace('%', '')) / 100, 1)
        except (TypeError,ValueError):
            pass
        return volume

    def get_pitch(self) -> float:
        pitch = 1.0
        try:
            pitch = round(float(re.sub(r'[hz%]', '', self.pitch,flags=re.I)), 1)
        except (TypeError,ValueError):
            pass
        return pitch

    # Return reference audio and reference text
    def get_ref_wav(self, item) -> Tuple[str, str]:
        role = item['role']
        ref_wav, ref_text = None, None
        if role == 'clone':
            ref_wav = item.get('ref_wav', '')
            ref_text = item.get('ref_text').strip()
        elif role in self.roledict:
            if not isinstance(self.roledict[role],dict):
                return ref_wav,ref_text
            ref_text = self.roledict[role]['ref_text']
            ref_wav = ROOT_DIR + f"/f5-tts/{role}"

        if not ref_wav or not Path(ref_wav).exists():
            raise RuntimeError(tr('The role {} does not exist', role))
        return ref_wav, ref_text

    def convert_to_wav(self, mp3_file_path: str, output_wav_file_path: str, extra=None):
        """Convert TTS output to 48kHz stereo PCM_16 WAV and trim leading/trailing silence fast via NumPy."""
        if self._exit() or not vail_file(mp3_file_path):
            return
        cmd = [
            "-y",
            "-i",
            mp3_file_path,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
        ]
        if extra:
            cmd += extra
        cmd.append(output_wav_file_path)
        try:
            from videotrans.util.help_ffmpeg import runffmpeg, remove_silence_wav
            runffmpeg(cmd, force_cpu=True)
            if settings.get('remove_dubb_silence', True) and Path(output_wav_file_path).is_file():
                try:
                    import math
                    import numpy as np
                    import soundfile as sf

                    data, sr = sf.read(output_wav_file_path, dtype="float32", always_2d=False)
                    if sr > 0 and data.size > 0:
                        mono = np.mean(data, axis=1) if data.ndim > 1 else data
                        rms = float(np.sqrt(np.mean(mono * mono)))
                        dbfs = 20.0 * math.log10(rms) if rms > 1e-10 else -float("inf")
                        if not math.isfinite(dbfs):
                            dbfs = -50.0
                        silence_thresh_db = min(-55.0, dbfs - 20.0)
                        thresh_amp = 10.0 ** (silence_thresh_db / 20.0)

                        step = max(1, int(sr * 0.010))  # 10ms frames
                        min_sil_frames = max(1, int(round(200 / 10)))
                        n_frames = len(mono) // step
                        if n_frames > 0:
                            trimmed_view = mono[: n_frames * step].reshape(n_frames, step)
                            frame_rms = np.sqrt(np.mean(trimmed_view * trimmed_view, axis=1))
                            is_nonsilent = frame_rms >= thresh_amp
                            if np.any(is_nonsilent):
                                first_ns = int(np.argmax(is_nonsilent))
                                last_ns = int(len(is_nonsilent) - 1 - np.argmax(is_nonsilent[::-1]))
                                # Match pydub detect_nonsilent semantics: only trim if silence >= 200ms
                                raw_start_ms = first_ns * 10 if first_ns >= min_sil_frames else 0
                                raw_end_ms = (last_ns + 1) * 10 if (n_frames - 1 - last_ns) >= min_sil_frames else int(len(data) * 1000 / sr)
                                total_ms = int(len(data) * 1000 / sr)
                                start_trim_ms = max(0, raw_start_ms - 80)
                                end_trim_ms = min(total_ms, raw_end_ms + 600)
                                if start_trim_ms > 0 or end_trim_ms < total_ms:
                                    s_idx = int(start_trim_ms * sr / 1000)
                                    e_idx = max(s_idx + 1, int(end_trim_ms * sr / 1000))
                                    tmp_trim = Path(output_wav_file_path).with_suffix(".trim.wav")
                                    try:
                                        sf.write(str(tmp_trim), data[s_idx:e_idx], sr, subtype="PCM_16", format="WAV")
                                        os.replace(tmp_trim, output_wav_file_path)
                                    finally:
                                        tmp_trim.unlink(missing_ok=True)
                except Exception:
                    remove_silence_wav(output_wav_file_path)
        except Exception as e:
            logger.exception(f'Failed to convert to 48k wav, skipping: {e}', exc_info=True)
            return False
        return True
