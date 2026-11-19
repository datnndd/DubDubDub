import base64
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from videotrans.configure.config import tr, settings, app_cfg, logger, TEMP_ROOT
from videotrans.util.help_misc import set_proxy,vail_file

@dataclass
class BaseCon:
    # Unique uuid for each task
    uuid: Optional[str] = field(default=None, init=False)
    # Used for components needing direct proxy string
    proxy_str: str = ''
    last_down_time:int=0
    event_sink: Optional[Callable[[dict], None]] = field(default=None, repr=False, compare=False)
    cancellation_token: object = field(default=None, repr=False, compare=False)


    def __post_init__(self):
        # Get proxy
        self.proxy_str = self._set_proxy(type='set')

    def _exit(self) -> bool:
        if app_cfg.exit_soft:
            return True
        if self.cancellation_token is not None:
            return self.cancellation_token.is_cancelled()
        if self.uuid and self.uuid in app_cfg.stoped_uuid_set:
            return True
        return False
    # All window and task information interacts via queue
    def signal(self, **kwargs):
        if self.event_sink:
            if 'uuid' not in kwargs or not kwargs.get('uuid'):
                kwargs['uuid'] = self.uuid
            if 'type' not in kwargs or not kwargs.get('type'):
                kwargs['type'] = 'logs'
            self.event_sink(kwargs)
            return
        if app_cfg.exit_soft: return
        if app_cfg.exec_mode=='cli':
            print(kwargs.get('text'))
            return
        logger.info(str(kwargs.get('text') or ''))

    def _process_callback(self, data):
        _t=time.time()
        if _t-self.last_down_time<1:
            return
        self.last_down_time=_t
        
        if isinstance(data, str):
            return self.signal(text=tr('Downloading please wait') + data)
        if not isinstance(data, dict):
            return
        msg_type = data.get("type")
        percent = data.get("percent")
        filename = data.get("filename")

        if msg_type == "file":
            self.signal(text=f"{tr('Downloading please wait')} {filename} {percent:.2f}%")
        else:
            current_file_idx = data.get("current")
            total_files = data.get("total")

            self.signal(text=f" {tr('Downloading please wait')} {current_file_idx}/{total_files} files")

    # Set and get proxy
    def _set_proxy(self, type='set'):
        if type == 'del':
            os.environ['bak_proxy'] = app_cfg.proxy or os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
            app_cfg.proxy = ''
            os.environ.pop('HTTPS_PROXY', None)
            os.environ.pop('HTTP_PROXY', None)
            return None

        if type == 'set':
            raw_proxy = app_cfg.proxy or os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
            if raw_proxy:
                app_cfg.proxy = raw_proxy
                return raw_proxy
            if not raw_proxy:

                proxy = set_proxy() or os.environ.get('bak_proxy')
                if proxy:
                    os.environ['HTTP_PROXY'] = proxy
                    os.environ['HTTPS_PROXY'] = proxy
                app_cfg.proxy = proxy
                return proxy
        return None


    # Uniformly convert to wav audio after speech synthesis for subsequent speed change and processing
    def convert_to_wav(self, mp3_file_path: str, output_wav_file_path: str, extra=None):
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
            "pcm_s16le"
        ]
        if extra:
            cmd += extra
        cmd += [
            output_wav_file_path
        ]
        try:
            from videotrans.util.help_ffmpeg import runffmpeg,remove_silence_wav
            runffmpeg(cmd, force_cpu=True)
            if settings.get('remove_dubb_silence', True):
                remove_silence_wav(output_wav_file_path)
        except Exception as e:
            logger.exception(f'Failed to convert to 48k wav, skipping: {e}', exc_info=True)
            return False
        return True


    def _base64_to_audio(self, encoded_str: str, output_path: str) -> None:
        if not encoded_str:
            raise ValueError("Base64 encoded string is empty.")
        from videotrans.util.help_ffmpeg import runffmpeg
        # If data prefix exists, save as converted format according to the audio format in prefix
        if encoded_str.startswith('data:audio/'):
            output_ext = Path(output_path).suffix.lower()[1:]
            mime_type, encoded_str = encoded_str.split(',', 1)  # Extract Base64 data portion
            # Extract audio format (e.g., 'mp3', 'wav')
            audio_format = mime_type.split('/')[1].split(';')[0].lower()
            support_format = {
                "mpeg": "mp3",
                "wav": "wav",
                "ogg": "ogg",
                "aac": "aac"
            }
            base64data_ext = support_format.get(audio_format, "")
            if base64data_ext and base64data_ext != output_ext:
                # Different formats require format conversion
                # Decode base64 encoded string to bytes
                wav_bytes = base64.b64decode(encoded_str)
                # Write decoded bytes to file
                with open(output_path + f'.{base64data_ext}', "wb") as wav_file:
                    wav_file.write(wav_bytes)

                runffmpeg([
                    "-y", "-i", output_path + f'.{base64data_ext}', "-b:a", "128k", output_path
                ])
                return
        # Decode base64 encoded string to bytes
        wav_bytes = base64.b64decode(encoded_str)
        # Write decoded bytes to file
        with open(output_path, "wb") as wav_file:
            wav_file.write(wav_bytes)

    def _audio_to_base64(self, file_path: str):
        if not file_path or not Path(file_path).exists():
            return None
        with open(file_path, "rb") as wav_file:
            wav_content = wav_file.read()
            base64_encoded = base64.b64encode(wav_content)
            return base64_encoded.decode("utf-8")

    def _signal_of_process(self, logs_file,status_dict=None):
        last_mtime = 0
        timeout = 0
        while 1:
            if self._exit(): return
            if status_dict and status_dict['is_end']:
                return
            timeout += 1
            if timeout > 3600:
                logger.warning(f'Child process exceeded 3600s without finishing, possible error: {logs_file}')
                return
            _p = Path(logs_file)
            # File already removed
            if last_mtime > 0 and not _p.exists():
                return
            try:
                if not _p.exists():
                    time.sleep(1)
                    continue
                # Get last modified time of log file
                _mtime = _p.stat().st_mtime
                if _mtime == last_mtime:
                    # Unmodified since last check
                    time.sleep(1)
                    continue
                last_mtime = _mtime
                timeout=0
                _content = _p.read_text(encoding='utf-8')
                if not _content:
                    time.sleep(1)
                    continue
                _tmp = json.loads(_content)
                if _tmp.get('type', '') == 'error':
                    return
                self.signal(text=_tmp.get('text', ''), type=_tmp.get('type', 'logs'))
            except Exception:
                # Reading temporary inter-process log file may fail if cleaned up; safe to ignore
                logger.warning(f'Error reading inter-process temporary file, may have been cleaned up: {logs_file}')
            time.sleep(1)

    # Execute task using a separate process
    def _new_process(self, callback=None, title="", is_cuda=False, kwargs=None):
        _st = time.time()
        from .excepts import VideoTransError,SttTimeoutError
        from concurrent.futures.process import BrokenProcessPool
        from videotrans.process.process_manager import GlobalProcessManager
        kwargs = kwargs or {}
        self.signal(text=f'[{title}] starting...')
        logger.debug(f'[Child process task start: {title=}]')

        # Submit task and pass parameters explicitly so the child process receives correct arguments
        logs_file = kwargs.get('logs_file',f'{TEMP_ROOT}/{_st}.log')
        device_index = 0
        status_dict={"is_end":False}
        try:
            Path(logs_file).touch()
            threading.Thread(target=self._signal_of_process, args=(logs_file,status_dict), daemon=True).start()
            # Check whether CUDA is valid again to prevent earlier retrieval failures
            if is_cuda:
                import torch
                if not torch.cuda.is_available():
                    is_cuda = False

            # If using GPU, obtain available device_index
            if is_cuda:
                # Multi-GPU mode enabled
                if settings.get('multi_gpus'):
                    from videotrans.util.gpus import get_cudaX
                    device_index = get_cudaX()
                if device_index == -1:
                    is_cuda = False
                    kwargs['is_cuda'] = False
                    logger.error(f'CUDA enabled but no available GPU detected, falling back to CPU')
                kwargs['device_index'] = max(device_index, 0)
            logger.debug(f'Child process task parameters: {kwargs=}')
            future = GlobalProcessManager.submit_task_cpu(
                callback,
                **kwargs
            ) if not is_cuda else GlobalProcessManager.submit_task_gpu(
                callback,
                **kwargs
            )

            _timeout=0
            while not future.done():
                if self._exit():
                    return None
                # faster-whisper may occasionally exit silently after completion;
                # pre-save transcription results to subtitle_srt before exiting to ensure continuation
                if kwargs.get('subtitle_srt') and Path(kwargs.get('subtitle_srt')).exists():
                    # Still looping after 20s of result generation; child process may have crashed
                    if _timeout>20:
                        status_dict['is_end']=True
                        logger.warning(f'faster-whisper has generated subtitles for over {_timeout}s, still looping, child process may have crashed, forcing SttTimeoutError')
                        raise SttTimeoutError("STT timeout")
                    _timeout+=1
                time.sleep(1)
            data,err = future.result(timeout=10)
            logger.debug(f'[Child process task {title=}] returned')
            status_dict['is_end']=True
            if err or not data:
                raise VideoTransError(err)
            self.signal(text=f'[{title}] end: {int(time.time() - _st)}s')
            return data
        except SttTimeoutError:
            raise
        except BrokenProcessPool as e:
            _model = ''
            _cuda = ''
            if kwargs.get('model_name'):
                _model = ' Model:' + kwargs.get('model_name')
            if is_cuda and device_index > -1:
                _cuda = f" GPU{device_index}"
            logger.exception(f'{title}: {_model}{_cuda}, {kwargs=},{e}', exc_info=True)
            raise VideoTransError(f'{_model}{_cuda} {e}')
        except BaseException as e:
            logger.exception(f'{title},{e}', exc_info=True)
            raise
        finally:
            status_dict['is_end']=True
            try:
                logger.debug(f'[Child process task finished: {title=}], elapsed: {time.time()-_st}s')
                if logs_file:
                    Path(logs_file).unlink(missing_ok=True)
            except OSError:
                pass
