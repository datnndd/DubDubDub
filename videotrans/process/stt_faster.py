# Speech recognition executed in a separate process
# Returns tuple:
# Failure: first value is False, second value stores failure reason
# Success: first value has desired return value or True, second value is None
import json, traceback
from pathlib import Path
from typing import List, Tuple, Union
from videotrans.task.taskcfg import SrtItem
from videotrans.configure.config import logger



def faster_whisper(
        *,
        prompt=None,
        detect_language=None,
        model_name=None,
        logs_file=None,
        is_cuda=False,
        no_speech_threshold=0.5,
        condition_on_previous_text=False,
        speech_timestamps=None,
        audio_file=None,
        local_dir=None,
        compute_type="default",
        beam_size=5,
        best_of=5,
        jianfan=False,
        audio_duration=0,
        temperature=None,
        hotwords=None,
        repetition_penalty=1.0,
        compression_ratio_threshold=2.2,
        device_index=0,  # GPU index
        max_speech_ms=6000,
        subtitle_srt=None
) -> Tuple[Union[List[SrtItem], bool], Union[str, None]]:
    import zhconv
    from videotrans.process._stt_utils import _write_log, _resegment
    from videotrans.util._srt_parse import ms_to_time_string
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    raws = []
    if detect_language == 'fil':
        detect_language = 'tl'

    def _create_model(_compute_type):
        try:
            logger.debug(f'[faster_whisper] Loading model {model_name}: {is_cuda=},{_compute_type=}')
            model = WhisperModel(
                local_dir,
                device="cuda" if is_cuda else 'cpu',
                device_index=device_index if is_cuda else 0,
                compute_type=_compute_type
            )
            return model
        except Exception as e:
            # Retry on data type errors
            # In CUDA mode, first try float16
            if is_cuda and _compute_type != 'float16':
                logger.warning(f'faster-whisper CUDA model loading failed, retrying with [float16]: {e}')
                return _create_model('float16')


            # If CPU mode and not int8, first try int8
            if not is_cuda and _compute_type != 'int8':
                logger.warning(f'faster-whisper CPU model loading failed, retrying with [int8]: {e}')
                return _create_model('int8')
            # Fallback to float32
            if _compute_type != 'float32':
                logger.warning(f'faster-whisper model loading failed, retrying with [float32], {is_cuda=}')
                return _create_model('float32')
            raise

    try:
        if speech_timestamps and isinstance(speech_timestamps, str):
            speech_timestamps = json.loads(Path(speech_timestamps).read_text(encoding='utf-8'))
        last_end_time = audio_duration / 1000.0 if audio_duration > 0 else (speech_timestamps[-1][1] / 1000.0 if speech_timestamps else 0)

        try:
            # 1. Load base model
            _write_log(logs_file, json.dumps({"type": "logs", "text": 'loading model'}))
            logger.debug(f'Loading faster-whisper model {model_name}, compute_type: {compute_type}')
            model = _create_model(compute_type)
        except Exception as e:
            error = traceback.format_exc()
            logger.error(f'[faster_whisper][{is_cuda=}] Speech transcription model loading failed: {local_dir=}\n{error}')
            return False, f'{e},{error}'

        if not temperature:
            temperature = [
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ]
        elif str(temperature).startswith('[') or str(temperature).startswith('('):
            temperature = [float(i) for i in str(temperature)[1:-1].split(',')]
        else:
            temperature = float(temperature)

        if speech_timestamps:

            _write_log(logs_file, json.dumps({"type": "logs", "text": 'Transcribe batch...'}))
            logger.debug(f'After pre-VAD processing, passing timestamps to BatchedInferencePipeline, batch_size=4')
            # 4. Execute batched inference
            # Using batched_model.transcribe
            batched_model = BatchedInferencePipeline(model=model)

            # 3. Convert timestamp format
            # BatchedInferencePipeline requires [{'start': start_sec, 'end': end_sec}, ...]
            clip_timestamps_dicts = [
                {"start": it[0] / 1000.0, "end": it[1] / 1000.0}
                for it in speech_timestamps
            ]
            segments, info = batched_model.transcribe(
                audio_file,
                batch_size=4,  #
                beam_size=beam_size,
                best_of=best_of,
                no_speech_threshold=no_speech_threshold,
                # vad_filter must be False, otherwise clip_timestamps may be ignored or conflict
                vad_filter=False,
                clip_timestamps=clip_timestamps_dicts,  # Custom segments
                condition_on_previous_text=condition_on_previous_text,
                word_timestamps=False,
                without_timestamps=True,
                temperature=temperature,
                hotwords=hotwords,
                repetition_penalty=repetition_penalty,
                compression_ratio_threshold=compression_ratio_threshold,
                language=detect_language.split('-')[0] if detect_language and detect_language != 'auto' else None,
                initial_prompt=prompt if prompt else None
            )
            i = 0
            logger.debug(f'faster-whisper模式下，预先使用VAD分割音频，对{model_name}模型返回的文字结果直接使用')
            for segment in segments:
                if segment.end > last_end_time:
                    continue
                text = segment.text
                if not text.strip():
                    continue
                i += 1
                s, e = int(segment.start * 1000), int(segment.end * 1000)
                if jianfan:
                    text = zhconv.convert(text, 'zh-hans')
                tmp = SrtItem(**{
                    'text': text,
                    'start_time': s,
                    'end_time': e
                })
                tmp['startraw'] = ms_to_time_string(ms=tmp['start_time'])
                tmp['endraw'] = ms_to_time_string(ms=tmp['end_time'])
                tmp['time'] = f"{tmp['startraw']} --> {tmp['endraw']}"
                raws.append(tmp)
                _write_log(logs_file, json.dumps({"type": "subtitle", "text": f'[{i}] {text}\n'}))
        else:
            logger.debug(f'直接传递完整音频，由faster-whisper内部VAD处理，返回字级时间戳数据')
            _write_log(logs_file, json.dumps({"type": "logs", "text": 'Transcribe word timestamps'}))
            segments, info = model.transcribe(
                audio_file,
                beam_size=beam_size,
                best_of=best_of,
                condition_on_previous_text=condition_on_previous_text,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=140, min_speech_duration_ms=0),
                no_speech_threshold=no_speech_threshold,
                # clip_timestamps="0",  # clip_timestamps,
                word_timestamps=True,
                # without_timestamps=False,
                temperature=temperature,
                hotwords=hotwords,
                repetition_penalty=repetition_penalty,
                compression_ratio_threshold=compression_ratio_threshold,
                language=detect_language.split('-')[0] if detect_language and detect_language != 'auto' else None,
                initial_prompt=prompt if prompt else None
            )
            texts = []
            i = 0
            for segment in segments:
                i += 1
                texts.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "words": [{'word': it.word, 'start': it.start, 'end': it.end} for it in segment.words]
                })
                _write_log(logs_file, json.dumps({"type": "subtitle", "text": f'[{i}] {segment.text}\n'}))

            logger.debug(f'faster-whisper模式下，对{model_name}模型返回的字级时间戳进行断句')
            if not texts:
                logger.error(f'no texts:{info=}\n{segments=}')
                return False, f"No transcription results returned. Please check the original audio/video or model and try again.\n{info=}"
            raws = _resegment(texts, info.language, max_speech_ms, logs_file)
            if jianfan and raws:
                for it in raws:
                    it['text'] = zhconv.convert(it['text'], 'zh-hans')
            logger.debug('Resegmentation complete, returning results')
        # Save recognition results to temporary directory to prevent deadlock if child process crashes
        if subtitle_srt:
            Path(subtitle_srt).parent.mkdir(parents=True, exist_ok=True)
            Path(subtitle_srt).write_text("\n\n".join([f'{i+1}\n{it.startraw} --> {it.endraw}\n{it.text}' for i,it in enumerate(raws)]),encoding="utf-8")
            logger.debug(f'faster-whisper saved temporary results to {subtitle_srt} to prevent subprocess hang')
        
        return raws,None
    except BaseException as e:
        msg = traceback.format_exc()
        logger.exception(e,exc_info=True)
        return False, f'{e}:{msg}'
