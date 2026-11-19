# Text-to-Speech synthesis executed in a separate process
# Returns tuple:
# Failure: first value is False, second value stores failure reason
# Success: first value has desired return value or True, second value is None
import time
from pathlib import Path
import traceback, json, os
from typing import Dict, Optional, Tuple, Union
from videotrans.configure.config import logger, ROOT_DIR, REDUBB_QUEUE_FILE, REDUBB_STATUS_FILE
from ._utils import _write_log, convert_to_wav


def omnivoice_fun(
        queue_tts_file=None, # Dubbing data stored in json file, loaded via path
        logs_file=None,
        is_cuda=False,
        speed=1.0,
        device_index=0, # GPU index
        is_redubb=False # Whether in single video proofreading dubbing workflow
) -> Tuple[bool, Union[str, None]]:
    from videotrans.util.help_role import get_omnivoice_voice_info
    from videotrans.util.help_misc import vail_file
    import torch
    from omnivoice import OmniVoice
    from videotrans.util import gpus
    import soundfile as sf

    model = OmniVoice.from_pretrained(
        f'{ROOT_DIR}/models/models--k2-fsa--OmniVoice',
        device_map=f'cuda:{device_index}' if is_cuda else gpus.mps_or_cpu(),
        dtype=torch.float32 if not is_cuda else torch.float16
    )
    logger.debug(f'OmniVoice-TTS本地内置渠道，{is_cuda=}')
    role_cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

    try:
        if is_redubb:
            queue_tts_file = REDUBB_QUEUE_FILE
        while 1:
            if is_redubb and Path(REDUBB_STATUS_FILE).exists():
                return True, None
            queue_tts = json.loads(Path(queue_tts_file).read_text(encoding='utf-8'))
            _len = len(queue_tts)
            ok, err = 0, 0
            last_error = ''
            for i, it in enumerate(queue_tts):
                if is_redubb and Path(REDUBB_STATUS_FILE).exists():
                    return True, None
                output_filename = it.get('filename', '') + "-24k.wav"
                if vail_file(it.get('filename', '')):
                    ok += 1
                    continue
                text = it.get('text')
                if not text:
                    err += 1
                    last_error = "No text for dubbing"
                    continue
                role = it.get('role')
                _write_log(logs_file, json.dumps({"type": "logs", "text": f'{i+1}/{_len} {role}'}))
                if role == 'clone':
                    wavfile = it.get('ref_wav', '')
                    ref_text = it.get('ref_text', '')
                else:
                    role_key = str(role or "").strip()
                    if role_key not in role_cache:
                        role_cache[role_key] = get_omnivoice_voice_info(role)
                    wavfile, ref_text = role_cache[role_key]

                if not wavfile or not Path(wavfile).is_file():
                    # Still does not exist, reference audio unavailable
                    msg = f"No ref_audio: {role=},{wavfile=}"
                    _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
                    err += 1
                    last_error = msg
                    continue
                wav = model.generate(
                    text=it['text'],
                    ref_audio=wavfile,
                    ref_text=ref_text,
                    speed=speed
                )
                if not wav or len(wav) == 0:
                    err += 1
                    last_error = "OmniVoice returned empty audio"
                    continue
                wav_item = wav[0]
                if hasattr(wav_item, "detach"):
                    wav_item = wav_item.detach().cpu().numpy()
                sf.write(output_filename, wav_item, 24000)
                if not vail_file(output_filename):
                    err += 1
                    continue

                ok += 1
                if is_redubb:
                    try:
                        convert_to_wav(output_filename, it['filename'])
                    finally:
                        Path(output_filename).unlink(missing_ok=True)

            # Re-dubbing mode, continue polling
            if is_redubb:
                time.sleep(1)
                continue
            break
        if ok < 1:
            logger.error(f'配音全部失败：{last_error}')
            return False, "Dubbing failed" + last_error
        logger.debug(f'配音成功{ok}个，失败{err}个')
        _write_log(logs_file, json.dumps({"type": "logs", "text": f'{ok=},{err=} {last_error}'}))
        return True, None
    except BaseException as e:
        msg = traceback.format_exc()
        logger.error(msg)
        return False, f'{e},{msg}'
