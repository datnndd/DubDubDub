# Speech recognition executed in a separate process
# Returns tuple:
# Failure: first value is False, second value stores failure reason
# Success: first value has desired return value or True, second value is None
import json, traceback
from pathlib import Path
from typing import List, Optional

from videotrans.configure.config import logger, ROOT_DIR


QWEN3_ASR_SUPPORTED_LANGUAGES = (
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch",
    "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian",
    "Greek", "Romanian", "Hungarian", "Macedonian",
)
_QWEN3_ASR_LANG_MAP = {lang.lower(): lang for lang in QWEN3_ASR_SUPPORTED_LANGUAGES}
_QWEN3_ASR_OVERRIDES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "chinese": "Chinese",
    "simplified chinese": "Chinese",
    "traditional chinese": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fa": "Persian",
    "persian": "Persian",
    "western persian": "Persian",
    "fil": "Filipino",
    "tl": "Filipino",
    "filipino": "Filipino",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


def _resolve_qwen_asr_language(detect_language: Optional[str]) -> Optional[str]:
    """Map repo language code/name to Qwen3-ASR's 30 canonical language names, or None for auto."""
    if not detect_language:
        return None
    code = str(detect_language).strip().lower().replace("_", "-")
    if code in ("", "auto", "-", "no", "none"):
        return None
    if code in _QWEN3_ASR_OVERRIDES:
        return _QWEN3_ASR_OVERRIDES[code]
    if code in _QWEN3_ASR_LANG_MAP:
        return _QWEN3_ASR_LANG_MAP[code]

    try:
        from videotrans.translator._lang_utils import get_language_qwen
    except Exception:
        get_language_qwen = None

    candidates = [code]
    if "-" in code:
        candidates.append(code.split("-", 1)[0])

    for cand in candidates:
        if cand in _QWEN3_ASR_OVERRIDES:
            return _QWEN3_ASR_OVERRIDES[cand]
        if cand in _QWEN3_ASR_LANG_MAP:
            return _QWEN3_ASR_LANG_MAP[cand]
        if get_language_qwen is not None:
            mapped = get_language_qwen(cand)
            if mapped:
                mapped_low = str(mapped).strip().lower()
                if mapped_low in _QWEN3_ASR_OVERRIDES:
                    return _QWEN3_ASR_OVERRIDES[mapped_low]
                if mapped_low in _QWEN3_ASR_LANG_MAP:
                    return _QWEN3_ASR_LANG_MAP[mapped_low]

    logger.debug(f"[Qwen3-ASR] Language '{detect_language}' not in Qwen3-ASR 30-language allowlist; falling back to auto-detect (None)")
    return None


def _load_qwen3_asr_model():
    from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel



# Supports hotwords
def qwen3asr_fun(
        cut_audio_list=None,
        logs_file=None,
        is_cuda=False,
        audio_file=None,
        model_name="1.7B",
        device_index=0,  # GPU index
        hotword=None,
        detect_language=None,
):
    import torch
    from videotrans.task.taskcfg import SrtItem
    from videotrans.process._stt_utils import _write_log

    Qwen3ASRModel = _load_qwen3_asr_model()

    if is_cuda:
        device_map = f'cuda:{device_index}'
        dtype = torch.float16
    else:
        device_map = 'cpu'
        dtype = torch.float32

    qwen_lang = _resolve_qwen_asr_language(detect_language)
    logger.debug(f'QwenASR本地渠道 {model_name} 模型，{device_map=}, {detect_language=}, {qwen_lang=}')
    try:
        """
        之所以未使用 Qwen/Qwen3-ForcedAligner-0.6B 返回字级时间戳，长语音例如1个小时、2个小时可能OOM，为避免需提前裁切
        1. 如果不使用VAD而是固定时间裁切，可能断在句子中间，影响效果
        2. 若VAD裁切，既然都用VAD了，干脆直接裁切为合适长度语句了，无需再对齐
        3. 返回的词级时间戳无标点符号，需要进一步根据静音区间等断句，可能产生过长过短的字幕
        """
        _write_log(logs_file, json.dumps({"type": "logs", "text": f'Load Qwen3ASR on {device_map}'}))
        model = Qwen3ASRModel.from_pretrained(
            f"{ROOT_DIR}/models/models--Qwen--Qwen3-ASR-{model_name}",
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=8,
            # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=2048,  # Maximum number of tokens to generate. Set a larger value for long audio input.
        )
        srts: List[SrtItem] = [SrtItem(**item) for item in json.loads(Path(cut_audio_list).read_text(encoding='utf-8'))]

        srts_chunk = [srts[i:i + 8] for i in range(0, len(srts), 8)]
        for i, it_list in enumerate(srts_chunk):
            results = model.transcribe(
                audio=[it['filename'] for it in it_list],
                language=[qwen_lang for _ in it_list],
                return_time_stamps=False,
                # context=hotword.split(',') if hotword else []
            )
            for j, it in enumerate(it_list):
                it['text'] = results[j].text
            srts_chunk[i] = it_list
            _write_log(logs_file, json.dumps({"type": "subtitle", "text": "\n".join([it['text'] for it in it_list])}))

        return srts, None
    except BaseException as e:
        msg = traceback.format_exc()
        return False, f'{e}:{msg}'
